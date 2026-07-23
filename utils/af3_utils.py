import os
import glob
import json
import math
import subprocess

from Bio.PDB import MMCIFParser
from Bio.PDB.vectors import calc_dihedral

cif_parser = MMCIFParser(QUIET=True)

def build_af3_json(name, protein_seq, ligand_ccd_code=None, ligand_smiles=None,
                    protein_chain_id='A', ligand_chain_id='B', model_seeds=[1],
                    user_ccd_path=None):
    '''
    Build an AF3 cofold input record for one designed monomer + its native ligand

    name             - design ID, used as both the json's "name" field and the output filename
    protein_seq      - the designed protein sequence to cofold
    ligand_ccd_code  - PDB Chemical Component Dictionary code for the ligand (e.g. 'RET').
                        Preferred over ligand_smiles when available: gives the correct template
                        geometry/connectivity and standard atom names in AF3's output, which
                        get_ligand_dihedral/select_best_af3_sample depend on.
    ligand_smiles    - SMILES string of the ligand, used only if ligand_ccd_code is not given
    protein_chain_id - chain ID to assign the protein (default='A')
    ligand_chain_id  - chain ID to assign the ligand (default='B')
    model_seeds      - list of integer AF3 seeds to run
    user_ccd_path    - optional path to a user-provided CCD mmCIF file (e.g.
                        utils/data/retinal_11cis.cif) defining ligand_ccd_code's structure -
                        for ligands not in the standard CCD, or needing corrected geometry
                        (e.g. the standard RET entry is definitionally all-trans retinal)

    returns a dict matching AF3's native json schema (dialect='alphafold3'), ready for json.dump.
    No unpairedMsa/pairedMsa fields are set, so AF3 runs its own live MSA search -
    these are novel designed sequences, not known binders with a precomputed MSA.
    '''
    ligand_entry = {"ccdCodes": [ligand_ccd_code]} if ligand_ccd_code else {"smiles": ligand_smiles}

    af3_json = {
        "name": name,
        "modelSeeds": model_seeds,
        "sequences": [
            {
                "protein": {
                    "id": protein_chain_id,
                    "sequence": protein_seq
                }
            },
            {
                "ligand": {
                    "id": ligand_chain_id,
                    **ligand_entry
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 3
    }

    if user_ccd_path:
        af3_json["userCCDPath"] = user_ccd_path

    return af3_json

def run_af3_singularity(json_path, output_dir, af3_sif, af3_weights_dir, af3_db_dir, af3_shared_root):
    '''
    Run one AF3 cofold via singularity exec, matching the lab's confirmed-working invocation

    json_path       - path to the AF3 input json for this design
    output_dir      - directory for this design's AF3 output (created by caller)
    af3_sif         - path to the AF3 .sif container image
    af3_weights_dir - path to the AF3 code + model weights directory
    af3_db_dir      - path to the AF3 genetic/template databases
    af3_shared_root - shared storage root to bind-mount (so any path referenced inside
                       json_path, e.g. a template mmcif, resolves in-container)

    no returns. Raises CalledProcessError if the singularity/AF3 run fails.
    '''
    json_dir = os.path.dirname(json_path)

    cmd = [
        "singularity", "exec", "--nv",
        "--bind", f"{af3_shared_root}:{af3_shared_root}:ro",
        "--bind", f"{json_dir}:{json_dir}:ro",
        "--bind", f"{output_dir}:{output_dir}",
        af3_sif,
        "python", f"{af3_weights_dir}/run_alphafold.py",
        f"--json_path={json_path}",
        f"--model_dir={af3_weights_dir}",
        f"--db_dir={af3_db_dir}",
        f"--output_dir={output_dir}"
    ]

    subprocess.run(cmd, check=True)

def parse_af3_summary(summary_json_path):
    '''
    Parse an AF3 *_summary_confidences.json file into a flat score dict

    summary_json_path - path to the top-level <name>_summary_confidences.json

    returns a dict with keys: af3_iptm, af3_ptm, af3_ranking_score,
    af3_chain_pair_pae_min, af3_fraction_disordered, af3_has_clash
    '''
    with open(summary_json_path, 'r') as f:
        summary = json.load(f)

    return {
        'af3_iptm': summary['iptm'],
        'af3_ptm': summary['ptm'],
        'af3_ranking_score': summary['ranking_score'],
        'af3_chain_pair_pae_min': summary['chain_pair_pae_min'],
        'af3_fraction_disordered': summary['fraction_disordered'],
        'af3_has_clash': summary['has_clash']
    }

def get_ligand_dihedral(cif_path, ligand_chain_id, atom_names):
    '''
    Compute a dihedral angle (degrees, -180 to 180) from 4 named atoms in a ligand residue

    cif_path        - path to a predicted AF3 model (<name>_model.cif)
    ligand_chain_id - chain ID of the ligand in the predicted structure
    atom_names      - list of 4 atom names defining the dihedral, in order
                       (e.g. ['C10','C11','C12','C13'] for retinal's C11=C12 bond)

    returns the dihedral angle in degrees, or None if any named atom is missing

    NOTE: atom names must match the ligand's PDB Chemical Component Dictionary naming -
    only reliable when the AF3 input ligand was specified via ccdCodes, not smiles
    '''
    model = cif_parser.get_structure('structure', cif_path)[0]
    ligand_residue = next(iter(model[ligand_chain_id]))

    try:
        vectors = [ligand_residue[name].get_vector() for name in atom_names]
    except KeyError as e:
        print(f'ERROR: atom {e} not found in ligand chain {ligand_chain_id} of {cif_path}')
        return None

    return math.degrees(calc_dihedral(*vectors))

def select_best_af3_sample(model_dir, name, ligand_chain_id, dihedral_atoms, cis_cutoff=90.0):
    '''
    Across all seed/sample AF3 outputs for one design, pick the best cis-isomer sample

    model_dir       - AF3 output directory for one design (contains seed-*_sample-*/ subfolders)
    name            - design ID (matches the AF3 input json's "name" field)
    ligand_chain_id - chain ID of the ligand
    dihedral_atoms  - list of 4 atom names defining the isomer-diagnostic dihedral
    cis_cutoff      - degrees; abs(dihedral) < cutoff is classified cis (default=90.0)

    returns a dict combining parse_af3_summary's scores plus af3_ligand_dihedral and
    af3_ligand_is_cis, for whichever sample was selected - or None if no sample could
    be scored at all (e.g. every model.cif was missing the named atoms)

    Picks the highest-ranking_score sample among those classified cis; if none are cis,
    falls back to the highest-ranking_score sample overall and reports
    af3_ligand_is_cis=False - AF3 can't be constrained to a specific isomer, so this is
    a post-hoc pick-and-flag, not a guarantee a cis structure exists for every design.
    '''
    sample_dirs = sorted(glob.glob(os.path.join(model_dir, 'seed-*_sample-*')))

    samples = []
    for sample_dir in sample_dirs:
        cif_path = os.path.join(sample_dir, f'{name}_model.cif')
        summary_path = os.path.join(sample_dir, f'{name}_summary_confidences.json')

        dihedral = get_ligand_dihedral(cif_path, ligand_chain_id, dihedral_atoms)
        if dihedral is None:
            continue

        scores = parse_af3_summary(summary_path)
        scores['af3_ligand_dihedral'] = dihedral
        scores['af3_ligand_is_cis'] = abs(dihedral) < cis_cutoff

        samples.append(scores)

    if len(samples) == 0:
        print(f'ERROR: no valid AF3 samples found in {model_dir}')
        return None

    cis_samples = [s for s in samples if s['af3_ligand_is_cis']]

    return max(cis_samples if len(cis_samples) > 0 else samples, key=lambda s: s['af3_ranking_score'])
