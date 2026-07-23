import os
import json
import subprocess

def build_af3_json(name, protein_seq, ligand_smiles, protein_chain_id='A', ligand_chain_id='B', model_seeds=[1]):
    '''
    Build an AF3 cofold input record for one designed monomer + its native ligand

    name             - design ID, used as both the json's "name" field and the output filename
    protein_seq      - the designed protein sequence to cofold
    ligand_smiles    - SMILES string of the native ligand
    protein_chain_id - chain ID to assign the protein (default='A')
    ligand_chain_id  - chain ID to assign the ligand (default='B')
    model_seeds      - list of integer AF3 seeds to run

    returns a dict matching AF3's native json schema (dialect='alphafold3'), ready for json.dump.
    No unpairedMsa/pairedMsa fields are set, so AF3 runs its own live MSA search -
    these are novel designed sequences, not known binders with a precomputed MSA.
    '''
    return {
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
                    "smiles": ligand_smiles
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 3
    }

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
