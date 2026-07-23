"""
AF3 cofold filtering for solubilized designs

Main script
"""

# general imports
import sys
import argparse
import os
import csv
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#imports for protein design
from utils.common import make_directory, initialise_checkpoint_file, initialise_csv_file, write_to_csv, update_checkpoint_file
from utils.af3_utils import build_af3_json, run_af3_singularity, select_best_af3_sample

# ===============================
# PARSING ARGUMENTS
# ===============================

parser = argparse.ArgumentParser()

parser.add_argument("--working_dir", type=str, help="the directory to work in")
parser.add_argument("--input_csv", type=str, help="path to the stage-04 scores csv from solubilization.py (columns: ID, sequence, path, plddt, top_model_plddt, pae, ptm, ca_rmsd, surf_apolar_frac)")
parser.add_argument("--ligand_ccd_code", type=str, default=None, help="PDB Chemical Component Dictionary code of the native ligand (e.g. RET for retinal) - preferred over --ligand_smiles, gives reliable atom names for the isomer check")
parser.add_argument("--ligand_ccd_path", type=str, default=None, help="path to a user-provided CCD mmCIF file defining --ligand_ccd_code's structure (e.g. utils/data/retinal_11cis.cif) - use this when the standard CCD entry has the wrong isomer/geometry for your ligand")
parser.add_argument("--ligand_smiles", type=str, default=None, help="SMILES string of the native ligand, used only if --ligand_ccd_code is not given")
parser.add_argument("--af3_sif", type=str, help="path to the AF3 singularity container (.sif)")
parser.add_argument("--af3_weights_dir", type=str, help="path to the AF3 code + model weights directory")
parser.add_argument("--af3_db_dir", type=str, help="path to the AF3 genetic/template databases")
parser.add_argument("--af3_shared_root", type=str, help="shared storage root to bind-mount into the container")
parser.add_argument("--protein_chain_id", type=str, default='A', help="chain ID to assign the designed protein in the AF3 input json (default=A)")
parser.add_argument("--ligand_chain_id", type=str, default='B', help="chain ID to assign the ligand in the AF3 input json (default=B)")
parser.add_argument("--model_seeds", type=str, default='1', help="comma-separated AF3 model seeds to run per design (default=1)")
parser.add_argument("--min_top_model_plddt", type=float, default=0.0, help="skip designs below this stage-04 top_model_plddt before cofolding (default=0.0, i.e. no filtering)")
parser.add_argument("--dihedral_atoms", type=str, default='C10,C11,C12,C13', help="comma-separated names of the 4 ligand atoms defining the isomer-diagnostic dihedral (default=C10,C11,C12,C13, retinal's C11=C12 bond)")
parser.add_argument("--cis_cutoff", type=float, default=90.0, help="degrees; abs(dihedral) below this is classified cis (default=90.0)")

args = parser.parse_args(sys.argv[1:])

# =============================
# MAIN FUNCTION
# =============================

def main(args):
    """
    Main function of the script
    """
    ####################################################
    # Begin AF3 cofold filtering of solubilized designs
    ####################################################

    if args.ligand_ccd_code is None and args.ligand_smiles is None:
        print('ERROR: must provide either --ligand_ccd_code or --ligand_smiles')
        return

    # set up the directories for the elements of this script
    af3_inputs_dir = os.path.join(args.working_dir, '05_af3_inputs')
    af3_outputs_dir = os.path.join(args.working_dir, '06_af3_outputs')

    make_directory(args.working_dir)
    make_directory(af3_inputs_dir)
    make_directory(af3_outputs_dir)

    # set up a csv file to save the AF3 cofold scores
    columns = ['ID', 'sequence', 'af3_iptm', 'af3_ptm', 'af3_ranking_score', 'af3_chain_pair_pae_min', 'af3_fraction_disordered', 'af3_has_clash', 'af3_ligand_dihedral', 'af3_ligand_is_cis']

    #setting up a csv file for the scores
    log_csv = os.path.join(args.working_dir, '03_af3_scores.csv')
    initialise_csv_file(log_csv, columns=columns)

    #set up a checkpoint file
    checkpoint_file = os.path.join(args.working_dir, '03_af3_checkpoint.txt')
    initialise_checkpoint_file(checkpoint_file)

    #reading in the designs to be cofolded
    with open(args.input_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    model_seeds = [int(s) for s in args.model_seeds.split(',')]
    dihedral_atoms = args.dihedral_atoms.split(',')

    # AF3 resolves a relative userCCDPath against the input json's own directory
    # (05_af3_inputs/), not the cwd - resolve to an absolute path up front so it
    # works regardless of --working_dir's location/depth
    ligand_ccd_path = os.path.abspath(args.ligand_ccd_path) if args.ligand_ccd_path else None

    print(f'\n==========Starting AF3 cofold filtering for {len(rows)} designs==========\n')

    for row in rows:
        design_id = row['ID']

        if f'{design_id}_seq' in open(checkpoint_file).read():
            print(f'{design_id} already scored, skipping')
            continue

        if float(row['top_model_plddt']) < args.min_top_model_plddt:
            print(f'{design_id} below min_top_model_plddt threshold, skipping cofold')
            continue

        print(f'\nCofolding {design_id}\n')

        #building the AF3 input json for this design
        af3_json = build_af3_json(design_id,
                                   row['sequence'],
                                   args.ligand_ccd_code,
                                   args.ligand_smiles,
                                   args.protein_chain_id,
                                   args.ligand_chain_id,
                                   model_seeds,
                                   ligand_ccd_path)

        json_path = os.path.join(af3_inputs_dir, f'{design_id}.json')
        with open(json_path, 'w') as f:
            json.dump(af3_json, f, indent=2)

        #running the cofold
        design_out_dir = os.path.join(af3_outputs_dir, design_id)
        make_directory(design_out_dir)

        run_af3_singularity(json_path,
                            design_out_dir,
                            args.af3_sif,
                            args.af3_weights_dir,
                            args.af3_db_dir,
                            args.af3_shared_root)

        #picking the best cis-isomer sample across all seeds, and collecting its scores
        model_dir = os.path.join(design_out_dir, design_id)
        scores = select_best_af3_sample(model_dir, design_id, args.ligand_chain_id, dihedral_atoms, args.cis_cutoff)

        if scores is None:
            print(f'{design_id}: no scoreable AF3 sample found, skipping')
            continue

        scores['ID'] = design_id
        scores['sequence'] = row['sequence']

        write_to_csv(log_csv, columns, scores)
        update_checkpoint_file(checkpoint_file, f'{design_id}_seq')

    print(f'\n==================Completed AF3 cofold filtering for {len(rows)} designs=================\n')

# =============================
# RUN THE SCRIPT
# =============================

main(args)
