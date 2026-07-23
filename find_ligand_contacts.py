"""
Find protein residues within a distance cutoff of a bound ligand

"""

# general imports
import sys
import argparse

#imports for protein design
from utils.biopdb_utils import get_ligand_contacts

# ===============================
# PARSING ARGUMENTS
# ===============================

parser = argparse.ArgumentParser()

parser.add_argument("--input_pdb", type=str, help='path to the input pdb file')
parser.add_argument('--chain_id', type=str, default='A', help='the chain ID of the protein to check for ligand contacts')
parser.add_argument('--ligand_resname', type=str, help='the 3-letter residue name of the ligand (e.g. RET)')
parser.add_argument('--distance', type=float, default=5.0, help='distance cutoff in Angstroms, heavy atoms only (default=5.0)')

args = parser.parse_args(sys.argv[1:])

# =============================
# MAIN FUNCTION
# =============================

def main(args):
    """
    Main function of the script
    """
    ##########################################################
    # Find protein residues within the distance cutoff of the ligand
    ##########################################################

    fixed_pos = get_ligand_contacts(args.input_pdb, args.chain_id, args.ligand_resname, args.distance)

    print(f'\n==========Found {len(fixed_pos)} residues within {args.distance}A of {args.ligand_resname}==========\n')

    fix_pos_str = ','.join(str(r) for r in fixed_pos)

    print(f"--fix_pos '{fix_pos_str}'\n")

# =============================
# RUN THE SCRIPT
# =============================

main(args)
