"""
One-time derivation script: builds a genuinely cis-isomer retinal ligand definition

Not part of the runtime pipeline - run once to (re)generate utils/data/retinal_11cis.cif

The standard PDB ligand code RET ("retinal") is used generically for retinal in the
PDB regardless of its actual isomeric state in a given structure - but RET's own
official ideal coordinates (https://files.rcsb.org/ligands/view/RET.cif) are
definitionally all-trans (the C10-C11=C12-C13 dihedral of the official ideal
coordinates below is -179.6 degrees). AF3 builds its ligand reference conformer from
exactly this information, so passing --ligand_ccd_code RET to filtering.py is not a
neutral choice - it points AF3 toward the wrong isomer for this project.

This script takes RET's own official ideal coordinates and bonds, and rotates the
distal fragment on the far side of the C11=C12 bond by exactly 180 degrees around the
C11-C12 bond axis. This flips only that one bond's torsion (trans -> cis) while
leaving every bond length and bond angle in the molecule unchanged - a physically
valid alternative 3D structure of the same molecular graph, not an approximation.

Atom names, elements and bonds are otherwise identical to RET, so existing code that
measures the dihedral by atom name (utils.af3_utils.get_ligand_dihedral) works
unchanged on this component's output.
"""

import os
import numpy as np

from Bio.PDB.vectors import Vector, calc_dihedral

COMPONENT_ID = 'RET-11CIS'
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'data', 'retinal_11cis.cif')

# official RET ideal coordinates, from https://files.rcsb.org/ligands/view/RET.cif
RET_ATOMS = {
    'C1':  ('C', -4.327,  0.909, -0.480),
    'C2':  ('C', -5.466,  0.132, -1.143),
    'C3':  ('C', -5.998, -0.901, -0.142),
    'C4':  ('C', -4.921, -1.966,  0.075),
    'C5':  ('C', -3.604, -1.314,  0.380),
    'C6':  ('C', -3.340, -0.042,  0.131),
    'C7':  ('C', -2.051,  0.450,  0.458),
    'C8':  ('C', -0.944, -0.145, -0.059),
    'C9':  ('C',  0.334,  0.315,  0.300),
    'C10': ('C',  1.446, -0.283, -0.220),
    'C11': ('C',  2.720,  0.176,  0.138),
    'C12': ('C',  3.842, -0.427, -0.387),
    'C13': ('C',  5.115,  0.031, -0.030),
    'C14': ('C',  6.238, -0.573, -0.555),
    'C15': ('C',  7.510, -0.115, -0.198),
    'O1':  ('O',  8.500, -0.648, -0.661),
    'C16': ('C', -3.620,  1.768, -1.531),
    'C17': ('C', -4.899,  1.816,  0.611),
    'C18': ('C', -2.522, -2.155,  1.006),
    'C19': ('C',  0.478,  1.469,  1.259),
    'C20': ('C',  5.259,  1.185,  0.929),
    'H21':  ('H', -6.266,  0.819, -1.419),
    'H22':  ('H', -5.093, -0.377, -2.032),
    'H31':  ('H', -6.225, -0.411,  0.804),
    'H32':  ('H', -6.898, -1.368, -0.542),
    'H41':  ('H', -5.209, -2.606,  0.910),
    'H42':  ('H', -4.824, -2.572, -0.825),
    'H7':   ('H', -1.949,  1.298,  1.119),
    'H8':   ('H', -1.047, -0.972, -0.747),
    'H10':  ('H',  1.343, -1.109, -0.908),
    'H11':  ('H',  2.823,  1.003,  0.825),
    'H12':  ('H',  3.740, -1.254, -1.074),
    'H14':  ('H',  6.135, -1.400, -1.242),
    'H15':  ('H',  7.613,  0.712,  0.489),
    'H161': ('H', -4.351,  2.401, -2.035),
    'H162': ('H', -2.872,  2.394, -1.045),
    'H163': ('H', -3.134,  1.122, -2.262),
    'H171': ('H', -5.409,  1.209,  1.359),
    'H172': ('H', -4.089,  2.371,  1.084),
    'H173': ('H', -5.608,  2.516,  0.167),
    'H181': ('H', -2.917, -3.145,  1.234),
    'H182': ('H', -1.686, -2.248,  0.312),
    'H183': ('H', -2.179, -1.681,  1.926),
    'H191': ('H',  0.528,  1.089,  2.279),
    'H192': ('H',  1.391,  2.019,  1.031),
    'H193': ('H', -0.381,  2.133,  1.160),
    'H201': ('H',  5.309,  0.805,  1.950),
    'H202': ('H',  6.172,  1.735,  0.701),
    'H203': ('H',  4.400,  1.849,  0.830),
}

# official RET bonds (atom_id_1, atom_id_2, value_order, aromatic_flag)
RET_BONDS = [
    ('C1', 'C2', 'SING', 'N'), ('C1', 'C6', 'SING', 'N'), ('C1', 'C16', 'SING', 'N'),
    ('C1', 'C17', 'SING', 'N'), ('C2', 'C3', 'SING', 'N'), ('C2', 'H21', 'SING', 'N'),
    ('C2', 'H22', 'SING', 'N'), ('C3', 'C4', 'SING', 'N'), ('C3', 'H31', 'SING', 'N'),
    ('C3', 'H32', 'SING', 'N'), ('C4', 'C5', 'SING', 'N'), ('C4', 'H41', 'SING', 'N'),
    ('C4', 'H42', 'SING', 'N'), ('C5', 'C6', 'DOUB', 'N'), ('C5', 'C18', 'SING', 'N'),
    ('C6', 'C7', 'SING', 'N'), ('C7', 'C8', 'DOUB', 'N'), ('C7', 'H7', 'SING', 'N'),
    ('C8', 'C9', 'SING', 'N'), ('C8', 'H8', 'SING', 'N'), ('C9', 'C10', 'DOUB', 'N'),
    ('C9', 'C19', 'SING', 'N'), ('C10', 'C11', 'SING', 'N'), ('C10', 'H10', 'SING', 'N'),
    ('C11', 'C12', 'DOUB', 'N'), ('C11', 'H11', 'SING', 'N'), ('C12', 'C13', 'SING', 'N'),
    ('C12', 'H12', 'SING', 'N'), ('C13', 'C14', 'DOUB', 'N'), ('C13', 'C20', 'SING', 'N'),
    ('C14', 'C15', 'SING', 'N'), ('C14', 'H14', 'SING', 'N'), ('C15', 'O1', 'DOUB', 'N'),
    ('C15', 'H15', 'SING', 'N'), ('C16', 'H161', 'SING', 'N'), ('C16', 'H162', 'SING', 'N'),
    ('C16', 'H163', 'SING', 'N'), ('C17', 'H171', 'SING', 'N'), ('C17', 'H172', 'SING', 'N'),
    ('C17', 'H173', 'SING', 'N'), ('C18', 'H181', 'SING', 'N'), ('C18', 'H182', 'SING', 'N'),
    ('C18', 'H183', 'SING', 'N'), ('C19', 'H191', 'SING', 'N'), ('C19', 'H192', 'SING', 'N'),
    ('C19', 'H193', 'SING', 'N'), ('C20', 'H201', 'SING', 'N'), ('C20', 'H202', 'SING', 'N'),
    ('C20', 'H203', 'SING', 'N'),
]

# atoms on the far side of the C11=C12 bond from C11 (i.e. bonded to C12 or beyond,
# not counting C11 itself) - this whole rigid fragment gets rotated together
DISTAL_FRAGMENT = ['H12', 'C13', 'C14', 'H14', 'C15', 'H15', 'O1', 'C20', 'H201', 'H202', 'H203']


def rotate_180(coords, pivot, axis):
    '''180 degree rotation of a point around an axis through pivot (Rodrigues' formula at theta=180)'''
    v = np.array(coords) - np.array(pivot)
    u = np.array(axis)
    v_rotated = 2 * np.dot(v, u) * u - v
    return tuple(np.array(pivot) + v_rotated)


def main():
    c11 = np.array(RET_ATOMS['C11'][1:])
    c12 = np.array(RET_ATOMS['C12'][1:])
    axis = (c12 - c11) / np.linalg.norm(c12 - c11)

    cis_atoms = dict(RET_ATOMS)
    for atom_id in DISTAL_FRAGMENT:
        element, *coords = RET_ATOMS[atom_id]
        cis_atoms[atom_id] = (element, *rotate_180(coords, c12, axis))

    # sanity check: confirm the C11=C12 torsion actually flipped to ~cis before writing anything
    dihedral_atoms = ['C10', 'C11', 'C12', 'C13']
    vectors = [Vector(*cis_atoms[a][1:]) for a in dihedral_atoms]
    import math
    dihedral = math.degrees(calc_dihedral(*vectors))
    assert abs(dihedral) < 5.0, f'expected a cis (~0 deg) dihedral, got {dihedral:.2f} deg'
    print(f'C10-C11=C12-C13 dihedral after rotation: {dihedral:.2f} degrees (cis)')

    write_ccd(cis_atoms, RET_BONDS, OUTPUT_PATH)
    print(f'wrote {OUTPUT_PATH}')


def write_ccd(atoms, bonds, output_path):
    '''writes a minimal user-provided-CCD mmCIF file, matching the format in AF3 docs/input.md'''
    lines = []

    lines.append(f'data_{COMPONENT_ID}')
    lines.append('#')
    lines.append(f'_chem_comp.id {COMPONENT_ID}')
    lines.append("_chem_comp.name 'RETINAL, 11-cis isomer (corrected from RET)'")
    lines.append('_chem_comp.type non-polymer')
    lines.append("_chem_comp.formula 'C20 H28 O'")
    lines.append('_chem_comp.mon_nstd_parent_comp_id ?')
    lines.append('_chem_comp.pdbx_synonyms ?')
    lines.append('_chem_comp.formula_weight 284.436')
    lines.append('#')
    lines.append('loop_')
    lines.append('_chem_comp_atom.comp_id')
    lines.append('_chem_comp_atom.atom_id')
    lines.append('_chem_comp_atom.type_symbol')
    lines.append('_chem_comp_atom.charge')
    lines.append('_chem_comp_atom.pdbx_leaving_atom_flag')
    lines.append('_chem_comp_atom.pdbx_model_Cartn_x_ideal')
    lines.append('_chem_comp_atom.pdbx_model_Cartn_y_ideal')
    lines.append('_chem_comp_atom.pdbx_model_Cartn_z_ideal')
    for atom_id, (element, x, y, z) in atoms.items():
        lines.append(f'{COMPONENT_ID} {atom_id} {element} 0 N {x:.3f} {y:.3f} {z:.3f}')
    lines.append('#')
    lines.append('loop_')
    lines.append('_chem_comp_bond.atom_id_1')
    lines.append('_chem_comp_bond.atom_id_2')
    lines.append('_chem_comp_bond.value_order')
    lines.append('_chem_comp_bond.pdbx_aromatic_flag')
    for atom_1, atom_2, value_order, aromatic_flag in bonds:
        lines.append(f'{atom_1} {atom_2} {value_order} {aromatic_flag}')
    lines.append('#')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
