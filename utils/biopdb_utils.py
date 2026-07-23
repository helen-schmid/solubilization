import numpy as np

import Bio.PDB
from Bio.PDB import *
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Superimposer
from Bio.PDB import NeighborSearch
from Bio import BiopythonWarning

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)

pdb_parser = PDBParser(QUIET=True)
super_imposer = Superimposer()
pdb_io = PDBIO()

def expand(residues: list) -> list:
    """['A1-3', 'A5'] -> ['A1', 'A2', 'A3', 'A5']"""

    expanded = residues.copy()

    for res in expanded:
        if '-' in res:
            # get index
            idx = expanded.index(res)

            # remove entry
            expanded.pop(idx)

            # expand range
            start, end = res.split('-')
            new = [start[0] + str(x) for x in range(int(start[1:]), int(end)+1)]
            new.reverse()

            # insert new values
            for new_residue in new:
                expanded.insert(idx, new_residue)
    
    return expanded

def align_pdbs(ref_model, sample_model, residues:list=[], ref_residues:list=[], 
               mode:str='CA', return_aligned:bool=False, per_res_rmsd: bool=False):
    """
        ref_model:  Bio.PDB.Model.Model, model of the reference, must be called 'reference' 
                    (or str: path to the pdb file)
        sample_model:   Bio.PDB.Model.Model, model to be aligned to ref_model, must be called 'sample'
                        (or str: path to the pdb file)
        residues: list, list of residues to be aligned (pdb numbering), will align all residues if empty
        ref_residues: list, residues for alignment in the reference pdb. Will use residues if not provided
        mode: str, either 'CA' or 'all_atom'
        return_aligned: bool, whether to return the aligned sample_pdb
        per_res_rmsd: bool, whether to return per residue rmsd
        
        Aligns the C alphas in two pdb structures using Biopython. Returns the C alpha rmsd of the alignment.
        You can specify the chain letter in the residues list like this: ['A10', 'A13', 'A16-18'].
        
        NOTE: each residue in residues needs to start with the chain letter!
        NOTE: list of residues for reference and sample structure must be of same length!
        NOTE: in all atom mode hydrogen atoms are removed!
    """

    # Sanity check
    mode = mode.upper()
    if mode not in ['CA', 'ALL_ATOM']:
        print(f'ERROR: align_pdbs() mode must be "CA" or "ALL_ATOM"! You provided "{mode}"')
        return None

    if type(ref_model) != Bio.PDB.Model.Model:
        ref_model = pdb_parser.get_structure("reference", ref_model)[0]
    if type(sample_model) != Bio.PDB.Model.Model:
        sample_model = pdb_parser.get_structure("sample", sample_model)[0]

    if residues == []:
        align_all = True
    else:
        align_all = False
        residues = expand(residues)
                    
        if ref_residues == []:
            ref_residues = residues
        else:
            ref_residues = expand(ref_residues)
    
    # creating dict with residues to be aligned
    alignment_residues = {'reference': ref_residues, 'sample': residues}

    # now creating lists of atoms to align - stored in a dict
    align_atoms = {'reference':[], 'sample':[]}

    for model in [ref_model, sample_model]:

        for residue in model.get_residues():
            # check if residue is HETATM and skip if True
            hetatm = residue.get_full_id()[3][0]
            if hetatm != " ":
                continue

            # generating residue name (e.g. A10) for residue
            res_name = residue.parent.get_id() + str(residue.get_id()[1])

            # name of the structure (either 'reference' or 'sample')
            structure = residue.get_full_id()[0]
            
            if res_name in alignment_residues[structure] or align_all:
                if mode == 'CA':
                    align_atoms[structure].append(residue['CA'])
                elif mode == 'ALL_ATOM':                    
                    for atom in residue.get_atoms():
                        align_atoms[structure].append(atom)
                
    # removing hydrogen atoms from all atom mode
    if mode == 'ALL_ATOM':
        for structure in align_atoms:
            align_atoms[structure] = [atom for atom in align_atoms[structure] if not atom.id.startswith('H')]

    # Superimpose sample model on reference model
    super_imposer.set_atoms(align_atoms['reference'], align_atoms['sample'])
    super_imposer.apply(sample_model.get_atoms())

    # this is probably not the best idea but works
    if per_res_rmsd:
        rmsd = {}
        for ref_atm, spl_atm in zip(align_atoms['reference'], align_atoms['sample']):
            dst = np.sum((ref_atm.coord - spl_atm.coord) ** 2)
            chain = spl_atm.parent.parent.get_id()
            res_num = str(spl_atm.parent.get_id()[1])
            res_id = chain + res_num
            if res_id not in rmsd.keys():
                rmsd[res_id] = []

            rmsd[res_id].append(dst)
        
        # convert to mean of entire residue
        for res_id in rmsd.keys():
            rmsd[res_id] = np.sqrt(sum(rmsd[res_id]) / len(rmsd[res_id]))
            
    else:
        rmsd = super_imposer.rms

    if return_aligned:
        return rmsd, sample_model
    else:
        return rmsd

def get_ligand_contacts(pdb_filename:str, chain_id:str, ligand_resname:str, distance:float=5.0) -> list:
    '''
    Find protein residues in chain_id with a heavy atom within distance of ligand_resname's heavy atoms

    pdb_filename    - path to the input pdb file
    chain_id        - chain ID of the protein to check for ligand contacts
    ligand_resname  - 3-letter residue name of the ligand (e.g. RET)
    distance        - distance cutoff in Angstroms, heavy atoms only (default=5.0)

    returns a sorted list of protein residue numbers (int) with at least one heavy atom
    within distance of the ligand's heavy atoms

    NOTE: only heavy atoms (no hydrogens) are considered, on both the protein and ligand side
    NOTE: the ligand is matched by residue name anywhere in the structure (not restricted to chain_id),
          since a ligand HETATM may share a chain letter with the protein (as in input/1jgj.pdb)
    '''
    model = pdb_parser.get_structure('structure', pdb_filename)[0]

    ligand_atoms = [atom for residue in model.get_residues() for atom in residue
                    if residue.get_resname() == ligand_resname and atom.element != 'H']

    if len(ligand_atoms) == 0:
        print(f'ERROR: no atoms found for ligand "{ligand_resname}" in {pdb_filename}')
        return []

    ns = NeighborSearch(ligand_atoms)

    contacts = []
    for residue in model[chain_id]:
        if residue.get_full_id()[3][0] != " ":
            continue # skip HETATM residues - only fix protein residues

        for atom in residue:
            if atom.element == 'H':
                continue
            if ns.search(atom.coord, distance):
                contacts.append(residue.get_id()[1])
                break

    return sorted(contacts)