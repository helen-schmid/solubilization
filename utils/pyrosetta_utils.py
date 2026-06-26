from pyrosetta import *

#pyrosetta.init("-beta_nov16 -mute all")

def get_surface_apolar_fraction(pose) -> float:
    '''Function to calculate the fraction of apolar residues on the surface - modified from Casper's code'''

    if type(pose) != pyrosetta.rosetta.core.pose.Pose:
        pose = pose_from_pdb(pose)

    layer_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(pose)

    exp_apol_count = 0
    total_count = 0
    
    apolar_res = ['ALA','PHE','ILE','MET','LEU','TRP','VAL','TYR']
    
    for i in range(1, len(surface_res)+1):
        if surface_res[i] == True:
            res = pose.residue(i)
            if res.name() in apolar_res:
                exp_apol_count += 1
            total_count += 1

    return round(exp_apol_count/total_count, 4)