"""
An end-to-end pipeline for designing a soluble analogue of a membrane protein


"""

# general imports
import sys
import argparse
import os
import numpy as np
import glob
import csv
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#imports for protein design
from utils.common import make_directory, initialise_checkpoint_file, initialise_csv_file, write_to_csv, update_checkpoint_file
from utils.biopdb_utils import align_pdbs
from utils.colabdesign_utils import full_dgram_loss, get_best_seq, rank_and_write_pdb, relax_me
from utils.pyrosetta_utils import get_surface_apolar_fraction

import pyrosetta
from colabdesign import mk_af_model
from colabdesign.mpnn import mk_mpnn_model

from Bio.PDB import PDBParser

pdb_parser = PDBParser(QUIET=True)

# ===============================
# PARSING ARGUMENTS
# ===============================

parser = argparse.ArgumentParser()

MPNN_MODELS = ['v_48_002', 'v_48_010', 'v_48_010', 'v_48_020', 'v_48_030']

parser.add_argument("-working_dir", type=str, help="the directory to work in")
parser.add_argument("-input_pdb", type=str, help='path to the input pdb file')
parser.add_argument("-fix_pos", type=str, help='the fixed positions to be parsed in')
parser.add_argument("-sidechain_loss_pos", type=str, default='', help='the positions to apply sidechain loss on')
parser.add_argument('-params', type=str, default='/work/lpdi/users/hilditch/software/params', help="Path to ColabDesign AF params")
parser.add_argument('-chain_id', type=str, default='A', help='the chain ID to be designed')
parser.add_argument('-num_backbones', type=int, default=2, help='the number of backbones to hallucinate')
parser.add_argument("-mpnn_temp", type=float, default=0.1, help='temperature for the mpnn model (default=0.1)')
parser.add_argument("-nseq", type=int, default=5, help='number of sequences to predict using ProteinMPNN')
parser.add_argument("-mpnn_model", choices=MPNN_MODELS, default='v_48_030', help='model weights to use for ProteinMPNN')
parser.add_argument('-backbone_noise', type=float, default=0.01, help='The variance of random noise to add to the input atomic coordinates (default=0.0)')

args = parser.parse_args(sys.argv[1:])

# =============================
# MAIN FUNCTION
# =============================

def main(args):
    """
    Main function of the script
    """
    ####################################################
    # Begin running the AF2-seq back propogation process
    ####################################################

    print(f'\n==========Starting AF2seq backpropogation for {args.num_backbones} designs==========\n')

    # set up the directories for all the elements of this script
    af2_seq_dir = os.path.join(args.working_dir, '01_af2_seq_bbs')
    af2_bb_repredictions_dir = os.path.join(args.working_dir, '02_bb_repredictions')
    mpnn_dir = os.path.join(args.working_dir, '03_sol_mpnn')
    af2_final_predictions_dir = os.path.join(args.working_dir, '04_af2_final_predictions')
    
    make_directory(args.working_dir)
    make_directory(af2_seq_dir)
    make_directory(af2_bb_repredictions_dir)
    make_directory(mpnn_dir)
    make_directory(af2_final_predictions_dir)

    # set up a csv file to save the scores from af2seq and the forward prediction
    columns = ['ID', 'sequence', 'af2_seq_plddt', 'af2_seq_rmsd']

    #setting up a csv file for the scores
    log_csv = os.path.join(args.working_dir, '01_backprop_scores.csv')
    initialise_csv_file(log_csv, columns=columns)

    #set up a checkpoint file
    checkpoint_file = os.path.join(args.working_dir, '01_backprop_checkpoint.txt')
    initialise_checkpoint_file(checkpoint_file)
    
    #create lists of positions to fix and design
    print(f'Fixing positions: {args.fix_pos}')

    #compile the model - now we decide if we use the sidechain loss or not

    if args.sidechain_loss_pos == 0:
        use_sidechain_loss = False
    else:
        use_sidechain_loss = True

    if use_sidechain_loss:
        af_model = mk_af_model(protocol='fixbb',
                    use_templates=True,
                    data_dir=args.params,
                    loss_callback=full_dgram_loss)
    else:
        af_model = mk_af_model(protocol='fixbb',
                    use_templates=True,
                    data_dir=args.params)

    #begin backpropogation
    for i in range(args.num_backbones):

        if f'traj_{i}_seq' in open(checkpoint_file).read():
            print(f'Trajectory {i} already done, skipping')
            continue
        else:
            print(f'\nBeginning design {i}\n')

        if use_sidechain_loss:
            #prep the inputs
            af_model.prep_inputs(pdb_filename=args.input_pdb,
                                chain=args.chain_id,
                                fix_pos=args.fix_pos,
                                use_sidechains=True,
                                pos = args.sidechain_loss_pos)
            
            #set the loss weights
            af_model.opt["weights"]["dgram_cce"] = 0.0
            af_model.opt["weights"]["full_dgram_loss"] = 0.5
            af_model.opt["weights"]["fape"] = 1.0

        else:
            #prep the inputs
            af_model.prep_inputs(pdb_filename=args.input_pdb,
                                chain=args.chain_id,
                                fix_pos=args.fix_pos)
            
            #setting the loss weights        
            af_model.opt["weights"]["fape"] = 1.0
            af_model.opt["weights"]["plddt"] = 0.1
            af_model.opt["weights"]["dgram_cce"] = 0.0

        # make a copy for the full template loss
        af_model._inputs['batch_2'] = copy.deepcopy(af_model._inputs['batch'])

        #initialise the starting sequence - gumbel initialises with a random sequence
        af_model.set_seq(mode='gumbel',
                        rm_aa="C")

        af_model.design_3stage(100,100,10)

        af_model.design_semigreedy(20, tries=20, models=[0,1], num_models=2)
        
        #collecting the scores of the best design
        best_d = af_model._tmp['best']['aux']
        best_seq = get_best_seq(best_d)[0]

        #writing a fasta file with the sequence of the best design to be later used for forward prediction
        with open(f"{af2_seq_dir}/traj_{i}.fasta","w") as fasta:
            line = f'>traj_{i}\n{best_seq}'
            fasta.write(line)
            fasta.close()
            
        #saving the pdb file of the best design
        rank_and_write_pdb(af_model,
                        name=f'{af2_seq_dir}/traj_{i}')
        
        #collecting the scores of the best design into a dictionary
        scores = {'ID': f'traj_{i}',
                  'sequence': best_seq,
                  'af2_seq_plddt': np.mean(best_d['plddt']),
                  'af2_seq_rmsd': best_d['losses']['rmsd']
                    }
        
        write_to_csv(log_csv, columns, scores)
        update_checkpoint_file(checkpoint_file, f'traj_{i}_seq')

    print(f'\n==================Completed AF2seq back-propogation for {args.num_backbones} designs=================\n')
    
    ############################################################
    # Begin re-predicting the sequences with an AF2 forward pass
    ############################################################

    print(f'\n==========================Repredicting sequences with AF2=========================\n')

    #collecting the fasta files output from af2seq backprop
    num_sequences_to_predict = len(glob.glob(f'{af2_seq_dir}/*.fasta'))

    print(f'Found {num_sequences_to_predict} sequences for forward prediction')

    #setting up the af2 forward prediction with fixbb
    af_model = mk_af_model(protocol='fixbb',
                        use_templates=True,
                        data_dir=args.params)

    #iterating through the files
    for file in glob.glob(f'{af2_seq_dir}/*.fasta'):
        with open(file,"r") as fasta:
            lines = fasta.readlines()
            seq_name = lines[0].strip('>').strip('\n')
            sequence = lines[1]

            print(f'\nRunning sequence: {seq_name}\n')

            #prep the inputs
            if use_sidechain_loss:
                af_model.prep_inputs(pdb_filename=args.input_pdb,
                                    chain=args.chain_id,
                                    use_sidechains=True,
                                    pos = args.sidechain_loss_pos)

            else:
                af_model.prep_inputs(pdb_filename=args.input_pdb,
                                    chain=args.chain_id)        

            af_model.set_seq(sequence)
            af_model.predict(num_recycles=3)

            pdbs, rank = rank_and_write_pdb(af_model,
                                            predict=True,
                                            name=f'{af2_bb_repredictions_dir}/{seq_name}')
            
            #relaxing the predicted structure
            print('Relaxing...')
            for pdb in pdbs:
                pdb_out = f'{af2_bb_repredictions_dir}/{seq_name}_relaxed.pdb'
                relax_me(pdb, pdb_out)
                os.remove(pdb) # delete unrelaxed pdb file
            
            #collecting the plddt of the repredicted structure
            repredicted_plddt= af_model.aux["log"]["plddt"]

            #appending the repredicted plddt to the csv file
            with open(log_csv, 'r') as csvinput:
                with open(log_csv.replace('.csv', '_repredicted.csv'), 'w') as csvoutput:
                    reader = csv.reader(csvinput)
                    writer = csv.writer(csvoutput)
                    for n, row in enumerate(reader):
                        if n == 0:
                            row.append('repredicted_plddt')
                            writer.writerow(row)
                        if seq_name in row:
                            row.append(repredicted_plddt)
                            writer.writerow(row)

    print(f'\n=====================Completed AF2 re-prediction for {num_sequences_to_predict} sequences=====================\n')

    num_pdb_files=len(glob.glob(f'{af2_bb_repredictions_dir}/*.pdb'))

    ############################################################
    # Begin re-designing the sequences with soluble protein MPNN
    ############################################################

    print(f'\n===================Beginning soluble ProteinMPNN sequence re-design==================\n')

    print(f"{num_pdb_files} PDB files identified")
    print(f'Reference PDB: {args.input_pdb}')

    print(f'ProteinMPNN model: {args.mpnn_model}')
    print(f'ProteinMPNN weights: soluble')
    print(f'ProteinMPNN temperature: {args.mpnn_temp}')
    print(f'ProteinMPNN backbone noise: {args.backbone_noise}')
    print(f'ProteinMPNN number of sequences: {args.nseq}\n')

    #collecting the pdb files output from forward af2 prediction
    files=glob.glob(f'{af2_bb_repredictions_dir}/*.pdb')

    #compiling the mpnn model
    mpnn_model = mk_mpnn_model(args.mpnn_model,
                                backbone_noise=args.backbone_noise,
                                weights='soluble')
    
    #iterating through the files
    for fn in files:
        fname_no_extension = os.path.basename(fn).replace('.pdb','')
        print(f"Opening file: {fname_no_extension}")
        
        # Prepping inputs for MPNN and running the designs      
        mpnn_model.prep_inputs(pdb_filename=fn,
                               chain=args.chain_id,
                               fix_pos=args.fix_pos,
                               rm_aa = "C")
        
        print(f'Generating {args.nseq} sequences\n')
        out=mpnn_model.sample(num=args.nseq,
                              temperature=args.mpnn_temp)
        
        # Writing fasta files for each mpnn output

        with open(f"{mpnn_dir}/{fname_no_extension}_mpnn.fasta","w") as fasta:
            for n in range(args.nseq):
                line = f'>{fname_no_extension}_mpnn_{n}\n{out["seq"][n]}\n'
                fasta.write(line)
        fasta.close()
 
    print(f'\n===================Soluble ProteinMPNN sequence design complete==================\n')

    ##############################################################
    # Begin the final re-prediction of the MPNN sequences with AF2
    ##############################################################

    print(f'\n===================Beginning final AF2 forward prediction==================\n')

    files=glob.glob(f'{mpnn_dir}/*.fasta')

    # set up the csv file to save the final scores from af2seq
    columns = ['ID', 'sequence', 'path', 'plddt', 'top_model_plddt', 'pae', 'ptm', 'ca_rmsd', 'surf_apolar_frac']
    
    #setting up a csv file for the scores
    log_csv = os.path.join(args.working_dir, '02_final_prediction_scores.csv')
    initialise_csv_file(log_csv, columns=columns)

    #set up a checkpoint file
    checkpoint_file = os.path.join(args.working_dir, '02_forward_prediction_checkpoint.txt')
    initialise_checkpoint_file(checkpoint_file)

    #initialise the model
    af_model = mk_af_model(protocol='fixbb',
                        use_templates=False,
                        initial_guess=False,
                        use_initial_atom_pos=False,
                        data_dir=args.params
                        )

    #beginning to iterate through the fasta files output by proteinmpnn
    for file in files:
        with open(file,"r") as fasta:
            lines = fasta.readlines()

            for i in range(0, len(lines), 2):
                seq_name = lines[i].strip('>').strip('\n')
                sequence = lines[i+1].strip('\n')

                print(f'Predicting: {seq_name}')
                print(f'Sequence: {sequence}\n')
                
                af_model.prep_inputs(pdb_filename=args.input_pdb,
                                    chain=args.chain_id
                                    )
                
                af_model.set_seq(sequence)
                
                af_model.predict(num_recycles=3)

                #saving the final output from proteinmpnn
                pdbs, rank = rank_and_write_pdb(af_model,
                                                predict=True,
                                                write_all=False,
                                                renum_pdb=True,
                                                name=f'{af2_final_predictions_dir}/{seq_name}')
                
                # collecting scores from AF2 prediction
                af_score = af_model.aux["log"]

                plddt = af_score["plddt"]
                pae = 31.0 * af_score["pae"]
                ptm = af_score["ptm"]
                top_model_plddt = np.mean(af_model.aux['all']['plddt'], -1)[rank]

                #doing a final rmsd measurement of the predicted structure to the reference either with the complete structure, or ref atoms
                full_structure_ca_rmsd = align_pdbs(ref_model=args.input_pdb, sample_model=pdbs[0], mode='ca')

                #calculating fraction apolar residues using pyrosetta
                pose = pyrosetta.pose_from_pdb(pdbs[0])

                frac_apolar = get_surface_apolar_fraction(pose)

                # create score dict
                final_scores = {'ID': seq_name,
                            'sequence': sequence,
                            'path': pdbs[0],
                            'plddt': round(plddt, 4),
                            'top_model_plddt': round(top_model_plddt, 4),
                            'pae': round(pae, 4),
                            'ptm': round(ptm, 4),
                            'ca_rmsd': full_structure_ca_rmsd,
                            'surf_apolar_frac': frac_apolar
                            }
                
                write_to_csv(log_csv, columns, final_scores)
                update_checkpoint_file(checkpoint_file, f'{seq_name}_seq')

    print(f'\n==============Final AF2 predictions on soluble sequences complete==============\n')

    print(f'\n=========================Script completed successfully=========================\n')

# =============================
# RUN THE SCRIPT
# =============================
pyrosetta.init("-beta_nov16 -mute all")

main(args)