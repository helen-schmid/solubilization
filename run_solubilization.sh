#!/bin/bash
#SBATCH --chdir /work/lpdi/users/hilditch/scripts/repos/solubilization
#SBATCH --job-name=solubilize
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem 32gb
#SBATCH --time 10:00:00
#SBATCH --output=solubilize.log

source /home/hilditch/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate ProteinDesign_kuma

python -u /work/lpdi/users/hilditch/scripts/repos/solubilization/solubilization.py \
        --working_dir ./test \
        --input_pdb ./1jgj.pdb \
        --num_backbones 3\
        --fix_pos '73,76,79,80,83,108,109,112,127,130,131,134,171,174,175,178,201,204,205' \
        --sidechain_loss_pos '73,76,79,80,83,108,109,112,127,130,131,134,171,174,175,178,201,204,205'
