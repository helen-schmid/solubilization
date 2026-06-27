#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem 32gb
#SBATCH --time 10:00:00
#SBATCH --output=solubilize.log

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/Solubilization

python -u solubilization.py \
        --params path/to/params \
        --working_dir ./output \
        --input_pdb ./input/1jgj.pdb \
        --num_backbones 1 \
        --fix_pos '73,76,79,80,83,108,109,112,127,130,131,134,171,174,175,178,201,204,205' \
        --sidechain_loss_pos '73,76,79,80,83,108,109,112,127,130,131,134,171,174,175,178,201,204,205'
