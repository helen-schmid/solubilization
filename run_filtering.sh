#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mem 64gb
#SBATCH --time 08:00:00
#SBATCH --output=filtering.log
# Kuma (SCITAS pay-per-use, the only cluster with AF3 installed) requires a
# billing account and QOS - unlike run_solubilization.sh's cluster.
#SBATCH --account=TODO_your_scitas_account
#SBATCH --qos=normal

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/Solubilization

# --time above assumes one design at a time (roughly ~1h per live-MSA cofold) -
# scale it up for a larger design set, or split designs across multiple
# submissions (03_af3_checkpoint.txt makes reruns safe to resume).

# AF3_SIF/AF3_WEIGHTS_DIR/AF3_DB_DIR/AF3_SHARED_ROOT below are the lab's
# shared AF3 install on Kuma - reference them in place, do not copy the
# weights elsewhere or share them outside the lab.
python -u filtering.py \
        --working_dir ./output \
        --input_csv ./output/02_final_prediction_scores.csv \
        --ligand_smiles 'SMILES_STRING_HERE' \
        --af3_sif /work/lpdi/users/dobbelst/tools/alphafold3/alphafold3.sif \
        --af3_weights_dir /work/lpdi/users/dobbelst/tools/alphafold3 \
        --af3_db_dir /work/lpdi/databases/alphafold3_dbs \
        --af3_shared_root /work/lpdi \
        --model_seeds '1' \
        --min_top_model_plddt 0.8
