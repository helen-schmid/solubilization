# Solubilization

## Overview

An end-to-end pipeline for designing a soluble analogue of a membrane protein using AF2seq

## Installation

First you clone this repository. Replace [install_folder] with the path where you want to install it.

```bash
git clone https://github.com/alexhilditch/solubilization [install_folder]
```

AF2seq requires a CUDA-compatible NVIDIA graphics card to run. This was tested on CUDA version 12.4 on an H100 graphics card.

Installation can be done through Conda, or an alternative package manager. Note that this script installs PyRosetta, which requires a licence for commercial use.

Then follow the following installation steps to create a Conda environment:

```bash
conda create --name Solubilization python=3.10 -y

conda activate Solubilization

CONDA_OVERRIDE_CUDA="12.4" conda install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib=*=*cuda12* jax cuda-nvcc cudnn -c conda-forge -c nvidia  --channel https://conda.rosettacommons.org -y

pip install git+https://github.com/sokrypton/ColabDesign.git --no-deps

pip install alphafold-colabfold
```

We will also need to download the AF2 weights. Note this requires approx 5.3 GB

```bash
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params
```

## Usage

There is an example submission script run_solubilization.sh provided for reference, created for use with a
SLURM job scheduler.

Run the pipeline using the provided SLURM submission script, or directly in the terminal with:

```bash
conda activate Solubilization

python -u solubilization.py --input_pdb 'path/to/input/pdb/1JGJ.pdb' --working_dir '/path/to/output/directory'
```

Note that the backpropagation step is by far the slowest step in the pipeline. 

A good place to start could be to make 10 backbones, and approximately 50 sequences per backbone with MPNN.

Input arguments:

```bash
--working_dir        - the directory to work in
--input_pdb          - the path to the input pdb file
--fix_pos            - the positions to be fixed during design
--sidechain_loss_pos - the positions to apply the sidechain loss on 
--num_backbones      - the number of backbones to generate by AF2 backpropagation
--nseq               - the number of MPNN generated sequences to sample and predict per backbone
--chain_id           - the chain ID to be designed
--params             - the path to ColabDesign AF params
--mpnn_temp          - the sampling temperature for MPNN: T=0.0 means taking argmax, T>>1.0 means sampling randomly
--mpnn_model         - the model weights to use for MPNN, the soluble model is selected by default
--backbone_noise     - the level of backbone noise for mpnn
```

## Inputs

The only input is a protein structure in PDB format. You need to assign positions to be fixed during design if necessary (optional), and positions which you would like to assign the sidechain loss on (optional).

## Filtering

`filtering.py` takes the designs from `02_final_prediction_scores.csv` (stage 4 of
`solubilization.py`) and cofolds each one with its native ligand using AlphaFold3, to
check whether the solubilized analogue still folds a pocket that binds the ligand.
There is an example submission script `run_filtering.sh` provided for reference,
created for use with a SLURM job scheduler on a cluster with a shared AF3 install
(container + weights + databases, see `--af3_sif`/`--af3_weights_dir`/`--af3_db_dir`
below).

Run the pipeline using the provided SLURM submission script, or directly in the
terminal with:

```bash
conda activate Solubilization

python -u filtering.py --working_dir '/path/to/output/directory' --input_csv '/path/to/02_final_prediction_scores.csv' --ligand_smiles 'ligand SMILES string' --af3_sif '/path/to/alphafold3.sif' --af3_weights_dir '/path/to/af3/weights' --af3_db_dir '/path/to/af3/databases' --af3_shared_root '/path/to/shared/storage/root'
```

Input arguments:

```bash
--working_dir         - the directory to work in
--input_csv           - path to the stage-04 scores csv from solubilization.py
--ligand_smiles       - SMILES string of the native ligand to cofold each design with
--af3_sif             - path to the AF3 singularity container (.sif)
--af3_weights_dir     - path to the AF3 code + model weights directory
--af3_db_dir          - path to the AF3 genetic/template databases
--af3_shared_root     - shared storage root to bind-mount into the container
--protein_chain_id    - chain ID to assign the designed protein in the AF3 input json (default=A)
--ligand_chain_id     - chain ID to assign the ligand in the AF3 input json (default=B)
--model_seeds         - comma-separated AF3 model seeds to run per design (default=1)
--min_top_model_plddt - skip designs below this stage-04 top_model_plddt before cofolding (default=0.0, i.e. no filtering)
```

Scores are written to `03_af3_scores.csv`: `af3_iptm`, `af3_ptm`, `af3_ranking_score`,
`af3_chain_pair_pae_min`, `af3_fraction_disordered`, `af3_has_clash`. As with the rest
of the pipeline, there is no automatic pass/fail filtering - use `--min_top_model_plddt`
to control how many (expensive, live-MSA) cofolds get submitted, and apply your own
cutoffs on the resulting scores.

## Scores

Each of the designed sequences is evaluated by Alphafold2 and in PyRosetta creating the following scores:

```bash
pLDDT               - mean pLDDT confidence score of AF2 complex prediction, normalised to 0-1
top_model_pLDDT     - pLDDT confidence score of the best AF2 model
pAE                 - predicted alignment error of AF2 complex prediction, normalised compared AF2 by n/31 to 0-1
pTM                 - pTM confidence score of AF2 complex prediction, normalised to 0-1
Ca_RMSD             - The Ca root mean square devation between the final AF2 model and the input PDB
Surf_apolar_frac    - The surface hydrophobicity fraction for the designed protein
```

There is no filtering implimented in the pipeline, and this is up to the user to decide. However, a good place to start would
be to select for models with a high pLDDT (>80) and a low RMSD to the input (ideally, <2 or 3).

## Acknowledgements

Thanks to Casper Goverde and Nicolas Goldbach for help with code generation. In addition, this repository uses code from: 

- Sergey Ovchinnikov's ColabDesign (https://github.com/sokrypton/ColabDesign)
- Martin Pacesa's BindCraft (https://github.com/martinpacesa/BindCraft)
- Justas Dauparas's ProteinMPNN (https://github.com/dauparas/ProteinMPNN)
- PyRosetta (https://github.com/RosettaCommons/PyRosetta.notebooks)