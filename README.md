# solubilization

## Overview

A simple end-to-end pipeline for designing a soluble analogue of a membrane protein using AF2seq

## Installation

AF2seq requires a CUDA-compatible NVIDIA graphics card to run. This was tested on CUDA version 12.4 on an H100 graphics card.

Installation can be done through conda, or an alternative package manager.

Note that this script installs PyRosetta, which requires a licence for commercial use.

```bash
conda create --name Solubilization python=3.10 -y

conda activate Solubilization

conda install pip pandas matplotlib numpy"<2.0.0" biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg pyrosetta fsspec py3dmol chex dm-haiku flax"<0.10.0" dm-tree joblib ml-collections immutabledict optax jaxlib=*=*cuda* jax cuda-nvcc cudnn -c conda-forge -c nvidia  --channel https://conda.graylab.jhu.edu -y

pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps

pip install fsspec
pip install py3Dmol
pip install alphafold-colabfold

# download AF2 weights to a params directory - approx 5.3 GB
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params
```

## Usage

There is an example submission script run_solubilization.sh provided for reference, created for use with a
SLURM job scheduler.

Run the pipeline using:

```bash
python solubilization.py
```

Note that the backpropagation step is by far the slowest step in the pipeline. 

A good place to start could be to make 10 backbones, and approximately 50 sequences per backbone with MPNN.

Input arguments:

```bash
-working_dir        - the directory to work in
-input_pdb          - the path to the input pdb file
-fix_pos            - the positions to be fixed during design
-sidechain_loss_pos - the positions to apply the sidechain loss on 
-num_backbones      - the number of backbones to generate by AF2 backpropagation
-nseq               - the number of MPNN generated sequences to sample and predict per backbone
-chain_id           - the chain ID to be designed
-params             - the path to ColabDesign AF params
-mpnn_temp          - the sampling temperature for MPNN: T=0.0 means taking argmax, T>>1.0 means sampling randomly
-mpnn_model         - the model weights to use for MPNN, the soluble model is selected by default
-backbone_noise     - the level of backbone noise for mpnn
```

## Inputs

The only input is a protein structure in PDB format. You need to assign positions to be fixed during design if necessary (optional), and positions which you would like to assign the sidechain loss on (optional).

## Filters

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