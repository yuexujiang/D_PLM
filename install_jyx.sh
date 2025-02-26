#!/bin/bash

#todo: need to be tested

module load miniconda3
#export CONDA_ENVS_PATH=/cluster/pixstor/xudong-lab/yuexu/env
#conda create --name splm_gvp_v1 python=3.10 
#source activate splm_gvp_v1
#pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics
pip install matplotlib
pip install accelerate
pip install transformers
pip install timm
pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
pip install tensorboard
pip install python-box
pip install pandas
pip install peft
pip install -U scikit-learn
pip install fair-esm
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch_geometric
pip install h5py
pip install scipy
pip install openpyxl
pip install mdtraj
pip install biopython
