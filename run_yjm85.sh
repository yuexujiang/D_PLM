#!/bin/bash
#SBATCH --mem 30G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1
#SBATCH --time 05:00:00 #Time for the job to run
#SBATCH --job-name long
#SBATCH -p gpu
##SBATCH -p xudong-gpu
#SBATCH -A xudong-lab


module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /cluster/pixstor/xudong-lab/yuexu/env/splm_gvp_v1

export TORCH_HOME=/cluster/pixstor/xudong-lab/yuexu/torch_cache/
export HF_HOME=/cluster/pixstor/xudong-lab/yuexu/transformers_cache/



accelerate launch train.py --config_path ./configs_hell/gvp_v2/config_plddtallweight_noseq_v2.yaml \
--result_path ./results/plddtallweight_noseq_v2/
