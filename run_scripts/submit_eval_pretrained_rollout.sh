#!/bin/bash
#SBATCH -A m4790_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J poseidon_eval_rollout
#SBATCH -o run_scripts/logs/%x-%j.out
#SBATCH -e run_scripts/logs/%x-%j.err

module load conda
conda activate poseidon

bash run_scripts/eval_pretrained_rollout.sh
