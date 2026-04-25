#!/bin/bash
#SBATCH -A m4790_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -t 23:00:00
#SBATCH -J poseidon_finetune
#SBATCH -o run_scripts/logs/%x-%j.out
#SBATCH -e run_scripts/logs/%x-%j.err

module load conda
conda activate poseidon

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

srun --gpus=4 bash run_scripts/run_doubly_periodic.sh
