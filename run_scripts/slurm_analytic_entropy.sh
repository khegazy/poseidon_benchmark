#!/bin/bash
#SBATCH -A m4790_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH -t 06:00:00
#SBATCH -J poseidon-analytic
#SBATCH -o /pscratch/sd/j/jalb/poseidon_entropy_analytic/slurm_logs/%x-%j.out
#SBATCH -e /pscratch/sd/j/jalb/poseidon_entropy_analytic/slurm_logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load conda
conda activate kinet

export WANDB_MODE="${WANDB_MODE:-disabled}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export PYTHON="${PYTHON:-python}"

if [ -z "${ANALYTIC_COMMAND:-}" ]; then
    echo "ANALYTIC_COMMAND must be set, e.g. train-tgv-re50k-baseline"
    exit 2
fi

srun --gpus=1 bash run_scripts/analytic_entropy_experiments.sh "${ANALYTIC_COMMAND}"
