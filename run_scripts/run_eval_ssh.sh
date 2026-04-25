#!/bin/bash
module load conda
conda activate poseidon
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=0
cd /global/u2/k/khegazy/projects/pde/poseidon
WANDB_MODE=disabled python -m scOT.eval_rollout \
    --model_path camlab-ethz/Poseidon-B \
    --data_path datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704.h5 \
    --results_dir experiments/doubly_periodic/Re-50k_Ma-1en1_l-80_eps-5en2_r0-1/results
