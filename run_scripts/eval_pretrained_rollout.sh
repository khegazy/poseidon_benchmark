PROJ=/global/u2/k/khegazy/projects/pde/poseidon
DATA=$PROJ/datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704.h5
MODEL=camlab-ethz/Poseidon-B
RESULTS=$PROJ/experiments/doubly_periodic/Re-50k_Ma-1en1_l-80_eps-5en2_r0-1/results

export HDF5_USE_FILE_LOCKING=FALSE
WANDB_MODE=disabled python -m scOT.eval_rollout \
        --model_path $MODEL \
        --data_path $DATA \
        --results_dir $RESULTS \
        --skip_scalar_eval
