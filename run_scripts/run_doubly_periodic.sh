
# HuggingFace cache on local tmpfs to avoid filelock failures on Lustre
export HF_HOME=/tmp/hf_cache
# Disable HDF5 file locking (required on parallel filesystems)
export HDF5_USE_FILE_LOCKING=FALSE

#WANDB_MODE=disabled
accelerate launch scOT/train.py \
        --config configs/kinet_finetune.yaml \
        --data_path datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704.h5 \
        --checkpoint_path experiments/doubly_periodic/Re-50k_Ma-1en1_l-80_eps-5en2_r0-1/checkpoints \
        --finetune_from camlab-ethz/Poseidon-L \
        --replace_embedding_recovery \
        --wandb_run_name doubly_periodic_b6e704 \
        --disable_tqdm
