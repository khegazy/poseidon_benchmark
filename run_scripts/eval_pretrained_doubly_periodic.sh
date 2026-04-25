DATA=datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704.h5
MODEL=camlab-ethz/Poseidon-B
RESULTS=experiments/doubly_periodic/Re-50k_Ma-1en1_l-80_eps-5en2_r0-1/results

# Step 1: Multi-GPU scalar evaluation via accelerate
WANDB_MODE=disabled accelerate launch --num_processes 4 -m scOT.inference \
        --model_path $MODEL \
        --dataset kinet.D2Q9.WeaklyCompressible \
        --data_path $DATA \
        --file $RESULTS/eval_metrics.csv \
        --ckpt_dir . \
        --mode eval \
        --batch_size 32

# Step 2: Single-GPU autoregressive rollout (inherently sequential)
WANDB_MODE=disabled python -m scOT.eval_rollout \
        --model_path $MODEL \
        --data_path $DATA \
        --results_dir $RESULTS \
        --skip_scalar_eval
