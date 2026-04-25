# Fine-Tuning Poseidon on Kinet (LBM) Data

This guide walks through fine-tuning a pretrained [Poseidon](https://arxiv.org/abs/2405.19101) model on lattice Boltzmann method (LBM) simulation data and evaluating the results. It assumes you are working on **NERSC Perlmutter** and have the `poseidon` conda environment set up.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Training](#2-training)
3. [Evaluation](#3-evaluation)
4. [Configuration Reference](#4-configuration-reference)
5. [Data Details](#5-data-details)
6. [Suggested Experiment Plan](#6-suggested-experiment-plan)

---

## 1. Prerequisites

### Environment setup

```bash
module load conda
conda activate poseidon
```

### Install the package (one-time)

From the repo root:

```bash
pip install -e .
```

### Accelerate setup (one-time, for multi-GPU)

```bash
accelerate config
```

Recommended answers:
- This machine (not distributed across nodes)
- multi-GPU
- How many GPUs? **4** (on a Perlmutter GPU node)
- DeepSpeed? **No**
- FullyShardedDataParallel? **No**
- Megatron-LM? **No**
- Mixed precision? **bf16**

This writes `~/.cache/huggingface/accelerate/default_config.yaml`. You only need to do this once.

### Data

The kinet HDF5 files should be in `datasets/kinet/`. Each file contains a single long LBM simulation with 4 channels: density, u-velocity, v-velocity, and vorticity.

Available files in `datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/`:

| File | Resolution | Timesteps |
|------|-----------|-----------|
| `D2Q9_shape-256-256_T-10000_H-b6e704.h5` | 256x256 | 10001 |
| `..._downsampled-128-128.h5` | 128x128 | 5001 |
| `..._downsampled-64-64.h5` | 64x64 | — |

The 128x128 file is recommended for initial experiments (~4x less GPU memory per sample).

---

## 2. Training

### How it works

Fine-tuning loads a pretrained Poseidon model from [HuggingFace Hub](https://huggingface.co/camlab-ethz) and continues training on the kinet data. Because the kinet channels (`[density, u, v, vorticity]`) differ from the pretrained channels (`[density, u, v, pressure]`), the embedding and recovery (first/last) layers must be re-initialized with `--replace_embedding_recovery`.

The long simulation is windowed into pseudo-trajectories of 15 consecutive timesteps each. Within each window, the model trains on all valid `(input_time, output_time)` pairs spaced by multiples of `time_step_size=2`. The data is split 75% train / 10% val / 15% test.

Early stopping monitors validation loss and saves the best model automatically.

### Quick start

Run the training script from the repo root:

```bash
bash run_scripts/run_doubly_periodic.sh
```

Or submit to SLURM:

```bash
sbatch run_scripts/submit_doubly_periodic.sh
```

### Debug run (Poseidon-T)

For debugging or testing pipeline changes, use Poseidon-T (~3M params) with the 128x128 downsampled data. This runs on a single GPU in minutes rather than hours.

1. Temporarily override `model_name` and training duration in the config, or create a separate debug config:

    ```yaml
    # configs/kinet_debug.yaml
    dataset:
      value: "kinet.D2Q9.WeaklyCompressible"
    num_trajectories:
      value: -8            # use only 1/8 of the training data
    model_name:
      value: "T"
    lr:
      value: 0.0001
    lr_embedding_recovery:
      value: 0.001
    lr_time_embedding:
      value: 0.001
    weight_decay:
      value: 0.000001
    lr_scheduler:
      value: "cosine"
    warmup_ratio:
      value: 0.0
    early_stopping_patience:
      value: 10
    num_epochs:
      value: 20
    batch_size:
      value: 16
    max_grad_norm:
      value: 5.0
    ```

2. Run on a single GPU (no `accelerate config` needed):

    ```bash
    WANDB_MODE=disabled python scOT/train.py \
        --config configs/kinet_debug.yaml \
        --data_path datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704_downsampled-128-128.h5 \
        --checkpoint_path experiments/debug/checkpoints \
        --finetune_from camlab-ethz/Poseidon-T \
        --replace_embedding_recovery \
        --wandb_run_name debug_run
    ```

Key differences from a production run:
- **Poseidon-T** instead of L (~200x fewer parameters, fits on any GPU)
- **128x128 data** instead of 256x256 (~4x less memory per sample)
- **`num_trajectories: -8`** uses only 1/8 of the training windows (faster epochs)
- **20 epochs** with patience 10 — enough to verify the loss is decreasing
- **`batch_size: 16`** — Poseidon-T is small enough for larger batches

If the loss decreases on both train and val, the pipeline is working correctly. The model quality won't be meaningful at this size — it's purely for verifying that training runs end-to-end.

### Full training command

```bash
accelerate launch scOT/train.py \
    --config configs/kinet_finetune.yaml \
    --data_path datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704.h5 \
    --checkpoint_path <CHECKPOINT_DIR> \
    --finetune_from camlab-ethz/Poseidon-L \
    --replace_embedding_recovery \
    --wandb_run_name <RUN_NAME> \
    --disable_tqdm
```

- Prefix with `WANDB_MODE=disabled` to skip Weights & Biases logging.
- For single-GPU, replace `accelerate launch` with `python`.
- For inline GPU count (no accelerate config), add `--num_processes 4` after `accelerate launch`.

### Key arguments

| Argument | Description |
|----------|-------------|
| `--config` | YAML config file with hyperparameters (see [Section 4](#4-configuration-reference)) |
| `--data_path` | Full path to the HDF5 data file (not a directory) |
| `--checkpoint_path` | Base directory for saving model checkpoints |
| `--finetune_from` | HuggingFace model ID or local path to pretrained weights |
| `--replace_embedding_recovery` | Re-initialize embedding/recovery layers (required when channels differ from pretrained) |
| `--wandb_run_name` | Name for the W&B run and checkpoint subdirectory |
| `--disable_tqdm` | Disable progress bars (cleaner batch job logs) |

### Where checkpoints are saved

The best model (by validation loss) is saved to:

```
<CHECKPOINT_DIR>/scOT/<WANDB_RUN_NAME>/
```

This directory contains `config.json` and `model.safetensors`, and can be passed directly to evaluation scripts.

### Multi-GPU notes

- `accelerate launch` uses DistributedDataParallel under the hood.
- Effective batch size = `batch_size` (per device) x number of GPUs.
- Only rank 0 logs to W&B and saves checkpoints.
- Data loading is automatically distributed across GPUs.

### Perlmutter / Lustre filesystem issues

Perlmutter's parallel filesystem does not support file locking. Two environment variables must be set before running:

```bash
export HF_HOME=/tmp/hf_cache           # HuggingFace cache on local tmpfs (avoids filelock errors)
export HDF5_USE_FILE_LOCKING=FALSE      # Disables HDF5 file locking
```

These are already included in `run_scripts/run_doubly_periodic.sh`.

---

## 3. Evaluation

Training automatically runs test-set evaluation after finishing (single-step and 7-step autoregressive rollout). The sections below are for running evaluation separately.

### 3a. Single-step scalar metrics

Computes relative L1 errors per channel group (density, uv, vorticity) on the test set and writes a CSV.

```bash
WANDB_MODE=disabled python -m scOT.inference \
    --model_path <MODEL_PATH> \
    --dataset kinet.D2Q9.WeaklyCompressible \
    --data_path <HDF5_PATH> \
    --file <OUTPUT_CSV> \
    --ckpt_dir . \
    --mode eval \
    --batch_size 32
```

`--model_path` accepts either a HuggingFace model ID (e.g., `camlab-ethz/Poseidon-L`) for evaluating the pretrained model, or a local checkpoint directory for evaluating a fine-tuned model.

For multi-GPU evaluation, use `accelerate launch --num_processes 4 -m scOT.inference ...`.

### 3b. Autoregressive rollout with plots

`scOT/eval_rollout.py` performs a long autoregressive rollout from the first timestep of each split (train, val, test), feeding predictions back as input at each step. It generates:

- **Comparison plots** (ground truth vs prediction vs |error|) at milestone steps
- **NumPy arrays** of predictions and ground truth at each milestone
- **Scalar metrics CSV** (same as 3a, unless `--skip_scalar_eval` is passed)

Milestone steps are: step 0, each eighth of the rollout (12.5%, 25%, ..., 87.5%), and the final step.

```bash
WANDB_MODE=disabled python -m scOT.eval_rollout \
    --model_path <MODEL_PATH> \
    --data_path <HDF5_PATH> \
    --results_dir <RESULTS_DIR>
```

Output structure:

```
<RESULTS_DIR>/
  eval_metrics.csv
  train/
    rollout_step00000_t0.png
    rollout_step00062_t124.png
    ...
    npy/
      pred_step00000_t0.npy
      gt_step00000_t0.npy
      ...
  val/
    ...
  test/
    ...
```

This script is single-GPU only. On Perlmutter, submit via:

```bash
sbatch run_scripts/submit_eval_pretrained_rollout.sh
```

---

## 4. Configuration Reference

All hyperparameters are set in `configs/kinet_finetune.yaml` (YAML with nested `value:` format).

### Model size

```yaml
model_name:
  value: "L"
```

| Size | Embed dim | Approx. params | Notes |
|------|-----------|----------------|-------|
| `"T"` (Tiny) | 48 | ~3M | Fast experiments |
| `"S"` (Small) | 48 | ~5M | |
| `"B"` (Base) | 96 | ~158M | Good balance |
| `"L"` (Large) | 192 | ~630M | Most capacity, most memory |

**Important:** `model_name` must match the pretrained model passed to `--finetune_from`. If the config says `"L"`, use `camlab-ethz/Poseidon-L`. A mismatch causes shape errors and silently loses pretrained weights.

### Learning rates

These are the most important hyperparameters for fine-tuning:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.00005 | Backbone (attention, MLP, norms). Keep small to preserve pretrained representations. Range: 1e-6 to 5e-4. |
| `lr_embedding_recovery` | 0.0005 | Embedding/recovery layers (randomly initialized due to `--replace_embedding_recovery`). Needs higher LR. Range: 1e-4 to 5e-3. |
| `lr_time_embedding` | 0.0005 | Time-conditioning layers (ConditionalLayerNorm). Range: 1e-4 to 5e-3. |
| `lr_scheduler` | "cosine" | Options: "cosine", "linear", "constant". |
| `warmup_ratio` | 0.0 | Fraction of steps for linear warmup. Try 0.05-0.1 if training is unstable. |

### Regularization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weight_decay` | 0.000001 | L2 regularization. Range: 1e-7 to 1e-4. |
| `max_grad_norm` | 5.0 | Gradient clipping. Lower (1.0-2.0) = more stable but slower. |

### Training duration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 200 | Maximum epochs (early stopping usually ends training sooner). |
| `early_stopping_patience` | 200 | Epochs without val improvement before stopping. Set lower (20-50) for faster iteration. |

### Batch size

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4 | Per-device batch size. Reduce if OOM. For 256x256 on A100 80GB, 4 is typical for Poseidon-L. |

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | "kinet.D2Q9.WeaklyCompressible" | Dataset identifier. |
| `num_trajectories` | -1 | Training pseudo-trajectories: -1 = all, -2 = half, -8 = one eighth, or a positive integer. |

### Additional CLI arguments (not in config)

| Argument | Description |
|----------|-------------|
| `--max_num_train_time_steps N` | Override default max_num_time_steps (7). Fewer = more windows but shorter temporal range. |
| `--train_time_step_size N` | Override default time_step_size (2). Larger = bigger time gaps between input/output. |
| `--train_small_time_transition` | Train on next-step prediction only (allowed_time_transitions=[1]). |
| `--move_data <scratch_dir>` | Copy data to local scratch for faster I/O. |

---

## 5. Data Details

### Format

The HDF5 file contains a single long LBM simulation with variables:

| HDF5 key | Shape | Description |
|----------|-------|-------------|
| `density` | (1, 1, T, H, W) | Mass density |
| `velocity` | (1, 2, T, H, W) | Velocity components (u, v) |
| `vorticity` | (1, 1, T, H, W) | Vorticity |

These are concatenated into 4 model channels: `[density, u, v, vorticity]`.

### Windowing

The simulation is sliced into pseudo-trajectories (windows):

```
window_size = max_num_time_steps * time_step_size + 1 = 7 * 2 + 1 = 15 timesteps
```

Each window spans 15 consecutive HDF5 timesteps. Within a window, the model trains on all `(t_in, t_out)` pairs where both are multiples of `time_step_size` and `t_out > t_in`, giving 36 training pairs per window.

### Splits

```
N_max = T_total // window_size
Train: 75%   Val: 10%   Test: 15%
```

### Normalization

Data is z-score normalized per channel: `(x - mean) / std`, where mean and std are computed from the training split.

---

## 6. Suggested Experiment Plan

1. **Sanity check** — Use `model_name: "T"`, the 128x128 data, and `num_epochs: 20`. Verify the loss decreases on both train and val.

2. **Baseline fine-tune** — Use the default config with Poseidon-L. Train on 128x128 data first, then 256x256 if results look promising.

3. **Learning rate sweep** — The three learning rates are the most impactful hyperparameters:
   - `lr`: {1e-5, 5e-5, 1e-4}
   - `lr_embedding_recovery`: {5e-4, 1e-3, 5e-3}

4. **Evaluate** — Test evaluation runs automatically after training. For separate evaluation or rollout plots, see [Section 3](#3-evaluation).
