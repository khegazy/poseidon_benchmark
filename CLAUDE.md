# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poseidon is a PyTorch-based framework for building and training scalable Operator Transformer (ScOT) models for solving partial differential equations (PDEs). Based on the paper [*Poseidon: Efficient Foundation Models for PDEs*](https://arxiv.org/abs/2405.19101). Pretrained models are hosted on HuggingFace Hub under `camlab-ethz/Poseidon-{T,B,L}`.

This fork fine-tunes Poseidon on kinetic (LBM) simulation data stored in `datasets/kinet/`. The kinet data has 4 channels `[density, u, v, vorticity]` which differ from the pretrained channels `[density, u, v, pressure]`, so fine-tuning requires `--replace_embedding_recovery` to re-initialize the first/last layers.

## Commands

### Install
```bash
pip install -e .
```

### Fine-tune on kinet data
```bash
# Environment (NERSC Perlmutter)
module load conda && conda activate poseidon

# Quick start
bash run_scripts/run_doubly_periodic.sh

# Or submit to SLURM
sbatch run_scripts/submit_doubly_periodic.sh

# Full command
accelerate launch scOT/train.py \
    --config configs/kinet_finetune.yaml \
    --data_path datasets/kinet/doubly_periodic/.../file.h5 \
    --checkpoint_path <CHECKPOINT_DIR> \
    --finetune_from camlab-ethz/Poseidon-L \
    --replace_embedding_recovery \
    --wandb_run_name <NAME>
```
Disable W&B logging with `WANDB_MODE=disabled`.

### Evaluate (scalar metrics)
```bash
WANDB_MODE=disabled python -m scOT.inference \
    --model_path <MODEL_PATH> \
    --dataset kinet.D2Q9.WeaklyCompressible \
    --data_path <HDF5_PATH> \
    --file <OUTPUT_CSV> --ckpt_dir . --mode eval --batch_size 32
```
For multi-GPU: `accelerate launch --num_processes 4 -m scOT.inference ...`

### Evaluate (autoregressive rollout with plots)
```bash
WANDB_MODE=disabled python -m scOT.eval_rollout \
    --model_path <MODEL_PATH> \
    --data_path <HDF5_PATH> \
    --results_dir <RESULTS_DIR>
```
Single-GPU only. Generates comparison plots at milestone steps and saves predictions as `.npy` files.

### Debug run (Poseidon-T, fast iteration)
Use `configs/kinet_debug.yaml` with 128x128 data and Poseidon-T for quick pipeline verification:
```bash
WANDB_MODE=disabled python scOT/train.py \
    --config configs/kinet_debug.yaml \
    --data_path datasets/kinet/...downsampled-128-128.h5 \
    --checkpoint_path experiments/debug/checkpoints \
    --finetune_from camlab-ethz/Poseidon-T \
    --replace_embedding_recovery \
    --wandb_run_name debug_run
```

## Architecture

### Package: `scOT/`

- **model.py** — Core `ScOT` model (extends HuggingFace `Swinv2PreTrainedModel`). Encoder-decoder U-Net architecture with Swin Transformer V2 attention blocks, skip connections, and patch embedding/recovery. Supports autoregressive rollouts, time-dependent conditioning via `ConditionalLayerNorm`, and dynamic resolution via FFT-based up/downsampling. Config class: `ScOTConfig`. Model sizes: T, S, B, L.
- **trainer.py** — Custom HuggingFace `Trainer` subclass with separate learning rate groups (standard params, no-decay params, embedding/recovery params, time embedding params) and autoregressive rollout support.
- **train.py** — Training entry point. Reads YAML configs (see `configs/`), sets up model/dataset/trainer, supports finetuning from pretrained with `--finetune_from` and `--replace_embedding_recovery`.
- **inference.py** — Evaluation script with multiple modes (`eval`, `save_samples`, `eval_accumulation_error`, `eval_resolutions`). `eval` mode computes scalar metrics and writes CSV. Does not produce plots.
- **eval_rollout.py** — Autoregressive rollout evaluation. Runs the model forward from the first timestep of each split, feeding predictions back as input. Saves comparison plots (GT vs prediction vs |error|) at milestones and `.npy` files. Also runs scalar eval unless `--skip_scalar_eval`.
- **metrics.py** — Lp error metrics: `lp_error`, `relative_lp_error`, `mean_relative_lp_error`, `median_relative_lp_error`.
- **utils.py** — CLI argument parsing and parameter counting utilities.

### Dataset Framework: `scOT/problems/`

- **base.py** — `BaseDataset` (steady problems) and `BaseTimeDataset` (time-dependent) abstract classes. Data stored in HDF5 format. Factory function `get_dataset()` maps string identifiers to dataset classes. Default time settings for kinet: `max_num_time_steps=7`, `time_step_size=2` (each AR step advances 2 raw HDF5 timesteps).
- **kinet/lbm.py** — `WeaklyCompressibleDataset` for LBM data. 4 channels: density, u, v, vorticity. Windows long simulations into pseudo-trajectories of `window_size = 7*2+1 = 15` timesteps. Split: 75% train / 10% val / 15% test.
- Other subdirectories: `fluids/`, `elliptic/`, `wave/`, `reaction_diffusion/` — dataset subclasses from the original Poseidon paper.
- Dataset identifier for kinet: `kinet.D2Q9.WeaklyCompressible`.
- To add a new dataset: subclass `BaseTimeDataset`, register in `get_dataset()`.

### Configuration

- **configs/kinet_finetune.yaml** — Production fine-tuning config (Poseidon-L, full data).
- **configs/kinet_debug.yaml** — Fast debug config (Poseidon-T, 1/8 data, 20 epochs).
- **configs/run.yaml** / **configs/sweep.yaml** — Original Poseidon configs for reference.

Key config parameters: `model_name` (must match `--finetune_from`), `lr`, `lr_embedding_recovery`, `lr_time_embedding`, `batch_size`, `num_epochs`, `early_stopping_patience`.

### Run Scripts: `run_scripts/`

- **run_doubly_periodic.sh** — Training command with env vars for Lustre filesystem workarounds.
- **submit_doubly_periodic.sh** — SLURM submission for training (4 GPU, regular queue).
- **eval_pretrained_*.sh** — Evaluation commands for pretrained baseline.
- **submit_eval_pretrained_*.sh** — SLURM submissions for evaluation.

## Parallel Filesystem Workarounds (NERSC Perlmutter)

Perlmutter's Lustre filesystem does not support `flock()`, causing errno 524 errors. Two workarounds are needed:

```bash
export HF_HOME=/tmp/hf_cache           # HuggingFace filelock on local tmpfs
export HDF5_USE_FILE_LOCKING=FALSE      # Disable HDF5 file locking
```

These are set in `run_scripts/run_doubly_periodic.sh` and `scOT/eval_rollout.py`. If you see errno 524, check that these are set.

Additionally, `scOT/eval_rollout.py` forces `CUDA_VISIBLE_DEVICES=0` to avoid HuggingFace Trainer wrapping the model in `DataParallel` (which breaks `model.config` access). This script is single-GPU only by design.

## Working Principles

- **Correctness above all else.** This is scientific code — incorrect results can silently propagate. Always verify math, physics, indexing, and boundary conditions carefully.
- **Ask questions when uncertain.** The problems here may not have existing solutions. When unsure about an approach, ask rather than guess.
- **Understand before modifying.** Read and thoroughly understand code before making changes.
- **Minimal changes.** This is a fork of the ETH Zurich Poseidon codebase. Prefer config files, CLI args, and new files over modifying core model/trainer code.

## Key Dependencies (pinned)

- `torch==2.0.1`, `transformers==4.29.2`, `accelerate==0.31.0`, `wandb==0.14.2`
- Build system: flit

## Further Reading

See [FINETUNING.md](FINETUNING.md) for a comprehensive guide to fine-tuning and evaluation, including hyperparameter descriptions, data details, and a suggested experiment plan.
