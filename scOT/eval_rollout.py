"""
Evaluate a model by performing long autoregressive rollouts starting from
the first timestep of each dataset split (train, val, test).

Generates comparison plots (ground truth vs prediction) at milestones:
  - Start (step 0, initial condition)
  - Each eighth of the split (12.5%, 25%, ..., 87.5%)
  - End (last AR step)

Also runs single-step scalar metric evaluation (same as inference.py --mode eval)
and saves the CSV alongside the plots in the results directory.

Usage:
    python -m scOT.eval_rollout \
        --model_path camlab-ethz/Poseidon-B \
        --data_path datasets/kinet/.../file.h5 \
        --results_dir experiments/.../results
"""

import argparse
import os
import shutil

# Disable HDF5 file locking (required on parallel filesystems like Lustre/GPFS)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Restrict to single GPU (this script is single-GPU; avoids DataParallel issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scOT.model import ScOT
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.metrics import relative_lp_error, lp_error


DATASET_NAME = "kinet.D2Q9.WeaklyCompressible"
CHANNEL_NAMES = ["density", "u", "v", "vorticity"]
TIME_STEP_SIZE = 2
MAX_NUM_TIME_STEPS = 7


def read_raw_state(reader, t):
    """Read raw (unnormalized) state at absolute timestep t from HDF5."""
    den = reader["density"][0, :, t, :, :]    # (1, H, W)
    vel = reader["velocity"][0, :, t, :, :]   # (2, H, W)
    vort = reader["vorticity"][0, :, t, :, :] # (1, H, W)
    return np.concatenate([den, vel, vort], axis=0)  # (4, H, W)


def normalize(raw, mean, std):
    return (torch.from_numpy(raw).float() - mean) / std


def denormalize(normalized, mean, std):
    return (normalized * std + mean).numpy()


def get_split_info(N_max, N_val, N_test, window_size):
    """Return start timestep and total raw timesteps for each split."""
    N_train = N_max - N_val - N_test
    return {
        "train": {
            "start_t": 0,
            "n_windows": N_train,
            "total_raw_t": N_train * window_size,
        },
        "val": {
            "start_t": N_train * window_size,
            "n_windows": N_val,
            "total_raw_t": N_val * window_size,
        },
        "test": {
            "start_t": (N_max - N_test) * window_size,
            "n_windows": N_test,
            "total_raw_t": N_test * window_size,
        },
    }


def get_milestones(n_ar_steps):
    """Compute the set of AR steps at which to save results."""
    milestones = set([i for i in range(25)])
    for k in range(1, 8):                    # each 1/8
        milestones.add(k * n_ar_steps // 8)
    milestones.add(n_ar_steps)               # end
    return sorted(milestones)


def save_comparison_plot(pred, gt, split_name, ar_step, abs_t, n_ar_steps,
                         results_dir, channel_names=CHANNEL_NAMES):
    """
    Save a figure comparing ground truth and prediction for all channels.

    Layout: 3 rows x N_channels cols
      Row 0: Ground truth
      Row 1: Prediction
      Row 2: Absolute error
    """
    n_ch = len(channel_names)
    fig, axes = plt.subplots(3, n_ch, figsize=(4 * n_ch, 10))

    for c in range(n_ch):
        gt_c = gt[c]
        pred_c = pred[c]
        err_c = np.abs(pred_c - gt_c)

        vmin = min(gt_c.min(), pred_c.min())
        vmax = max(gt_c.max(), pred_c.max())

        im0 = axes[0, c].imshow(gt_c, cmap="RdBu_r", origin="lower",
                                vmin=vmin, vmax=vmax)
        axes[0, c].set_title(f"GT: {channel_names[c]}")
        axes[0, c].set_xticks([])
        axes[0, c].set_yticks([])
        fig.colorbar(im0, ax=axes[0, c], fraction=0.046, pad=0.04)

        im1 = axes[1, c].imshow(pred_c, cmap="RdBu_r", origin="lower",
                                vmin=vmin, vmax=vmax)
        axes[1, c].set_title(f"Pred: {channel_names[c]}")
        axes[1, c].set_xticks([])
        axes[1, c].set_yticks([])
        fig.colorbar(im1, ax=axes[1, c], fraction=0.046, pad=0.04)

        im2 = axes[2, c].imshow(err_c, cmap="hot", origin="lower")
        axes[2, c].set_title(f"|Error|: {channel_names[c]}")
        axes[2, c].set_xticks([])
        axes[2, c].set_yticks([])
        fig.colorbar(im2, ax=axes[2, c], fraction=0.046, pad=0.04)

    pct = 100 * ar_step / n_ar_steps if n_ar_steps > 0 else 0
    fig.suptitle(
        f"{split_name} — AR step {ar_step}/{n_ar_steps} "
        f"({pct:.0f}%) — abs timestep {abs_t}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    plot_dir = os.path.join(results_dir, split_name)
    os.makedirs(plot_dir, exist_ok=True)
    fname = f"rollout_step{ar_step:05d}_t{abs_t}.png"
    fig.savefig(os.path.join(plot_dir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_scalar_eval(model_path, data_path, results_dir):
    """Run single-step scalar metric evaluation (same as inference.py --mode eval)."""
    from scOT.inference import get_test_set, get_trainer, rollout, remove_underscore_dict
    import pandas as pd

    dataset = get_test_set(DATASET_NAME, data_path)
    trainer = get_trainer(model_path, batch_size=32, dataset=dataset)
    _, _, metrics = rollout(trainer, dataset, ar_steps=1)

    data = [remove_underscore_dict({
        "dataset": DATASET_NAME,
        "initial_time": None,
        "final_time": None,
        "ar_steps": 1,
        **metrics,
    })]
    df = pd.DataFrame(data)
    csv_path = os.path.join(results_dir, "eval_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Scalar metrics saved to {csv_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Autoregressive rollout evaluation with comparison plots."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF Hub model ID or local checkpoint path.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Full path to the HDF5 data file.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory to save plots and metrics.")
    parser.add_argument("--skip_scalar_eval", action="store_true",
                        help="Skip single-step scalar metric evaluation.")
    args = parser.parse_args()

    # Resolve relative paths against the project root (parent of scOT/)
    # so the script works regardless of cwd (e.g. when run via SSH)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(project_root, args.data_path)
    if not os.path.isabs(args.results_dir):
        args.results_dir = os.path.join(project_root, args.results_dir)

    os.makedirs(args.results_dir, exist_ok=True)

    # --- Scalar evaluation ---
    if not args.skip_scalar_eval:
        print("=== Single-step scalar evaluation ===")
        run_scalar_eval(args.model_path, args.data_path, args.results_dir)

    # --- Setup for rollout ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScOT.from_pretrained(args.model_path).to(device).eval()

    # Load a dataset instance to get normalization constants and split sizes
    ref_dataset = get_dataset(
        DATASET_NAME, which="test", num_trajectories=1,
        data_path=args.data_path,
    )
    mean = ref_dataset.constants["mean"]  # (4, 1, 1)
    std = ref_dataset.constants["std"]    # (4, 1, 1)
    max_time = ref_dataset.constants["time"]  # 14.0
    T_total = ref_dataset.T_total
    window_size = ref_dataset.window_size
    N_max = ref_dataset.N_max
    N_val = ref_dataset.N_val
    N_test = ref_dataset.N_test

    time_conditioning = TIME_STEP_SIZE / max_time  # 2/14

    splits = get_split_info(N_max, N_val, N_test, window_size)

    reader = h5py.File(args.data_path, "r")
    # pixel_mask must be None during inference (no labels to fill masked channels)
    # All channels are False (unmasked) for this dataset anyway.

    # --- Rollout per split ---
    print("\n=== Autoregressive rollout evaluation ===")
    for split_name, info in splits.items():
        start_t = info["start_t"]
        total_raw_t = info["total_raw_t"]
        n_ar_steps = (total_raw_t - 1) // TIME_STEP_SIZE

        milestones = get_milestones(n_ar_steps)
        milestone_set = set(milestones)

        # Clean previous results to avoid stale files
        plot_dir = os.path.join(args.results_dir, split_name)
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)

        print(f"\n--- {split_name}: {n_ar_steps} AR steps "
              f"(abs t {start_t}..{start_t + n_ar_steps * TIME_STEP_SIZE}) ---")
        print(f"    Plot milestones: {milestones}")

        # Read initial condition
        raw_init = read_raw_state(reader, start_t)
        state = normalize(raw_init, mean, std).unsqueeze(0).to(device)  # (1,4,H,W)

        time_tensor = torch.tensor([time_conditioning], dtype=torch.float32,
                                   device=device)

        for step in range(n_ar_steps + 1):
            if step in milestone_set:
                abs_t = start_t + step * TIME_STEP_SIZE
                # Clamp to valid range
                abs_t = min(abs_t, T_total - 1)

                # Ground truth (denormalized)
                gt_raw = read_raw_state(reader, abs_t)  # (4, H, W)

                # Prediction (denormalize from current state)
                pred_raw = denormalize(state[0].cpu(), mean, std)  # (4, H, W)

                save_comparison_plot(
                    pred_raw, gt_raw, split_name, step, abs_t,
                    n_ar_steps, args.results_dir,
                )

                # Save prediction and ground truth as numpy arrays
                npy_dir = os.path.join(args.results_dir, split_name, "npy")
                os.makedirs(npy_dir, exist_ok=True)
                np.save(os.path.join(npy_dir, f"pred_step{step:05d}_t{abs_t}.npy"),
                        pred_raw)
                np.save(os.path.join(npy_dir, f"gt_step{step:05d}_t{abs_t}.npy"),
                        gt_raw)

                # Per-channel relative L1 error at this milestone
                channel_slices = [0, 1, 3, 4]
                errors = []
                for i in range(len(channel_slices) - 1):
                    s = channel_slices[i]
                    e = channel_slices[i + 1]
                    err = np.abs(pred_raw[s:e] - gt_raw[s:e]).mean()
                    gt_norm = np.abs(gt_raw[s:e]).mean()
                    rel_err = 100 * err / gt_norm if gt_norm > 0 else float("inf")
                    errors.append(rel_err)
                print(f"    step {step:5d} (t={abs_t:6d}): "
                      f"density={errors[0]:6.1f}%  uv={errors[1]:6.1f}%  "
                      f"vorticity={errors[2]:6.1f}%")

            # Advance one AR step
            if step < n_ar_steps:
                with torch.no_grad():
                    outputs = model(
                        pixel_values=state,
                        time=time_tensor,
                    )
                    state = outputs.output.detach()

    reader.close()
    print(f"\nResults saved to {args.results_dir}")


if __name__ == "__main__":
    main()
