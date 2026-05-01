"""Long autoregressive rollout evaluation for analytic TGV and Burgers tests."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scOT.model import ScOT
from scOT.problems.base import get_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["tgv", "burgers"], required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--entropy-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--viscosity", type=float, default=0.01)
    parser.add_argument("--substeps-per-step", type=int, default=1)
    parser.add_argument("--snapshot-steps", default=None)
    parser.add_argument("--burgers-direction", default="x")
    parser.add_argument("--burgers-case-mix", default="shock")
    parser.add_argument("--max-abs-cap", type=float, default=1.0e6)
    return parser.parse_args()


def build_dataset(args):
    if args.problem == "tgv":
        dt = 0.05 if args.dt is None else args.dt
        return get_dataset(
            "fluids.incompressible.TaylorGreenVortex",
            which="test",
            num_trajectories=1,
            data_path=args.data_path,
            max_num_time_steps=20,
            time_step_size=1,
            allowed_time_transitions=[1],
            resolution=args.resolution,
            dt=dt,
            viscosity=args.viscosity,
            just_velocities=True,
        )

    dt = 0.01 if args.dt is None else args.dt
    return get_dataset(
        "scalar.Burgers2D",
        which="test",
        num_trajectories=1,
        data_path=args.data_path,
        max_num_time_steps=20,
        time_step_size=1,
        allowed_time_transitions=[1],
        resolution=args.resolution,
        dt=dt,
        direction=args.burgers_direction,
        case_mix=args.burgers_case_mix,
    )


def normalize(dataset, state):
    mean = dataset.constants["mean"].to(device=state.device, dtype=state.dtype)
    std = dataset.constants["std"].to(device=state.device, dtype=state.dtype)
    return (state - mean) / std


def denormalize(dataset, state):
    mean = dataset.constants["mean"].to(device=state.device, dtype=state.dtype)
    std = dataset.constants["std"].to(device=state.device, dtype=state.dtype)
    return state * std + mean


def relative_l1(pred, truth):
    return (
        (pred - truth).abs().sum()
        / torch.clamp(truth.abs().sum(), min=torch.as_tensor(1.0e-12, device=truth.device))
    ).item()


def relative_l2(pred, truth):
    return (
        torch.linalg.vector_norm(pred - truth)
        / torch.clamp(torch.linalg.vector_norm(truth), min=1.0e-12)
    ).item()


def grad_periodic(field, dx, dy):
    grad_x = (torch.roll(field, -1, dims=-1) - torch.roll(field, 1, dims=-1)) / (
        2.0 * dx
    )
    grad_y = (torch.roll(field, -1, dims=-2) - torch.roll(field, 1, dims=-2)) / (
        2.0 * dy
    )
    return grad_x, grad_y


def tgv_quantities(state, dx, dy):
    u = state[0]
    v = state[1]
    dudx, dudy = grad_periodic(u, dx, dy)
    dvdx, dvdy = grad_periodic(v, dx, dy)
    omega = dvdx - dudy
    div = dudx + dvdy
    cell_area = dx * dy
    energy = 0.5 * (u.pow(2) + v.pow(2)).sum() * cell_area
    enstrophy = 0.5 * omega.pow(2).sum() * cell_area
    divergence_l2 = torch.sqrt((div.pow(2).mean()))
    return {
        "energy": energy.item(),
        "enstrophy": enstrophy.item(),
        "divergence_l2": divergence_l2.item(),
        "omega": omega.detach().cpu(),
    }


def burgers_quantities(state, dx, dy):
    u = state[0]
    diff_x = torch.roll(u, -1, dims=-1) - u
    diff_y = torch.roll(u, -1, dims=-2) - u
    entropy = 0.5 * u.pow(2).sum() * dx * dy
    tv = (diff_x.abs() * dy + diff_y.abs() * dx).sum()
    return {
        "entropy": entropy.item(),
        "total_variation": tv.item(),
        "max": u.max().item(),
        "min": u.min().item(),
    }


def load_models(args, device):
    checkpoints = {"baseline": args.baseline_checkpoint}
    if args.entropy_checkpoint is not None:
        checkpoints["entropy"] = args.entropy_checkpoint

    models = {}
    for name, checkpoint in checkpoints.items():
        model = ScOT.from_pretrained(checkpoint)
        model.to(device)
        model.eval()
        models[name] = model
    return models


def snapshot_steps(args):
    if args.snapshot_steps:
        return sorted({int(x) for x in args.snapshot_steps.split(",")})
    candidates = [0, 1, 2, 5, 10, 20, 50, 100, args.steps // 2, args.steps]
    return sorted({step for step in candidates if 0 <= step <= args.steps})


def model_step(model, state, dt, substeps):
    out = state
    step_dt = dt / float(substeps)
    for _ in range(substeps):
        time = torch.full(
            (out.shape[0],),
            step_dt,
            device=out.device,
            dtype=out.dtype,
        )
        with torch.no_grad():
            out = model(pixel_values=out, time=time).output
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output_dir, rows, problem):
    names = sorted({row["model"] for row in rows if row["model"] != "truth"})
    steps = sorted({row["step"] for row in rows})

    def series(model, key):
        by_step = {row["step"]: row[key] for row in rows if row["model"] == model}
        return [by_step.get(step, np.nan) for step in steps]

    if problem == "tgv":
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for name in names:
            axes[0, 0].plot(steps, series(name, "rel_l2"), label=name)
            axes[0, 1].plot(steps, series(name, "energy_ratio"), label=name)
            axes[1, 0].plot(steps, series(name, "enstrophy_ratio"), label=name)
            axes[1, 1].plot(steps, series(name, "divergence_l2"), label=name)
        axes[0, 0].set_title("relative L2")
        axes[0, 1].set_title("energy / exact energy")
        axes[1, 0].set_title("enstrophy / exact enstrophy")
        axes[1, 1].set_title("divergence L2")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for name in names:
            axes[0, 0].plot(steps, series(name, "rel_l2"), label=name)
            axes[0, 1].plot(steps, series(name, "entropy_ratio"), label=name)
            axes[1, 0].plot(steps, series(name, "total_variation_ratio"), label=name)
            axes[1, 1].plot(steps, series(name, "overshoot"), label=name)
        axes[0, 0].set_title("relative L2")
        axes[0, 1].set_title("entropy / exact entropy")
        axes[1, 0].set_title("TV / exact TV")
        axes[1, 1].set_title("overshoot beyond exact bounds")

    for ax in axes.flat:
        ax.set_xlabel("autoregressive step")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.savefig(output_dir / "rollout_metrics.png", dpi=180)
    plt.close(fig)


def plot_snapshots(output_dir, snapshots, problem):
    if not snapshots:
        return
    steps = sorted(snapshots.keys())
    models = list(next(iter(snapshots.values())).keys())
    fig, axes = plt.subplots(
        len(models),
        len(steps),
        figsize=(2.4 * len(steps), 2.4 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    values = []
    for step in steps:
        for model in models:
            values.append(snapshots[step][model])
    vmin = min(float(value.min()) for value in values)
    vmax = max(float(value.max()) for value in values)
    cmap = "RdBu_r" if vmin < 0.0 < vmax else "viridis"
    for row, model in enumerate(models):
        for col, step in enumerate(steps):
            axes[row, col].imshow(
                snapshots[step][model],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_title(f"{model}, step {step}")
    filename = "vorticity_snapshots.png" if problem == "tgv" else "scalar_snapshots.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = build_dataset(args)
    models = load_models(args, device)
    global_trajectory_index = dataset.start + args.trajectory_index
    dt = dataset.dt
    dx = (2.0 * np.pi / args.resolution) if args.problem == "tgv" else (
        2.0 * dataset.domain_radius / float(args.resolution - 1)
    )
    dy = dx
    snap_steps = set(snapshot_steps(args))

    initial = dataset.exact_state(global_trajectory_index, 0).to(device)
    states = {
        name: normalize(dataset, initial).unsqueeze(0).to(device)
        for name in models
    }

    rows = []
    snapshots = {}
    blowup = {name: None for name in models}

    for step in range(args.steps + 1):
        truth = dataset.exact_state(global_trajectory_index, step).to(device)
        if args.problem == "tgv":
            truth_q = tgv_quantities(truth, dx, dy)
        else:
            truth_q = burgers_quantities(truth, dx, dy)

        if step in snap_steps:
            snapshots[step] = {}
            if args.problem == "tgv":
                snapshots[step]["truth"] = truth_q["omega"].numpy()
            else:
                snapshots[step]["truth"] = truth[0].detach().cpu().numpy()

        for name, state in states.items():
            pred = denormalize(dataset, state.squeeze(0)).to(device)
            row = {
                "model": name,
                "step": step,
                "rel_l1": relative_l1(pred, truth),
                "rel_l2": relative_l2(pred, truth),
                "rmse": torch.sqrt((pred - truth).pow(2).mean()).item(),
            }

            if args.problem == "tgv":
                pred_q = tgv_quantities(pred, dx, dy)
                row.update(
                    {
                        "energy": pred_q["energy"],
                        "truth_energy": truth_q["energy"],
                        "energy_ratio": pred_q["energy"] / max(truth_q["energy"], 1.0e-12),
                        "enstrophy": pred_q["enstrophy"],
                        "truth_enstrophy": truth_q["enstrophy"],
                        "enstrophy_ratio": pred_q["enstrophy"]
                        / max(truth_q["enstrophy"], 1.0e-12),
                        "divergence_l2": pred_q["divergence_l2"],
                    }
                )
                if step in snap_steps:
                    snapshots[step][name] = pred_q["omega"].numpy()
            else:
                pred_q = burgers_quantities(pred, dx, dy)
                upper = max(truth_q["max"], truth_q["min"])
                lower = min(truth_q["max"], truth_q["min"])
                overshoot = max(0.0, pred_q["max"] - upper) + max(0.0, lower - pred_q["min"])
                row.update(
                    {
                        "entropy": pred_q["entropy"],
                        "truth_entropy": truth_q["entropy"],
                        "entropy_ratio": pred_q["entropy"]
                        / max(truth_q["entropy"], 1.0e-12),
                        "total_variation": pred_q["total_variation"],
                        "truth_total_variation": truth_q["total_variation"],
                        "total_variation_ratio": pred_q["total_variation"]
                        / max(truth_q["total_variation"], 1.0e-12),
                        "overshoot": overshoot,
                    }
                )
                if step in snap_steps:
                    snapshots[step][name] = pred[0].detach().cpu().numpy()

            rows.append(row)

            if blowup[name] is None:
                finite = torch.isfinite(pred).all().item()
                capped = pred.abs().max().item() <= args.max_abs_cap
                if not finite:
                    blowup[name] = {"step": step, "reason": "nonfinite"}
                elif not capped:
                    blowup[name] = {"step": step, "reason": "max_abs_cap"}

        if step == args.steps:
            break

        for name, model in models.items():
            if blowup[name] is None:
                states[name] = model_step(
                    model,
                    states[name],
                    dt=dt,
                    substeps=max(int(args.substeps_per_step), 1),
                )

    write_csv(output_dir / "rollout_metrics.csv", rows)
    plot_curves(output_dir, rows, args.problem)
    plot_snapshots(output_dir, snapshots, args.problem)

    summary = {
        "problem": args.problem,
        "steps": args.steps,
        "trajectory_index": int(global_trajectory_index),
        "dt": dt,
        "viscosity": getattr(dataset, "viscosity", None),
        "dx": dx,
        "dy": dy,
        "substeps_per_step": args.substeps_per_step,
        "blowup": blowup,
        "final": {
            name: next(
                row
                for row in reversed(rows)
                if row["model"] == name
            )
            for name in models
        },
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
