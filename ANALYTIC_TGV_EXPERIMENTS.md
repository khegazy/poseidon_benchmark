# Analytic Taylor-Green Vortex Poseidon Experiments

This note documents the analytic Taylor-Green vortex (TGV) experiments for the
`entropy-flux-regularizer` branch. The goal is to compare a data-only
Poseidon/scOT fine-tune against a macroscopic energy/enstrophy regularized
fine-tune in long autoregressive rollouts.

The key claim is empirical:

```text
Macroscopic energy/enstrophy regularization reduces physical budget violations
and improves observed long-rollout stability.
```

It is not a theorem about unconditional stability of the learned Transformer.
The loss is soft, not an architecture-level or scheme-level stability proof.

## Implemented Files

Core implementation:

```text
scOT/problems/fluids/incompressible.py       TaylorGreenVortex
scripts/analytic_entropy_rollout.py          long rollout evaluator
run_scripts/analytic_entropy_experiments.sh  pscratch-backed run helper
run_scripts/slurm_analytic_entropy.sh        Slurm batch wrapper
run_scripts/submit_analytic_tgv_re50k.sh     Slurm pipeline submitter
```

TGV configs:

```text
configs/tgv_poseidon_t_baseline.yaml
configs/tgv_poseidon_t_energy.yaml
configs/tgv_re50k_poseidon_t_baseline.yaml
configs/tgv_re50k_poseidon_t_energy.yaml
```

The Re50K configs use:

```text
Re = 50000
nu = 2e-5
dt = 0.05
resolution = 128 x 128
model = Poseidon-T
```

The Reynolds convention in these configs is nondimensional:

```text
Re = U L / nu, with U = 1 and L = 1
```

If a different convention is needed, for example `L = 2*pi`, change both the
dataset viscosity and the regularizer viscosity consistently.

## Scratch Layout

Persistent outputs are kept off home and under pscratch:

```text
/pscratch/sd/j/jalb/poseidon_entropy_analytic
```

The repo has a convenience symlink:

```text
poseidon_entropy_analytic -> /pscratch/sd/j/jalb/poseidon_entropy_analytic
```

Expected subdirectories:

```text
pretrained/Poseidon-T/       local Poseidon-T weights
checkpoints/                 Hugging Face Trainer checkpoints
results/                     rollout CSV, JSON, and plots
slurm_logs/                  Slurm stdout/stderr
hf_home/                     Hugging Face cache
wandb/                       W&B files, with WANDB_MODE=disabled by default
tmp/                         temporary files
```

The local pretrained model directory is:

```text
/pscratch/sd/j/jalb/poseidon_entropy_analytic/pretrained/Poseidon-T
```

It should contain:

```text
config.json
pytorch_model.bin
```

Fine-tuning uses the local directory through:

```text
--finetune_from /pscratch/sd/j/jalb/poseidon_entropy_analytic/pretrained/Poseidon-T
```

Because TGV predicts `[u, v]`, while pretrained Poseidon uses the original
pretraining channels, runs use:

```text
--replace_embedding_recovery
```

## Analytic TGV Reference

The 2D Taylor-Green vortex is generated on the periodic square domain:

```text
[0, 2*pi]^2
```

The analytic velocity field is:

```text
u(x, y, t) =  A sin(x + phi_x) cos(y + phi_y) exp(-2 nu t)
v(x, y, t) = -A cos(x + phi_x) sin(y + phi_y) exp(-2 nu t)
```

The vorticity is:

```text
omega(x, y, t) = d_x v - d_y u
                = 2 A sin(x + phi_x) sin(y + phi_y) exp(-2 nu t)
```

For the ideal analytic solution:

```text
E(t) = E(0) exp(-4 nu t)
Z(t) = Z(0) exp(-4 nu t)
```

where `E` is kinetic energy and `Z` is enstrophy.

## Losses And Norms

Let `dx = dy = 2*pi/N` and `cell_area = dx * dy`.

### Field Error

For decoded prediction `U_pred` and analytic truth `U_true`:

```text
relative L1 = sum |U_pred - U_true| / max(sum |U_true|, eps)
relative L2 = ||U_pred - U_true||_2 / max(||U_true||_2, eps)
RMSE        = sqrt(mean((U_pred - U_true)^2))
```

These are reported at every autoregressive rollout step.

### Kinetic Energy

For velocity `u = (u, v)`:

```text
E^n = 0.5 * sum (u_n^2 + v_n^2) * cell_area
```

The resolved viscous dissipation is:

```text
D_E^n = sum (|grad u_n|^2 + |grad v_n|^2) * cell_area
```

The energy budget residual is:

```text
B_E^n = E^{n+1} - E^n + dt * nu * D_E^n
```

For analytic TGV, the regularized config uses the equality form:

```text
L_E = mean((B_E^n)^2)
```

For noisy or underresolved data, an inequality variant is also supported:

```text
L_E = mean(ReLU(B_E^n)^2)
```

### Enstrophy

The vorticity is:

```text
omega = d_x v - d_y u
```

The enstrophy is:

```text
Z^n = 0.5 * sum omega_n^2 * cell_area
```

The enstrophy dissipation is:

```text
D_Z^n = sum |grad omega_n|^2 * cell_area
```

The enstrophy budget residual is:

```text
B_Z^n = Z^{n+1} - Z^n + dt * nu * D_Z^n
```

The analytic TGV regularized config uses:

```text
L_Z = mean((B_Z^n)^2)
```

### Divergence

The incompressibility residual is:

```text
div u = d_x u + d_y v
```

The divergence loss is:

```text
L_div = mean((div u)^2)
```

### Total Training Objective

Baseline:

```text
L = L_data
```

Energy/enstrophy regularized:

```text
L = L_data + lambda_E L_E + lambda_Z L_Z + lambda_div L_div
```

The Re50K energy config currently uses:

```text
lambda_E   = 0.01
lambda_Z   = 0.001
lambda_div = 0.001
nu         = 2e-5
```

## How To Run Interactively

From the repo root:

```bash
module load conda
conda activate kinet
```

Download or verify the local Poseidon-T weights:

```bash
run_scripts/analytic_entropy_experiments.sh download-pretrained
```

Run Re50K TGV baseline:

```bash
run_scripts/analytic_entropy_experiments.sh train-tgv-re50k-baseline
```

Run Re50K TGV energy/enstrophy regularized:

```bash
run_scripts/analytic_entropy_experiments.sh train-tgv-re50k-energy
```

Evaluate after both checkpoints exist:

```bash
run_scripts/analytic_entropy_experiments.sh eval-tgv-re50k
```

Override rollout length:

```bash
STEPS=10000 run_scripts/analytic_entropy_experiments.sh eval-tgv-re50k
```

Override the run root:

```bash
RUN_ROOT=/pscratch/sd/j/jalb/my_other_run \
  run_scripts/analytic_entropy_experiments.sh train-tgv-re50k-baseline
```

## How To Submit With Slurm

The Slurm wrapper activates the requested environment:

```bash
module load conda
conda activate kinet
```

Submit the Re50K pipeline:

```bash
run_scripts/submit_analytic_tgv_re50k.sh
```

This submits:

```text
tgv-r50k-base      6 hour walltime
tgv-r50k-energy    6 hour walltime
tgv-r50k-eval      2 hour walltime, after both training jobs succeed
```

Override walltimes:

```bash
TRAIN_TIME=12:00:00 EVAL_TIME=04:00:00 \
  run_scripts/submit_analytic_tgv_re50k.sh
```

Check jobs:

```bash
squeue -u $USER
```

Check logs:

```bash
ls -lh /pscratch/sd/j/jalb/poseidon_entropy_analytic/slurm_logs
```

## Output Files

The long rollout evaluator writes:

```text
rollout_metrics.csv
summary.json
rollout_metrics.png
vorticity_snapshots.png
```

For Re50K TGV:

```text
/pscratch/sd/j/jalb/poseidon_entropy_analytic/results/tgv_re50k_rollout
```

The CSV includes stepwise field error, energy, enstrophy, energy ratio,
enstrophy ratio, and divergence.

## What To Look For

The main comparison is not only field error. Check:

```text
relative L1/L2 error over rollout
energy / exact energy
enstrophy / exact enstrophy
divergence L2
vorticity snapshots
blowup or nonfinite step
```

Expected qualitative behavior:

```text
baseline:
  may fit one-step data but can drift in energy/enstrophy over long rollouts

regularized:
  should reduce budget residuals and keep energy/enstrophy closer to the exact decay
```

## Notes On Re50K

At `Re = 50000`, `nu = 2e-5`, so analytic energy decay over moderate horizons is
slow. For `dt = 0.05` and `N = 5000` rollout steps:

```text
t_final = 250
E(t_final) / E(0) = exp(-4 * 2e-5 * 250) approx 0.9802
```

This means Re50K is mainly a long-rollout drift and stability test, not a fast
decay-rate test. Small unphysical energy growth can be visible because the true
decay is weak.

## Related Burgers Commands

The same runner also supports the scalar Burgers entropy-flux ablation:

```bash
run_scripts/analytic_entropy_experiments.sh train-burgers-baseline
run_scripts/analytic_entropy_experiments.sh train-burgers-entropy
run_scripts/analytic_entropy_experiments.sh eval-burgers
```

Burgers uses the convex entropy pair:

```text
eta(u) = 0.5 u^2
q(u)   = u^3 / 3
```

and penalizes positive violations of:

```text
d_t eta + div q <= 0
```

## References

This section is split by how each reference supports the experiment. The TGV
regularizer is motivated by the continuous incompressible energy/enstrophy
method. The Burgers ablation is the direct convex entropy-flux test.

### Poseidon/scOT Foundation Model

- Maximilian Herde, Bogdan Raonic, Tobias Rohner, Roger Kaeppeli, Roberto
  Molinaro, Emmanuel de Bezenac, Siddhartha Mishra. "Poseidon: Efficient
  Foundation Models for PDEs." CoRR abs/2405.19101, 2024.
  DOI: `10.48550/arXiv.2405.19101`.
  URL: https://arxiv.org/abs/2405.19101.
  Role in this experiment: pretrained scOT/Poseidon operator architecture and
  fine-tuning workflow.

- Poseidon pretrained model collection. Hugging Face, camlab-ethz.
  DOI: none.
  URL:
  https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a.
  Role in this experiment: source for `camlab-ethz/Poseidon-T`, downloaded to
  pscratch and loaded through `ScOT.from_pretrained`.

### Taylor-Green Vortex Reference

- G. I. Taylor and A. E. Green. "Mechanism of the Production of Small Eddies
  from Large Ones." Proceedings of the Royal Society of London. Series A,
  Mathematical and Physical Sciences, 158(895), 499-521, 1937.
  DOI: `10.1098/rspa.1937.0036`.
  URL: https://doi.org/10.1098/rspa.1937.0036.
  Role in this experiment: original Taylor-Green vortex reference. The present
  code uses the 2D decaying incompressible analytic form as the exact rollout
  target.

### Incompressible Energy/Enstrophy Method

- Roger Temam. "Navier-Stokes Equations: Theory and Numerical Analysis." AMS
  Chelsea Publishing, volume 343, 1984 reprint.
  DOI: `10.1090/chel/343`.
  URL: https://doi.org/10.1090/chel/343.
  Role in this experiment: standard reference for the incompressible
  Navier-Stokes energy method and viscous dissipation estimates.

- Charles R. Doering and J. D. Gibbon. "Applied Analysis of the Navier-Stokes
  Equations." Cambridge University Press, 1995.
  DOI: `10.1017/CBO9780511608803`.
  URL: https://doi.org/10.1017/CBO9780511608803.
  Role in this experiment: reference for energy/enstrophy estimates,
  dimensionless parameters, and stability analysis for Navier-Stokes flows.

- Juan Manzanero, Gonzalo Rubio, David A. Kopriva, Esteban Ferrer, Eusebio
  Valero. "An entropy-stable discontinuous Galerkin approximation for the
  incompressible Navier-Stokes equations with variable density and artificial
  compressibility." Journal of Computational Physics, 408, article 109241,
  2020.
  DOI: `10.1016/j.jcp.2020.109241`.
  URL: https://doi.org/10.1016/j.jcp.2020.109241.
  Role in this experiment: shows how incompressible Navier-Stokes stability can
  be cast as an entropy/energy bound in a numerical method. Our regularizer is
  a soft neural analogue, not a proof of discrete entropy stability.

### Entropy Stability And Entropy-Flux Inequalities

- Eitan Tadmor. "Entropy stability theory for difference approximations of
  nonlinear conservation laws and related time-dependent problems." Acta
  Numerica, 12, 451-512, 2003.
  DOI: `10.1017/S0962492902000156`.
  URL: https://doi.org/10.1017/S0962492902000156.
  Role in this experiment: core entropy-stability framework for conservation
  laws and entropy inequalities. This is the mathematical reference behind the
  Burgers entropy-flux loss.

- Ulrik S. Fjordholm, Siddhartha Mishra, Eitan Tadmor. "Arbitrarily High-order
  Accurate Entropy Stable Essentially Nonoscillatory Schemes for Systems of
  Conservation Laws." SIAM Journal on Numerical Analysis, 50(2), 544-573,
  2012.
  DOI: `10.1137/110836961`.
  URL: https://doi.org/10.1137/110836961.
  Role in this experiment: high-order entropy-stable finite-volume/scheme
  reference showing how entropy-conservative fluxes plus dissipation enforce
  entropy stability.

- Mark H. Carpenter, Travis C. Fisher, Eric J. Nielsen, Steven H. Frankel.
  "Entropy Stable Spectral Collocation Schemes for the Navier-Stokes Equations:
  Discontinuous Interfaces." SIAM Journal on Scientific Computing, 36(5),
  B835-B867, 2014.
  DOI: `10.1137/130932193`.
  URL: https://doi.org/10.1137/130932193.
  Role in this experiment: entropy-stable high-order Navier-Stokes
  discretization reference. It motivates using PDE-level stability budgets
  rather than only pointwise data error.

- Jesse Chan, Yimin Lin, Tim Warburton. "Entropy stable modal discontinuous
  Galerkin schemes and wall boundary conditions for the compressible
  Navier-Stokes equations." Journal of Computational Physics, 448, article
  110723, 2022.
  DOI: `10.1016/j.jcp.2021.110723`.
  URL: https://doi.org/10.1016/j.jcp.2021.110723.
  Role in this experiment: modern entropy-stable DG reference for viscous
  compressible Navier-Stokes and boundary treatments.

- Tim De Ryck, Siddhartha Mishra, Roberto Molinaro. "wPINNs: Weak Physics
  Informed Neural Networks for Approximating Entropy Solutions of Hyperbolic
  Conservation Laws." SIAM Journal on Numerical Analysis, 62(2), 811-841,
  2024.
  DOI: `10.1137/22M1522504`.
  URL: https://doi.org/10.1137/22M1522504.
  Role in this experiment: neural PDE reference where entropy conditions are
  used to select physically meaningful weak solutions.

### Entropic And Turbulence-Motivated Kinetic References

- Fabian Bosch, Shyam S. Chikatamarla, Ilya V. Karlin. "Entropic
  multirelaxation lattice Boltzmann models for turbulent flows." Physical
  Review E, 92(4), article 043309, 2015.
  DOI: `10.1103/PhysRevE.92.043309`.
  URL: https://doi.org/10.1103/PhysRevE.92.043309.
  Role in this experiment: turbulence-motivated entropy stabilization reference
  from lattice Boltzmann modeling. Our method is macroscopic and does not use
  kinetic populations.

- Benedikt Dorschner, Fabian Bosch, Shyam S. Chikatamarla, Konstantinos
  Boulouchos, Ilya V. Karlin. "Entropic multi-relaxation time lattice Boltzmann
  model for complex flows." Journal of Fluid Mechanics, 801, 623-651, 2016.
  DOI: `10.1017/jfm.2016.448`.
  URL: https://doi.org/10.1017/jfm.2016.448.
  Role in this experiment: entropic turbulence modeling reference with complex
  flow validation. It helps position the stability idea, while the implemented
  regularizer remains a decoded-field macroscopic loss.
