#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

RUN_ROOT="${RUN_ROOT:-/pscratch/sd/j/${USER}/poseidon_entropy_analytic}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RUN_ROOT}/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

TRAIN_TIME="${TRAIN_TIME:-06:00:00}"
EVAL_TIME="${EVAL_TIME:-02:00:00}"

baseline_job=$(
    sbatch --parsable \
        --job-name=tgv-r50k-base \
        --time="${TRAIN_TIME}" \
        --output="${SLURM_LOG_DIR}/%x-%j.out" \
        --error="${SLURM_LOG_DIR}/%x-%j.err" \
        --export=ALL,ANALYTIC_COMMAND=train-tgv-re50k-baseline \
        run_scripts/slurm_analytic_entropy.sh
)

energy_job=$(
    sbatch --parsable \
        --job-name=tgv-r50k-energy \
        --time="${TRAIN_TIME}" \
        --output="${SLURM_LOG_DIR}/%x-%j.out" \
        --error="${SLURM_LOG_DIR}/%x-%j.err" \
        --export=ALL,ANALYTIC_COMMAND=train-tgv-re50k-energy \
        run_scripts/slurm_analytic_entropy.sh
)

eval_job=$(
    sbatch --parsable \
        --job-name=tgv-r50k-eval \
        --time="${EVAL_TIME}" \
        --dependency="afterok:${baseline_job}:${energy_job}" \
        --output="${SLURM_LOG_DIR}/%x-%j.out" \
        --error="${SLURM_LOG_DIR}/%x-%j.err" \
        --export=ALL,ANALYTIC_COMMAND=eval-tgv-re50k \
        run_scripts/slurm_analytic_entropy.sh
)

cat <<JOBS
Submitted TGV Re50K Poseidon benchmark pipeline:
  baseline: ${baseline_job}
  energy:   ${energy_job}
  eval:     ${eval_job} depends on afterok:${baseline_job}:${energy_job}

Logs:
  ${SLURM_LOG_DIR}

Scratch outputs:
  ${RUN_ROOT}
JOBS
