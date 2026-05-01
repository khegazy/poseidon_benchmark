#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DEFAULT_PSCRATCH="/pscratch/sd/j/${USER}"
PSCRATCH_ROOT="${PSCRATCH:-${SCRATCH:-${DEFAULT_PSCRATCH}}}"
RUN_ROOT="${RUN_ROOT:-${PSCRATCH_ROOT}/poseidon_entropy_analytic}"
LINK_ROOT="${LINK_ROOT:-${REPO_ROOT}/poseidon_entropy_analytic}"

PYTHON="${PYTHON:-python}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-camlab-ethz/Poseidon-T}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${RUN_ROOT}/pretrained/Poseidon-T}"
WANDB_PROJECT="${WANDB_PROJECT:-poseidon_entropy_analytic}"
DEVICE="${DEVICE:-cuda}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
export HF_HOME="${HF_HOME:-${RUN_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/xdg_cache}"
export TMPDIR="${TMPDIR:-${RUN_ROOT}/tmp}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p \
    "${RUN_ROOT}/checkpoints" \
    "${RUN_ROOT}/results" \
    "${RUN_ROOT}/pretrained" \
    "${WANDB_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}"

if [ ! -e "${LINK_ROOT}" ]; then
    ln -s "${RUN_ROOT}" "${LINK_ROOT}"
fi

checkpoint_dir() {
    printf "%s/checkpoints/%s/%s" "${RUN_ROOT}" "${WANDB_PROJECT}" "$1"
}

download_file() {
    local filename="$1"
    local dest="${PRETRAINED_DIR}/${filename}"
    if [ -s "${dest}" ]; then
        printf "Using existing %s\n" "${dest}"
        return
    fi
    mkdir -p "${PRETRAINED_DIR}"
    curl \
        --location \
        --fail \
        --retry 5 \
        --retry-delay 5 \
        --output "${dest}" \
        "https://huggingface.co/${PRETRAINED_MODEL}/resolve/main/${filename}"
}

download_pretrained() {
    download_file "config.json"
    download_file "pytorch_model.bin"
    printf "Pretrained weights are in %s\n" "${PRETRAINED_DIR}"
}

train_one() {
    local config="$1"
    local run_name="$2"
    download_pretrained
    "${PYTHON}" -m scOT.train \
        --config "${REPO_ROOT}/${config}" \
        --data_path "${REPO_ROOT}" \
        --checkpoint_path "${RUN_ROOT}/checkpoints" \
        --wandb_project_name "${WANDB_PROJECT}" \
        --wandb_run_name "${run_name}" \
        --finetune_from "${PRETRAINED_DIR}" \
        --replace_embedding_recovery
}

rollout_tgv() {
    "${PYTHON}" "${REPO_ROOT}/scripts/analytic_entropy_rollout.py" \
        --problem tgv \
        --baseline-checkpoint "$(checkpoint_dir tgv_baseline_t)" \
        --entropy-checkpoint "$(checkpoint_dir tgv_energy_t)" \
        --output-dir "${RUN_ROOT}/results/tgv_rollout" \
        --steps "${STEPS:-5000}" \
        --device "${DEVICE}"
}

rollout_tgv_re50k() {
    "${PYTHON}" "${REPO_ROOT}/scripts/analytic_entropy_rollout.py" \
        --problem tgv \
        --baseline-checkpoint "$(checkpoint_dir tgv_re50k_baseline_t)" \
        --entropy-checkpoint "$(checkpoint_dir tgv_re50k_energy_t)" \
        --output-dir "${RUN_ROOT}/results/tgv_re50k_rollout" \
        --steps "${STEPS:-5000}" \
        --viscosity 0.00002 \
        --device "${DEVICE}"
}

rollout_burgers() {
    "${PYTHON}" "${REPO_ROOT}/scripts/analytic_entropy_rollout.py" \
        --problem burgers \
        --baseline-checkpoint "$(checkpoint_dir burgers_baseline_t)" \
        --entropy-checkpoint "$(checkpoint_dir burgers_entropy_t)" \
        --output-dir "${RUN_ROOT}/results/burgers_rollout" \
        --steps "${STEPS:-1000}" \
        --device "${DEVICE}"
}

usage() {
    cat <<USAGE
Usage: $(basename "$0") <command>

Commands:
  download-pretrained
  train-tgv-baseline
  train-tgv-energy
  eval-tgv
  train-tgv-re50k-baseline
  train-tgv-re50k-energy
  eval-tgv-re50k
  train-burgers-baseline
  train-burgers-entropy
  eval-burgers

Persistent paths:
  RUN_ROOT=${RUN_ROOT}
  repo symlink=${LINK_ROOT}
  HF_HOME=${HF_HOME}
  checkpoints=${RUN_ROOT}/checkpoints
  results=${RUN_ROOT}/results

Set PYTHON=/path/to/python if the active python is not the Poseidon environment.
USAGE
}

case "${1:-}" in
    download-pretrained)
        download_pretrained
        ;;
    train-tgv-baseline)
        train_one "configs/tgv_poseidon_t_baseline.yaml" "tgv_baseline_t"
        ;;
    train-tgv-energy)
        train_one "configs/tgv_poseidon_t_energy.yaml" "tgv_energy_t"
        ;;
    eval-tgv)
        rollout_tgv
        ;;
    train-tgv-re50k-baseline)
        train_one "configs/tgv_re50k_poseidon_t_baseline.yaml" "tgv_re50k_baseline_t"
        ;;
    train-tgv-re50k-energy)
        train_one "configs/tgv_re50k_poseidon_t_energy.yaml" "tgv_re50k_energy_t"
        ;;
    eval-tgv-re50k)
        rollout_tgv_re50k
        ;;
    train-burgers-baseline)
        train_one "configs/burgers2d_poseidon_t_baseline.yaml" "burgers_baseline_t"
        ;;
    train-burgers-entropy)
        train_one "configs/burgers2d_poseidon_t_entropy.yaml" "burgers_entropy_t"
        ;;
    eval-burgers)
        rollout_burgers
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
