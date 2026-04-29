"""
This script trains a scOT or pretrains Poseidon on a PDE dataset.
Can be also used for finetuning Poseidon.
Can be used in a single config or sweep setup.
"""

import argparse
import torch
import wandb
import numpy as np
import random
import json
import psutil
import os
import inspect

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
import matplotlib.pyplot as plt
import transformers
from accelerate.utils import broadcast_object_list
from scOT.trainer import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, TrainerCallback
from scOT.model import ScOT, ScOTConfig
from mpl_toolkits.axes_grid1 import ImageGrid
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.utils import get_num_parameters, read_cli, get_num_parameters_no_embed
from scOT.metrics import relative_lp_error

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}


def create_predictions_plot(predictions, labels, wandb_prefix):
    assert predictions.shape[0] >= 4

    indices = random.sample(range(predictions.shape[0]), 4)

    predictions = predictions[indices]
    labels = labels[indices]

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
    )

    vmax, vmin = max(predictions.max(), labels.max()), min(
        predictions.min(), labels.min()
    )

    for _i, ax in enumerate(grid):
        i = _i // 4
        j = _i % 4

        if i % 2 == 0:
            ax.imshow(
                predictions[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.imshow(
                labels[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
    plt.close()


class ARStepsCurriculumCallback(TrainerCallback):
    def __init__(self, trainer, target_ar_steps, config=None):
        self.trainer = trainer
        self.target_ar_steps = self._normalize_target(target_ar_steps)
        self.config = dict(config or {})
        self._last_ar_steps = None

    def _normalize_target(self, target_ar_steps):
        if isinstance(target_ar_steps, int):
            return max(int(target_ar_steps), 1)
        if isinstance(target_ar_steps, (list, tuple)):
            target = list(target_ar_steps)
            if len(target) == 0:
                raise ValueError("train_ar_steps curriculum requires a non-empty target.")
            return target
        raise ValueError("train_ar_steps must be an int or list to use curriculum.")

    def _target_length(self):
        if isinstance(self.target_ar_steps, int):
            return self.target_ar_steps
        return len(self.target_ar_steps)

    def _start_length(self):
        start = int(self.config.get("start", self.config.get("start_steps", 1)))
        return max(1, min(start, self._target_length()))

    def _warmup_steps(self, state):
        steps = self.config.get("steps", self.config.get("warmup_steps", None))
        if steps is not None:
            return max(int(steps), 1)

        ratio = self.config.get("warmup_ratio", self.config.get("ratio", 0.5))
        if state.max_steps is None or state.max_steps <= 0:
            return 1
        return max(int(float(ratio) * state.max_steps), 1)

    def _scheduled_ar_steps(self, state):
        target_length = self._target_length()
        start_length = self._start_length()
        if target_length <= start_length:
            current_length = target_length
        else:
            progress = min(
                max(float(state.global_step), 0.0) / float(self._warmup_steps(state)),
                1.0,
            )
            current_length = start_length + int(
                progress * float(target_length - start_length)
            )
            current_length = max(start_length, min(current_length, target_length))

        if isinstance(self.target_ar_steps, int):
            return current_length
        return self.target_ar_steps[:current_length]

    def _store_last_ar_steps(self, ar_steps):
        if isinstance(ar_steps, list):
            self._last_ar_steps = list(ar_steps)
        else:
            self._last_ar_steps = ar_steps

    def _set_ar_steps(self, ar_steps, state):
        if ar_steps == self._last_ar_steps:
            return
        self.trainer.set_ar_steps(ar_steps)
        self._store_last_ar_steps(ar_steps)
        if getattr(state, "is_world_process_zero", False):
            print(f"[AR curriculum] train_ar_steps -> {ar_steps}")

    def on_train_begin(self, args, state, control, **kwargs):
        self._set_ar_steps(self._scheduled_ar_steps(state), state)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._set_ar_steps(self._scheduled_ar_steps(state), state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._set_ar_steps(self.target_ar_steps, state)
        return control


def setup(params, model_map=True):
    config = None
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    CPU_CORES = len(psutil.Process().cpu_affinity())
    CPU_CORES = min(CPU_CORES, 16)
    print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
    if params.json_config:
        config = json.loads(params.config)
    else:
        config = params.config

    if RANK == 0 or RANK == -1:
        run = wandb.init(
            project=params.wandb_project_name, name=params.wandb_run_name, config=config
        )
        config = wandb.config
    else:

        def clean_yaml(config):
            d = {}
            for key, inner_dict in config.items():
                d[key] = inner_dict["value"]
            return d

        if not params.json_config:
            with open(params.config, "r") as s:
                config = yaml.safe_load(s)
            config = clean_yaml(config)
        run = None

    ckpt_dir = "./"
    if RANK == 0 or RANK == -1:
        run_project = run.project
        run_name = run.name
        if os.environ.get("WANDB_MODE", "").lower() == "disabled":
            run_project = params.wandb_project_name
            run_name = params.wandb_run_name or run.name
        if run.sweep_id is not None:
            ckpt_dir = (
                params.checkpoint_path
                + "/"
                + run_project
                + "/"
                + run.sweep_id
                + "/"
                + run_name
            )
        else:
            ckpt_dir = params.checkpoint_path + "/" + run_project + "/" + run_name
    if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ls = broadcast_object_list([ckpt_dir], from_process=0)
    ckpt_dir = ls[0]

    if model_map and (
        type(config["model_name"]) == str and config["model_name"] in MODEL_MAP.keys()
    ):
        config = {**config, **MODEL_MAP[config["model_name"]]}
        if RANK == 0 or RANK == -1:
            wandb.config.update(MODEL_MAP[config["model_name"]], allow_val_change=True)

    return run, config, ckpt_dir, RANK, CPU_CORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scOT or pretrain Poseidon.")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Set this to a str pointing to a HF Hub model checkpoint or a directory with a scOT checkpoint if you want to finetune.",
    )
    parser.add_argument(
        "--replace_embedding_recovery",
        action="store_true",
        help="Set this if you have to replace the embeddings and recovery layers because you are not just using the density, velocity and pressure channels. Only relevant for finetuning.",
    )
    params = read_cli(parser).parse_args()
    run, config, ckpt_dir, RANK, CPU_CORES = setup(params)
    seed = int(config.get("seed", SEED))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dataloader_num_workers = int(config.get("dataloader_num_workers", CPU_CORES))
    dataloader_pin_memory = bool(config.get("dataloader_pin_memory", True))

    train_eval_set_kwargs = (
        {"just_velocities": True}
        if ("incompressible" in config["dataset"]) and params.just_velocities
        else {}
    )
    if params.move_data is not None:
        train_eval_set_kwargs["move_to_local_scratch"] = params.move_data
    if isinstance(config.get("dataset_kwargs", None), dict):
        train_eval_set_kwargs.update(config["dataset_kwargs"])
    if params.max_num_train_time_steps is not None:
        train_eval_set_kwargs["max_num_time_steps"] = params.max_num_train_time_steps
    if params.train_time_step_size is not None:
        train_eval_set_kwargs["time_step_size"] = params.train_time_step_size
    if params.train_small_time_transition:
        train_eval_set_kwargs["allowed_time_transitions"] = [1]
    train_dataset = get_dataset(
        dataset=config["dataset"],
        which="train",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    eval_dataset = get_dataset(
        dataset=config["dataset"],
        which="val",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )

    config["effective_train_set_size"] = len(train_dataset)
    time_involved = isinstance(train_dataset, BaseTimeDataset) or (
        isinstance(train_dataset, torch.utils.data.ConcatDataset)
        and isinstance(train_dataset.datasets[0], BaseTimeDataset)
    )

    if not isinstance(train_dataset, torch.utils.data.ConcatDataset):
        resolution = train_dataset.resolution
        input_dim = train_dataset.input_dim
        output_dim = train_dataset.output_dim
        channel_slice_list = train_dataset.channel_slice_list
        printable_channel_description = train_dataset.printable_channel_description
    else:
        resolution = train_dataset.datasets[0].resolution
        input_dim = train_dataset.datasets[0].input_dim
        output_dim = train_dataset.datasets[0].output_dim
        channel_slice_list = train_dataset.datasets[0].channel_slice_list
        printable_channel_description = train_dataset.datasets[
            0
        ].printable_channel_description

    model_config = (
        ScOTConfig(
            image_size=resolution,
            patch_size=config["patch_size"],
            num_channels=input_dim,
            num_out_channels=output_dim,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            skip_connections=config["skip_connections"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=channel_slice_list,
            residual_model="convnext",
            use_conditioning=time_involved,
            learn_residual=False,
        )
        if params.finetune_from is None or params.replace_embedding_recovery
        else None
    )

    training_kwargs = dict(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,  #! OVERWRITE THIS DIRECTORY IN CASE, also for resuming training
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_accumulation_steps=16,
        max_grad_norm=config["max_grad_norm"],
        num_train_epochs=config["num_epochs"],
        optim="adamw_torch",
        learning_rate=config["lr"],
        learning_rate_embedding_recovery=(
            None
            if (params.finetune_from is None or "lr_embedding_recovery" not in config)
            else config["lr_embedding_recovery"]
        ),
        learning_rate_time_embedding=(
            None
            if (params.finetune_from is None or "lr_time_embedding" not in config)
            else config["lr_time_embedding"]
        ),
        rollout_grad_steps=int(config.get("rollout_grad_steps", 1)),
        weight_decay=config["weight_decay"],
        adam_beta1=0.9,  # default
        adam_beta2=0.999,  # default
        adam_epsilon=1e-8,  # default
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        log_level="passive",
        logging_strategy="steps",
        logging_steps=5,
        logging_nan_inf_filter=False,
        save_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        fp16=False,
        dataloader_num_workers=dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_pin_memory=dataloader_pin_memory,
        gradient_checkpointing=False,
        auto_find_batch_size=False,
        full_determinism=False,
        torch_compile=False,
        report_to="wandb",
        run_name=params.wandb_run_name,
    )
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    if "data_seed" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["data_seed"] = seed
    train_config = TrainingArguments(**training_kwargs)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=0.0,  # set no threshold for now
    )

    if params.finetune_from is not None:
        model = ScOT.from_pretrained(
            params.finetune_from, config=model_config, ignore_mismatched_sizes=True
        )
    else:
        model = ScOT(model_config)
    num_params = get_num_parameters(model)
    config["num_params"] = num_params
    num_params_no_embed = get_num_parameters_no_embed(model)
    config["num_params_wout_embed"] = num_params_no_embed
    if RANK == 0 or RANK == -1:
        print(f"Model size: {num_params}")
        print(f"Model size without embeddings: {num_params_no_embed}")

    def compute_metrics(eval_preds):
        channel_list = channel_slice_list

        def get_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_relative_l1_error": median_error,
                "mean_relative_l1_error": mean_error,
                "std_relative_l1_error": std_error,
                "min_relative_l1_error": min_error,
                "max_relative_l1_error": max_error,
            }

        error_statistics = [
            get_statistics(
                relative_lp_error(
                    eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                    p=1,
                    return_percent=True,
                )
            )
            for i in range(len(channel_list) - 1)
        ]

        if output_dim == 1:
            error_statistics = error_statistics[0]
            return error_statistics
        else:
            mean_over_means = np.mean(
                np.array(
                    [stats["mean_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            mean_over_medians = np.mean(
                np.array(
                    [stats["median_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            error_statistics_ = {
                "mean_relative_l1_error": mean_over_means,
                "mean_over_median_relative_l1_error": mean_over_medians,
            }
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[printable_channel_description[i] + "/" + key] = (
                        value
                    )
            return error_statistics_

    regularization_config = None
    if "physics_regularization" in config:
        regularization_config = config["physics_regularization"]
    elif "entropy_regularization" in config:
        regularization_config = config["entropy_regularization"]

    callbacks = [early_stopping]

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        physics_regularization_config=regularization_config,
    )
    train_ar_steps = config.get("train_ar_steps", None)
    if train_ar_steps not in [None, 0, 1]:
        trainer.set_ar_steps(train_ar_steps)

        curriculum_config = config.get("train_ar_steps_curriculum", None)
        target_length = (
            train_ar_steps
            if isinstance(train_ar_steps, int)
            else len(list(train_ar_steps))
        )
        if (
            curriculum_config is not None
            and bool(curriculum_config.get("enabled", True))
            and target_length > 1
        ):
            trainer.add_callback(
                ARStepsCurriculumCallback(
                    trainer=trainer,
                    target_ar_steps=train_ar_steps,
                    config=curriculum_config,
                )
            )

    trainer.train(resume_from_checkpoint=params.resume_training)
    trainer.save_model(train_config.output_dir)
    
    if (RANK == 0 or RANK == -1) and params.push_to_hf_hub is not None:
        model.push_to_hub(params.push_to_hf_hub)

    do_test = (
        True
        if params.max_num_train_time_steps is None
        and params.train_time_step_size is None
        and not params.train_small_time_transition
        and not ".time" in config["dataset"]
        else False
    )
    if do_test:
        print("Testing...")
        test_set_kwargs = (
            {"just_velocities": True}
            if ("incompressible" in config["dataset"]) and params.just_velocities
            else {}
        )
        out_test_set_kwargs = (
            {"just_velocities": True}
            if ("incompressible" in config["dataset"]) and params.just_velocities
            else {}
        )
        if params.move_data is not None:
            test_set_kwargs["move_to_local_scratch"] = params.move_data
            out_test_set_kwargs["move_to_local_scratch"] = params.move_data
        if time_involved:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 14,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 20,
                "allowed_time_transitions": [1],
            }
        if "RayleighTaylor" in config["dataset"]:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 7,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 10,
                "allowed_time_transitions": [1],
            }

        test_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=config["num_trajectories"],
            data_path=params.data_path,
            **test_set_kwargs,
        )
        try:
            out_dist_test_dataset = get_dataset(
                dataset=config["dataset"] + ".out",
                which="test",
                num_trajectories=config["num_trajectories"],
                data_path=params.data_path,
                **out_test_set_kwargs,
            )
        except:
            out_dist_test_dataset = None
        predictions = trainer.predict(test_dataset, metric_key_prefix="")
        if RANK == 0 or RANK == -1:
            metrics = {}
            for key, value in predictions.metrics.items():
                metrics["test/" + key[1:]] = value
            wandb.log(metrics)
            create_predictions_plot(
                predictions.predictions,
                predictions.label_ids,
                wandb_prefix="test",
            )

        # evaluate on out-of-distribution test set
        if out_dist_test_dataset is not None:
            predictions = trainer.predict(out_dist_test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test_out_dist/" + key[1:]] = value
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test_out_dist",
                )

        if time_involved and (test_set_kwargs["time_step_size"] // 2 > 0):
            trainer.set_ar_steps(test_set_kwargs["time_step_size"] // 2)
            predictions = trainer.predict(test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test/ar/" + key[1:]] = value
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test/ar",
                )

            # evaluate on out-of-distribution test set
            if out_dist_test_dataset is not None:
                trainer.set_ar_steps(out_test_set_kwargs["time_step_size"] // 2)
                predictions = trainer.predict(
                    out_dist_test_dataset, metric_key_prefix=""
                )
                if RANK == 0 or RANK == -1:
                    metrics = {}
                    for key, value in predictions.metrics.items():
                        metrics["test_out_dist/ar/" + key[1:]] = value
                    wandb.log(metrics)
                    create_predictions_plot(
                        predictions.predictions,
                        predictions.label_ids,
                        wandb_prefix="test_out_dist/ar",
                    )
