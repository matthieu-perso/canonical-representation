"""Training script for Grokking models."""

import logging
import os
import pathlib
import pprint

import hydra
import hydra.core
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from grokking.config_classes.constants import GROKKING_REPOSITORY_BASE_PATH, default_output_dir
from grokking.config_classes.local_estimates.plot_config import LocalEstimatesPlotConfig, PlotSavingConfig
from grokking.grokk_replica.datasets import AbstractDataset
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.logging.create_and_configure_global_logger import create_and_configure_global_logger
from grokking.logging.log_dataframe_info import rich_table_to_string
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.model_handling.set_seed import set_seed
from grokking.scripts.dataset_for_topological_analysis import DatasetForTopologicalAnalysis
from grokking.scripts.do_topological_analysis_step import do_topological_analysis_step
from grokking.scripts.generate_tsne_visualizations import generate_tsne_visualizations
from grokking.scripts.group_dataset import GroupDataset
from grokking.scripts.input_and_hidden_states_array import InputAndHiddenStatesArray
from grokking.scripts.lr_scheduler_config import LRSchedulerConfig
from grokking.geometry.analysis import compute_geometry_at_checkpoint
from grokking.geometry.path import get_flat_params, segment_natural_length
from grokking.scripts.training_loop_state import TrainingLoopState
from grokking.typing.constants import MODEL_CLASS, OPTIMIZER_CLASS
from grokking.typing.enums import Verbosity

# Increase the wandb service wait time to prevent errors on HPC cluster.
# https://github.com/wandb/wandb/issues/5214
os.environ["WANDB__SERVICE_WAIT"] = "300"


# Logger for this file
global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_device: torch.device = torch.device(
    device="cpu",
)


def train(
    config: dict,
    output_dir: os.PathLike = default_output_dir,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
    *,
    cuda_device_id: int | None = None,
) -> None:
    """Train the model using the provided configuration."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using config:\n{pprint.pformat(config)}",  # noqa: G004 - low overhead
        )

    train_cfg: dict = config["train"]
    max_steps: int = train_cfg["max_steps"]
    lr_scheduler_cfg: dict = train_cfg["lr_scheduler"]
    optimizer_cfg: dict = train_cfg["optimizer"]
    logging_cfg: dict = config["logging"]

    topological_analysis_cfg: dict = config["topological_analysis"]
    wandb_cfg: dict = config["wandb"]
    use_wandb: bool = wandb_cfg["use_wandb"]
    load_checkpoint_from_dir: None | str = train_cfg["load_checkpoint_from_dir"]

    # Use the global seed to initialize the random number generators and torch initialization.
    global_seed: int = train_cfg["global_seed"]
    set_seed(
        seed=global_seed,
        logger=logger,
    )

    device: torch.device = get_torch_device(
        preferred_torch_backend=train_cfg["preferred_torch_backend"],
        verbosity=verbosity,
        logger=logger,
        cuda_device_id=cuda_device_id,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(msg=f"Training device: {device} (MPS = Apple Silicon GPU)")

    if load_checkpoint_from_dir is not None:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading from directory: {load_checkpoint_from_dir = } ...",  # noqa: G004 - low overhead
            )

        # Load the TrainingLoopState from the specified directory
        training_loop_state: TrainingLoopState = TrainingLoopState.from_checkpoints_root_dir(
            checkpoints_root_dir=pathlib.Path(
                load_checkpoint_from_dir,
            ),
            output_dir=output_dir,
            map_location=device,
            verbosity=verbosity,
            logger=logger,
        )

        # Make sure that we increase the step by 1 after loading the checkpoint,
        # because this is necessary to be consistent for the next training step iteration,
        # since the training step increments only at the end of the training loop step after the saving.
        training_loop_state.step += 1

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading from directory: {load_checkpoint_from_dir = } DONE",  # noqa: G004 - low overhead
            )
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Preparing to train from scratch ...",
            )

        # # # # # # # #
        # Datasets
        dataset: AbstractDataset = load_item(
            config["dataset"],
        )
        train_data = GroupDataset(
            dataset=dataset,
            split="train",
        )
        val_data = GroupDataset(
            dataset=dataset,
            split="val",
        )

        # Notes:
        # - We create new instances of the datasets for the topological analysis,
        #   so that we can consume new batches of data for each analysis step.
        #   This is important because the datasets are iterable, so we need to
        #   create new instances to get new batches.
        datasets_for_topological_analysis_list: list[DatasetForTopologicalAnalysis] = [
            DatasetForTopologicalAnalysis(
                group_dataset=GroupDataset(
                    dataset=dataset,
                    split=split,
                ),
                split=split,
                train_cfg=train_cfg,
            )
            for split in [
                "train",
                "val",
            ]
        ]

        # Notes:
        # - In the current setup without shuffling, the dataloaders will not introduce any non-determinism.
        train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=train_cfg["bsize"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
        )
        val_dataloader = DataLoader(
            dataset=val_data,
            batch_size=train_cfg["bsize"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
        )

        # # # #
        # Model, Optimizer, LR Scheduler
        #
        # Notes:
        # - We collect the keyword args into separate objects,
        #   so that we can save them to the checkpoint checkpoint data.

        model_init_kwargs: dict = {
            "transformer_config": config["model"]["transformer_config"],
            "vocab_size": dataset.n_vocab,
            "output_size": dataset.n_out,
            "device": device,
        }
        model: MODEL_CLASS = MODEL_CLASS(
            **model_init_kwargs,
        )
        model.to(device=device)
        model.train()

        optimizer_init_kwargs: dict = {
            "lr": optimizer_cfg["lr"],
            "betas": optimizer_cfg["betas"],
            "eps": optimizer_cfg["eps"],
            "weight_decay": optimizer_cfg["weight_decay"],
        }
        optimizer: OPTIMIZER_CLASS = OPTIMIZER_CLASS(
            params=model.parameters(),
            **optimizer_init_kwargs,
        )

        lr_scheduler_init_kwargs: dict = {
            "lr_scheduler_type": lr_scheduler_cfg["lr_scheduler_type"],
            "warmup_steps": lr_scheduler_cfg["warmup_steps"],
            "total_steps": max_steps,
        }
        lr_schedule: SequentialLR = LRSchedulerConfig(
            **lr_scheduler_init_kwargs,
        ).build(
            optimizer=optimizer,
            last_step=-1,  # -1 means the learning rate schedule starts from the beginning
        )

        training_loop_state: TrainingLoopState = TrainingLoopState(
            model_init_kwargs=model_init_kwargs,
            model=model,
            optimizer=optimizer,
            optimizer_init_kwargs=optimizer_init_kwargs,
            lr_scheduler_init_kwargs=lr_scheduler_init_kwargs,
            lr_scheduler=lr_schedule,
            training_logs={},
            dataset=dataset,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            datasets_for_topological_analysis_list=datasets_for_topological_analysis_list,
            output_dir=output_dir,
            device=device,
            step=0,  # Set step to 0 when starting training from scratch.
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Preparing to train from scratch DONE",
            )

    if verbosity >= Verbosity.NORMAL:
        training_loop_state.log_info(
            logger=logger,
        )

    if use_wandb:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Adding wandb watch of model ...",
            )

        # Make a call to wandb.log() before the call to wandb.watch() to make sure the logging works correctly.
        # https://community.wandb.ai/t/wandb-watch-not-logging-parameters/1197/14
        wandb.log(
            data={"step": training_loop_state.step},
        )

        wandb.watch(
            models=training_loop_state.model,
            log=wandb_cfg["watch"]["log"],
            log_freq=wandb_cfg["watch"]["log_freq"],
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Adding wandb watch of model DONE",
            )

    training_log_example_batch_every: int = logging_cfg["training"]["log_example_batch_every"]
    number_of_entries_in_example_batch: int = logging_cfg["training"]["number_of_entries_in_example_batch"]
    training_create_plot_of_model_output_parameters_every: int = logging_cfg["training"][
        "create_plot_of_model_output_parameters_every"
    ]
    save_checkpoints_every: int = train_cfg["save_checkpoints_every"]
    compute_geometry_every: int = train_cfg.get("compute_geometry_every", 0)
    topological_analysis_compute_estimates_every: int = topological_analysis_cfg["compute_estimates_every"]
    topological_analysis_create_projection_plot_every: int = topological_analysis_cfg["create_projection_plot_every"]

    # # # #
    # Training loop
    training_loop_state.model.train()

    progress_bar = tqdm(
        training_loop_state.train_dataloader,
        desc="Training loop",
        total=max_steps,
        position=0,
    )

    prev_theta_for_path: torch.Tensor | None = None
    cumulative_natural_path_length: float = 0.0

    for (
        x,
        y,
    ) in progress_bar:
        if training_log_example_batch_every > 0 and training_loop_state.step % training_log_example_batch_every == 0:
            log_example_batch(
                x=x,
                y=y,
                dataset=training_loop_state.dataset,
                step=training_loop_state.step,
                number_of_entries_in_example_batch=number_of_entries_in_example_batch,
                verbosity=verbosity,
                logger=logger,
            )

        training_logs: dict = do_training_step(
            model=training_loop_state.model,
            optimizer=training_loop_state.optimizer,
            clip_grad_norm_max_norm=optimizer_cfg["clip_grad_norm_max_norm"],
            lr_schedule=training_loop_state.lr_scheduler,
            x=x,
            y=y,
            device=training_loop_state.device,
        )
        # Example of the `training_loop_state.training_logs` dict:
        # > {
        # >     "loss": (4.739171028137207, 512),
        # >     "accuracy": (0.00390625, 512),
        # >     "attn_entropy": (0.6624060869216919, 3072),
        # >     "param_norm": (122.12102613376149, 1),
        # > }

        progress_bar.set_postfix(
            ordered_dict=combine_logs(logs=[training_logs]),
        )

        # # # #
        # Evaluation step
        if (training_loop_state.step + 1) % train_cfg["eval_every"] == 0:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Running evaluation for {training_loop_state.step + 1 = } ...",  # noqa: G004 - low overhead
                )

            all_val_logs: list[dict] = do_eval_step(
                model=training_loop_state.model,
                val_dataloader=training_loop_state.val_dataloader,
                train_cfg=train_cfg,
                device=training_loop_state.device,
                use_tqdm_for_eval_step=False,
                verbosity=verbosity,
                logger=logger,
            )

            out_log: dict = {
                "val": combine_logs(logs=all_val_logs),
                "train": combine_logs(logs=[training_logs]),
                "step": (training_loop_state.step + 1),
                "lr": float(training_loop_state.lr_scheduler.get_last_lr()[0]),
            }
            training_loop_state.training_logs = out_log

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"{training_loop_state.training_logs}",  # noqa: G004 - low overhead
                )

            if use_wandb:
                # Example of the `training_loop_state.training_logs` dict:
                #
                # > {
                # >     "val": {
                # >         "loss": 4.6041805148124695,
                # >         "accuracy": 0.00927734375,
                # >         "attn_entropy": 0.7255031652748585,
                # >         "param_norm": 121.61177041725013,
                # >     },
                # >     "train": {
                # >         "loss": 4.638635158538818,
                # >         "accuracy": 0.015625,
                # >         "attn_entropy": 0.6404158473014832,
                # >         "param_norm": 121.60770918646537,
                # >     },
                # >     "step": 10,
                # >     "lr": 0.001,
                # > }
                wandb.log(
                    data=training_loop_state.training_logs,
                )

                logger.info(
                    msg=f"Running evaluation for {training_loop_state.step + 1 = } DONE",  # noqa: G004 - low overhead
                )

        # # # #
        # Embedding space analysis step

        if (
            topological_analysis_compute_estimates_every > 0
            and (training_loop_state.step + 1) % topological_analysis_compute_estimates_every == 0
        ):
            do_topological_analysis_step(
                datasets_for_topological_analysis_list=training_loop_state.datasets_for_topological_analysis_list,
                model=training_loop_state.model,
                output_dir=training_loop_state.output_dir,
                topological_analysis_cfg=topological_analysis_cfg,
                step=training_loop_state.step,
                topological_analysis_create_projection_plot_every=topological_analysis_create_projection_plot_every,
                use_wandb=use_wandb,
                device=training_loop_state.device,
                verbosity=verbosity,
                logger=logger,
            )

        # # # #
        # Analyse the model parameters
        if (
            training_create_plot_of_model_output_parameters_every > 0
            and (training_loop_state.step + 1) % training_create_plot_of_model_output_parameters_every == 0
        ):
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Creating plot of output layer ...",
                )

            output_layer: torch.nn.modules.linear.Linear = training_loop_state.model.transformer.output

            # Example shapes for p=97:
            # > output_layer.weight.shape
            # > torch.Size([97, 128])
            # > output_layer.bias.shape
            # > torch.Size([97])

            output_layer_weight_np: np.ndarray = output_layer.weight.cpu().detach().numpy()
            # The vectors in the output layer correspond to the input_ids
            # in the same order.
            output_layer_input_and_hidden_states_array = InputAndHiddenStatesArray(
                input_x=list(range(0, output_layer_weight_np.shape[0])),
                hidden_states=output_layer_weight_np,
            )

            local_estimates_plot_config = LocalEstimatesPlotConfig(
                pca_n_components=None,  # Skip the PCA step
                saving=PlotSavingConfig(
                    save_html=False,  # Since the .html is quite large, we skip saving it for now
                    save_pdf=True,
                    save_csv=True,
                ),
            )

            saved_plots_output_layer_root_dir: pathlib.Path = pathlib.Path(
                output_dir,
                "plots",
                "output_layer_projection",
                f"step+1={training_loop_state.step + 1}",
            )

            generate_tsne_visualizations(
                input_and_hidden_states_array=output_layer_input_and_hidden_states_array,
                pointwise_results_array_np=None,
                local_estimates_plot_config=local_estimates_plot_config,
                saved_plots_root_dir=saved_plots_output_layer_root_dir,
                verbosity=verbosity,
                logger=logger,
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Creating plot of output layer DONE",
                )

        # # # #
        # Optionally: Save the model, optimizer and dataloader
        #
        # Notes:
        # - We use `step + 1` here,
        #   because we do not want to save the model for step == 0,
        #   i.e., at the beginning of training.
        # - We save the dataloaders at the end of the training loop step,
        #   so that we can resume training from the last checkpoint,
        #   and the iterable loaders are at the correct position for the next step.
        if save_checkpoints_every > 0 and (training_loop_state.step + 1) % save_checkpoints_every == 0:
            training_loop_state.save_to_folder(
                verbosity=verbosity,
                logger=logger,
            )

        # # # #
        # Geometry analysis (Fisher, curvature, sharpness)
        if (
            compute_geometry_every > 0
            and (training_loop_state.step + 1) % compute_geometry_every == 0
        ):
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"step + 1 = {training_loop_state.step + 1}: Running geometry analysis ...",  # noqa: G004
                )
            try:
                curr_theta = get_flat_params(
                    training_loop_state.model,
                    device=training_loop_state.device,
                )
                segment_nat: float | None = None
                if prev_theta_for_path is not None:
                    try:
                        segment_nat = segment_natural_length(
                            prev_theta_for_path,
                            curr_theta,
                            training_loop_state.model,
                            training_loop_state.train_dataloader,
                            training_loop_state.device,
                            n_fisher_batches=20,
                            top_k=10,
                        )
                        cumulative_natural_path_length += segment_nat
                    except Exception as path_e:
                        if verbosity >= Verbosity.NORMAL:
                            logger.warning(
                                msg=f"step + 1 = {training_loop_state.step + 1}: path length failed: {path_e}",  # noqa: G004
                            )

                geometry_result: dict = compute_geometry_at_checkpoint(
                    model=training_loop_state.model,
                    train_dataloader=training_loop_state.train_dataloader,
                    device=training_loop_state.device,
                    step=training_loop_state.step + 1,
                )
                if segment_nat is not None:
                    geometry_result["segment_natural_length"] = segment_nat
                    geometry_result["cumulative_natural_path_length"] = cumulative_natural_path_length
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg=f"step + 1 = {training_loop_state.step + 1}: geometry = {geometry_result}",  # noqa: G004
                    )
                if use_wandb:
                    wandb.log(
                        data={f"geometry/{k}": v for k, v in geometry_result.items() if k != "step" and not isinstance(v, list)},
                        step=training_loop_state.step + 1,
                    )
                    for i, eig in enumerate(geometry_result.get("fisher_top_eigenvalues", [])):
                        wandb.log(
                            data={"geometry/fisher_eig_{}".format(i): eig},
                            step=training_loop_state.step + 1,
                        )
                    for i, eig in enumerate(geometry_result.get("representation_spectrum", [])):
                        wandb.log(
                            data={"geometry/representation_eig_{}".format(i): eig},
                            step=training_loop_state.step + 1,
                        )
                prev_theta_for_path = curr_theta.detach().clone()
            except Exception as e:
                if verbosity >= Verbosity.NORMAL:
                    logger.warning(
                        msg=f"step + 1 = {training_loop_state.step + 1}: geometry analysis failed: {e}",  # noqa: G004
                    )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"step + 1 = {training_loop_state.step + 1}: Running geometry analysis DONE",  # noqa: G004
                )

        # # # #
        # Finalize training loop step
        training_loop_state.step += 1  # >>> noqa: SIM113 - we want to have manual control over the step variable

        # Break condition
        if max_steps is not None and training_loop_state.step >= max_steps:
            logger.info(
                msg=f"Training loop finished after {training_loop_state.step + 1} steps by reaching max steps.",  # noqa: G004 - low overhead
            )
            break


def log_example_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    dataset: AbstractDataset,
    step: int,
    number_of_entries_in_example_batch: int,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Log an example batch of data."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Logging example batch for {step = } ...",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{x.shape = }; {y.shape = }",  # noqa: G004 - low overhead
        )

        # Comment about decoding:
        # `train_dataloader.dataset` is a GroupDataset, we need to go one level deeper to access the instance
        # which inherits from AbstractDataset and provides the decode method.
        # > train_dataloader.dataset.dataset.decode(x[0])
        # Here we still have access to the original dataset, so we can just use this for decoding.

    number_of_entries_to_log: int = min(
        number_of_entries_in_example_batch,
        len(x),
    )
    collected_examples_list: list = [
        batch_to_table_entry(
            x=x,
            y=y,
            index=i,
            dataset=dataset,
        )
        for i in range(number_of_entries_to_log)
    ]
    collected_examples_df: pd.DataFrame = pd.DataFrame(
        data=collected_examples_list,
    )

    table_str: str = rich_table_to_string(
        df=collected_examples_df,
        max_rows=number_of_entries_in_example_batch,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Example batch for {step = }:\n{table_str}",  # noqa: G004 - low overhead
        )


def batch_to_table_entry(
    x: torch.Tensor,
    y: torch.Tensor,
    index: int,
    dataset: AbstractDataset,
) -> dict:
    """Convert a batch to a table entry."""
    entry: dict = {
        "x": x[index].cpu().numpy().tolist(),
        "x_decoded": dataset.decode(sequence=x[index]),
        "y": y[index].cpu().numpy().tolist(),
    }
    return entry


def do_training_step(
    model: GrokkModel,
    optimizer: torch.optim.Optimizer,
    clip_grad_norm_max_norm: float | None,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> dict:
    (
        loss,
        logs,
    ) = model.get_loss(
        x=x.to(device=device),
        y=y.to(device=device),
    )

    optimizer.zero_grad()
    loss.backward()

    # Notes:
    # - Using instructions from here:
    #   https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
    # - The gradient norm clipping works in place
    if clip_grad_norm_max_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=clip_grad_norm_max_norm,
        )

    optimizer.step()
    lr_schedule.step()

    return logs


def do_eval_step(
    model: GrokkModel,
    val_dataloader: DataLoader,
    train_cfg: dict,
    device: torch.device,
    *,
    use_tqdm_for_eval_step: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[dict]:
    model.eval()

    num_eval_batches = train_cfg["eval_batches"]

    eval_iterable = enumerate(
        iterable=val_dataloader,
    )
    if use_tqdm_for_eval_step:
        eval_iterable = tqdm(
            eval_iterable,
            desc="Evaluating validation data.",
            total=num_eval_batches,
        )

    with torch.no_grad():
        all_val_logs = []
        for (
            i,
            (
                val_x,
                val_y,
            ),
        ) in eval_iterable:
            if i >= num_eval_batches:
                break

            (
                _,
                val_logs,
            ) = model.get_loss(
                val_x.to(device),
                val_y.to(device),
            )
            all_val_logs.append(
                val_logs,
            )

    model.train()

    return all_val_logs


@hydra.main(
    config_path="../../config",
    config_name="train_grokk",
    version_base="1.3",
)
def main(
    cfg: DictConfig,
) -> None:
    """Train the model."""
    logger: logging.Logger = global_logger
    verbosity: Verbosity = Verbosity.NORMAL

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Main function ...",
        )

    cfg_as_container = OmegaConf.to_container(
        cfg=cfg,
        resolve=True,
    )
    if not isinstance(
        cfg_as_container,
        dict,
    ):
        msg = "The configuration must be a dictionary."
        raise TypeError(
            msg,
        )
    config: dict = cfg_as_container

    cwd: pathlib.Path = pathlib.Path.cwd()
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore - problem with this import type

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Current working directory:\n{cwd = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Hydra output directory:\n{hydra_output_dir = }",  # noqa: G004 - low overhead
        )

    wandb_cfg: dict = config["wandb"]
    use_wandb: bool = wandb_cfg["use_wandb"]

    # In multirun, pin each job to a GPU by job index so parallel jobs (e.g. joblib) use different GPUs.
    cuda_device_id: int | None = None
    job_num_for_log: int | None = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        hydra_job = config.get("hydra") or {}
        job_cfg = hydra_job.get("job") if isinstance(hydra_job, dict) else {}
        job_num = job_cfg.get("num", 0) if isinstance(job_cfg, dict) else 0
        job_num_for_log = int(job_num)
        cuda_device_id = job_num_for_log % torch.cuda.device_count()

    if job_num_for_log is not None and verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Hydra job num = {job_num_for_log} -> GPU {cuda_device_id} (of {torch.cuda.device_count()} visible)",  # noqa: G004
        )

    if use_wandb:
        logger.info(
            msg=f"Initializing wandb with project name: {wandb_cfg['wandb_project']}",  # noqa: G004 - low overhead
        )

        # Save wandb files to a subdirectory with the project name
        wandb_output_dir = pathlib.Path(
            GROKKING_REPOSITORY_BASE_PATH,
            wandb_cfg["wandb_dir"],
        )
        if not wandb_output_dir.exists():
            logger.info(
                msg=f"Creating wandb output directory:\n{wandb_output_dir=}",  # noqa: G004 - low overhead
            )
            wandb_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        wandb.init(
            project=wandb_cfg["wandb_project"],
            dir=wandb_output_dir,
            notes=wandb_cfg["wandb_notes"],
            config=config,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling train() function ...",
        )

    train(
        config=config,
        output_dir=hydra_output_dir,
        verbosity=verbosity,
        logger=logger,
        cuda_device_id=cuda_device_id,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling train() function DONE",
        )

    if use_wandb:
        logger.info(
            msg="Finishing wandb project.",
        )
        wandb.finish()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Main function DONE",
        )


if __name__ == "__main__":
    main()
