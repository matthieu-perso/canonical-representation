#!/usr/bin/env python3
"""Fast modular-addition runs that log canonical-geometry alignment to Weights & Biases.

Trains a small transformer for a limited number of steps (no full grokking run) and logs
metrics under ``generalisation/*`` that quantify how the last-layer weight matrix aligns
with the Fourier (irrep) subspace for :math:`\\mathbb{Z}_p`.

Typical wall time: well under 15 minutes for default settings on a single GPU/MPS device.

Example::

    uv run python grokking/scripts/fast_grokking_geometry_wandb.py \\
        --wandb-project generalisation_grokking \\
        --p 197 --max-steps 4000 --eval-every 50

Environment: ``WANDB_ENTITY`` (optional), ``WANDB_MODE`` (e.g. ``offline`` for tests).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from grokking.config_classes.constants import GROKKING_REPOSITORY_BASE_PATH
from grokking.geometry.canonical_alignment import alignment_from_output_weight
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.model_handling.set_seed import set_seed
from grokking.typing.enums import PreferredTorchBackend, Verbosity
from grokking.scripts.group_dataset import GroupDataset
from grokking.scripts.lr_scheduler_config import LRSchedulerConfig
from grokking.scripts.train_grokk import do_eval_step, do_training_step

os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast grokking + canonical geometry logging for W&B.")
    p.add_argument("--wandb-project", type=str, default="generalisation_grokking")
    p.add_argument("--wandb-group", type=str, default="fast_canonical_alignment")
    p.add_argument("--wandb-mode", type=str, default=None, help="e.g. offline, disabled")
    p.add_argument("--p", type=int, default=197, help="Prime modulus (matches paper-style mod sum).")
    p.add_argument(
        "--fracs",
        type=str,
        default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45",
        help="Comma-separated training fractions (one W&B run each).",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--bsize", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--fourier-k", type=int, default=1, help="Irrep frequency index k (1 .. (p-1)/2).")
    p.add_argument("--eval-batches", type=int, default=16, help="Validation batches per eval (stochastic stream).")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-blocks", type=int, default=1)
    p.add_argument("--attn-dim", type=int, default=16)
    p.add_argument("--intermediate-dim", type=int, default=256)
    return p


def _small_transformer_config(
    *,
    hidden_dim: int,
    num_blocks: int,
    attn_dim: int,
    intermediate_dim: int,
) -> dict[str, Any]:
    return {
        "max_length": 5,
        "heads": 4,
        "hidden_dim": hidden_dim,
        "attn_dim": attn_dim,
        "intermediate_dim": intermediate_dim,
        "num_blocks": num_blocks,
        "block_repeats": 1,
        "dropout": 0.1,
        "pre_norm": True,
    }


def _train_cfg_template(
    *,
    bsize: int,
    max_steps: int,
    eval_batches: int,
) -> dict[str, Any]:
    return {
        "num_workers": 0,
        "bsize": bsize,
        "eval_every": 1,  # unused; we control eval in-script
        "eval_batches": eval_batches,
        "max_steps": max_steps,
        "preferred_torch_backend": "auto",
        "global_seed": 0,
        "lr_scheduler": {"lr_scheduler_type": "constant", "warmup_steps": 10},
        "optimizer": {
            "lr": 1e-3,
            "betas": [0.9, 0.98],
            "weight_decay": 0.01,
            "eps": 1e-6,
            "clip_grad_norm_max_norm": 1.0,
        },
    }


def _run_one_fraction(
    *,
    frac: float,
    p: int,
    seed: int,
    max_steps: int,
    eval_every: int,
    bsize: int,
    lr: float,
    weight_decay: float,
    fourier_k: int,
    eval_batches: int,
    wandb_project: str,
    wandb_group: str,
    wandb_mode: str | None,
    hidden_dim: int,
    num_blocks: int,
    attn_dim: int,
    intermediate_dim: int,
) -> None:
    set_seed(seed=seed, logger=log)

    device = get_torch_device(
        preferred_torch_backend=PreferredTorchBackend.AUTO,
        verbosity=Verbosity.QUIET,
        logger=log,
        cuda_device_id=None,
    )

    dataset = load_item(
        {
            "name": "mod_sum_dataset",
            "p": p,
            "frac_train": frac,
            "dataset_seed": seed,
        },
    )
    train_data = GroupDataset(dataset=dataset, split="train")
    val_data = GroupDataset(dataset=dataset, split="val")
    train_dataloader = DataLoader(dataset=train_data, batch_size=bsize, shuffle=False, num_workers=0)
    val_dataloader = DataLoader(dataset=val_data, batch_size=bsize, shuffle=False, num_workers=0)

    transformer_config = _small_transformer_config(
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        attn_dim=attn_dim,
        intermediate_dim=intermediate_dim,
    )
    model = GrokkModel(
        transformer_config=transformer_config,
        vocab_size=dataset.n_vocab,
        output_size=dataset.n_out,
        device=device,
    )
    model.to(device=device)
    model.train()

    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=weight_decay,
    )
    lr_schedule = LRSchedulerConfig(
        lr_scheduler_type="constant",
        warmup_steps=10,
        total_steps=max_steps,
    ).build(optimizer=optimizer, last_step=-1)

    train_cfg = _train_cfg_template(bsize=bsize, max_steps=max_steps, eval_batches=eval_batches)
    train_cfg["optimizer"]["lr"] = lr
    train_cfg["optimizer"]["weight_decay"] = weight_decay

    run_name = f"mod_sum_dataset_p{p}_frac{frac}_seed{seed}"

    wandb_dir = pathlib.Path(GROKKING_REPOSITORY_BASE_PATH) / "wandb_dir" / wandb_project
    wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=wandb_project,
        dir=str(wandb_dir),
        group=wandb_group,
        name=run_name,
        mode=wandb_mode,
        config={
            "task": "mod_sum",
            "p": p,
            "frac_train": frac,
            "seed": seed,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "fourier_k": fourier_k,
            "transformer_config": transformer_config,
        },
    )

    history: dict[str, list[float]] = {
        "step": [],
        "val_acc": [],
        "delta": [],
        "mag_err": [],
    }

    train_iter = iter(train_dataloader)
    progress = tqdm(range(max_steps), desc=run_name)
    for step in progress:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x, y = next(train_iter)

        logs = do_training_step(
            model=model,
            optimizer=optimizer,
            clip_grad_norm_max_norm=1.0,
            lr_schedule=lr_schedule,
            x=x,
            y=y,
            device=device,
        )
        progress.set_postfix(loss=f"{combine_logs([logs])['loss']:.4f}")

        if (step + 1) % eval_every == 0 or step == 0:
            all_val_logs = do_eval_step(
                model=model,
                val_dataloader=val_dataloader,
                train_cfg=train_cfg,
                device=device,
                use_tqdm_for_eval_step=False,
                verbosity=Verbosity.QUIET,
                logger=log,
            )
            val_logs = combine_logs(logs=all_val_logs)
            val_acc = float(val_logs["accuracy"])

            with torch.no_grad():
                w = model.transformer.output.weight
                m = alignment_from_output_weight(w, p=p, k=fourier_k)

            step_id = step + 1
            wandb.log(
                {
                    "generalisation/prediction_generalization_accuracy": val_acc,
                    "generalisation/ideal_2d_target_angle_rad": m["ideal_2d_target_angle_rad"],
                    "generalisation/ideal_distance_magnitude_error": m["ideal_distance_magnitude_error"],
                    "generalisation/ideal_implemented_angle_rad": m["ideal_implemented_angle_rad_mean"],
                    "generalisation/ideal_implemented_angle_rad_max": m["ideal_implemented_angle_rad_max"],
                    "generalisation/ideal_space_delta_to_ideal": m["ideal_space_delta_to_ideal"],
                    "generalisation/ideal_in_subspace_energy_ratio": m["ideal_in_subspace_energy_ratio"],
                    "train/loss": float(combine_logs([logs])["loss"]),
                    "train/accuracy": float(combine_logs([logs])["accuracy"]),
                },
                step=step_id,
            )

            history["step"].append(float(step_id))
            history["val_acc"].append(val_acc)
            history["delta"].append(m["ideal_space_delta_to_ideal"])
            history["mag_err"].append(m["ideal_distance_magnitude_error"])

    # Summary figure (dashboard-style panel)
    target_angle_rad = 2.0 * math.pi * fourier_k / p
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()
    ax0.plot(history["step"], history["val_acc"])
    ax0.set_title("prediction_generalization_accuracy")
    ax0.set_xlabel("step")
    ax0.set_ylabel("accuracy")

    if history["step"]:
        ax1.plot(
            [history["step"][0], history["step"][-1]],
            [target_angle_rad, target_angle_rad],
            color="C1",
        )
    ax1.set_title("ideal_2d_target_angle_rad (constant)")
    ax1.set_xlabel("step")

    ax2.plot(history["step"], history["mag_err"])
    ax2.set_title("ideal_distance_magnitude_error")
    ax2.set_xlabel("step")

    ax3.plot(history["step"], history["delta"])
    ax3.set_title("ideal_space_delta_to_ideal")
    ax3.set_xlabel("step")

    fig.suptitle(f"{run_name} — canonical Fourier alignment (last layer)")
    wandb.log({"generalisation/alignment_summary_figure": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    fracs = [float(x.strip()) for x in args.fracs.split(",") if x.strip()]

    t0 = time.perf_counter()
    for frac in fracs:
        log.info("Starting run for frac_train=%s", frac)
        _run_one_fraction(
            frac=frac,
            p=args.p,
            seed=args.seed,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            bsize=args.bsize,
            lr=args.lr,
            weight_decay=args.weight_decay,
            fourier_k=args.fourier_k,
            eval_batches=args.eval_batches,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            wandb_mode=args.wandb_mode or os.environ.get("WANDB_MODE"),
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            attn_dim=args.attn_dim,
            intermediate_dim=args.intermediate_dim,
        )

    elapsed = time.perf_counter() - t0
    log.info("Finished %d runs in %.1f s (%.2f min)", len(fracs), elapsed, elapsed / 60.0)


if __name__ == "__main__":
    main()
