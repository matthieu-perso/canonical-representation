#!/usr/bin/env python3
"""Seven experiments testing the canonical geometry theory of grokking.

Experiments (select with --experiment):

  A  geometry_phase_transition   Full training run tracking ε, per-irrep spectrum,
                                  margins, and projection intervention simultaneously.
                                  Covers paper claims C6, C7 and paper lines:
                                  "test the generalisation distance" (A),
                                  "test where we find the errors" (B),
                                  "test if we remove the noise" (C).

  D  noise_robustness            Add Gaussian noise σ to logits of a grokked model.
                                  Verifies the 2σ degradation bound from Section 3.1.
                                  Paper line: "test with perturbations of the spaces".

  E  data_threshold              Sweep frac_train.  Finds n* (critical data fraction
                                  below which ε never collapses).  Section 3.3 / C9.

  F  weight_decay_geometry       Sweep weight_decay.  Shows WD drives ε→0 (claim C8).
                                  Section 3.3 / Section 4.4 "Regularizer".

  G  canonical_regularizer       Add λ‖Π_{(V_ks)⊥}W_L‖²_F to the CE loss.  Tests
                                  whether directly penalising ε causes grokking even
                                  below n*.  Paper line: "test if we constraint the
                                  space".

Usage::

    uv run canonical_geometry_experiments --experiment geometry_phase_transition
    uv run canonical_geometry_experiments --experiment data_threshold \\
        --fracs 0.1,0.15,0.2,0.25,0.3,0.4,0.5
    uv run canonical_geometry_experiments --experiment canonical_regularizer \\
        --lambdas 0,0.01,0.1,1.0 --fracs 0.15,0.3
    uv run canonical_geometry_experiments --experiment noise_robustness \\
        --sigmas 0,0.5,1,2,5,10
    uv run canonical_geometry_experiments --experiment weight_decay_geometry \\
        --weight-decays 0,0.0001,0.001,0.01,0.1,1.0

Environment variables:
    WANDB_ENTITY   optional W&B entity
    WANDB_MODE     e.g. ``offline`` or ``disabled`` for local/CI runs
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import pathlib
from typing import Any

import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from grokking.config_classes.constants import GROKKING_REPOSITORY_BASE_PATH
from grokking.geometry.canonical_alignment import (
    build_delta_permutation,
    class_mean_activations,
    compute_K_min,
    embedding_geometry,
    empirical_margins,
    fourier_mode_subspace,
    frob_sq_off_canonical,
    mean_logit_canonical_deviation,
    per_irrep_energy_spectrum,
    per_sample_excess_loss,
    predicted_margin_curve,
    project_onto_canonical,
    reference_k_theory,
    spectrum_energy_on_ks,
)
from grokking.grokk_replica.grokk_model import GrokkModel
from grokking.grokk_replica.load_objs import load_item
from grokking.grokk_replica.utils import combine_logs
from grokking.model_handling.get_torch_device import get_torch_device
from grokking.model_handling.set_seed import set_seed
from grokking.scripts.group_dataset import GroupDataset
from grokking.scripts.lr_scheduler_config import LRSchedulerConfig
from grokking.scripts.train_grokk import do_eval_step, do_training_step
from grokking.typing.enums import PreferredTorchBackend, Verbosity

os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclasses (keeps function signatures ≤ 7 args)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ModelConfig:
    hidden_dim: int = 128
    num_blocks: int = 2
    attn_dim: int = 32
    intermediate_dim: int = 512
    heads: int = 4


@dataclasses.dataclass
class _RunConfig:
    """Consolidated training hyperparameters for a single run."""

    p: int = 113
    frac_train: float = 0.3
    seed: int = 42
    max_steps: int = 50_000
    eval_every: int = 200
    eval_batches: int = 16
    bsize: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    # Geometry logging frequency (every N eval checkpoints; 0 = only first+last)
    margins_every_n_evals: int = 5
    # Number of examples sampled for margin / error-localisation estimates
    margin_samples: int = 512
    do_projection_test: bool = True
    # Canonical regulariser (experiment G).
    # `lambda_canonical`   — penalises ‖Π_⊥ W_L‖²_F on the readout weight (weight-side).
    # `lambda_canonical_H` — penalises mean-per-sample ‖Π_⊥ z‖² on the batch's last-position
    #                        logits (logit-side; matches paper's β‖H‖² term).
    lambda_canonical: float = 0.0
    lambda_canonical_H: float = 0.0
    canonical_ks: tuple[int, ...] = (1,)
    # Reference irrep set K(p) from paper Eq. 8 (compute_K_min); used for K-aligned energy metrics
    k_theory_alpha: float = 1.0
    k_theory_sigma: float = 0.0
    k_theory_epsilon_target: float = 0.01


@dataclasses.dataclass
class _WandbConfig:
    project: str = "canonical_repr_grokking"
    group: str = "default"
    name: str = ""
    mode: str | None = None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_dataset(p: int, frac_train: float, seed: int) -> Any:
    return load_item({"name": "mod_sum_dataset", "p": p, "frac_train": frac_train, "dataset_seed": seed})


def _make_full_loader(dataset: Any, bsize: int) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """Build a DataLoader over ALL p² examples and class<->element mappings.

    Returns (full_loader, class_to_element_tensor, perm) where:
      class_to_element_tensor[i] = integer group element for class index i
      perm[delta-1, y_class]     = competitor class index at group offset delta
    """
    p = dataset.n_out
    n = len(dataset.ordered_group_elements1) * len(dataset.ordered_group_elements2)
    xs: list[list[int]] = []
    ys: list[int] = []
    for idx in range(n):
        x, y, _ = dataset.fetch_example(idx)
        xs.append(x)
        ys.append(y)

    full_ds = TensorDataset(
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
    )
    full_loader = DataLoader(full_ds, batch_size=bsize, shuffle=False, num_workers=0)

    # Build element<->class index mappings
    # vocab2idx[elem] - 2 = class index  (first two vocab slots are 'o' and '=')
    class_to_element: dict[int, int] = {
        (dataset.vocab2idx[e] - 2): int(e) for e in dataset.ordered_group_elements1
    }
    element_to_class: dict[int, int] = {v: k for k, v in class_to_element.items()}
    c2e_tensor = torch.tensor([class_to_element[i] for i in range(p)], dtype=torch.long)
    perm = build_delta_permutation(p, class_to_element, element_to_class)
    return full_loader, c2e_tensor, perm


def _make_loaders(dataset: Any, bsize: int) -> tuple[DataLoader, DataLoader]:
    train_ds = GroupDataset(dataset=dataset, split="train")
    val_ds = GroupDataset(dataset=dataset, split="val")
    train_loader = DataLoader(train_ds, batch_size=bsize, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bsize, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model / optimizer helpers
# ---------------------------------------------------------------------------


def _build_model(model_cfg: _ModelConfig, dataset: Any, device: torch.device) -> GrokkModel:
    transformer_cfg = {
        "max_length": 5,
        "heads": model_cfg.heads,
        "hidden_dim": model_cfg.hidden_dim,
        "attn_dim": model_cfg.attn_dim,
        "intermediate_dim": model_cfg.intermediate_dim,
        "num_blocks": model_cfg.num_blocks,
        "block_repeats": 1,
        "dropout": 0.1,
        "pre_norm": True,
    }
    model = GrokkModel(
        transformer_config=transformer_cfg,
        vocab_size=dataset.n_vocab,
        output_size=dataset.n_out,
        device=device,
    )
    return model.to(device=device)


def _build_optimizer(model: GrokkModel, run_cfg: _RunConfig) -> tuple[AdamW, Any]:
    optimizer = AdamW(
        model.parameters(),
        lr=run_cfg.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=run_cfg.weight_decay,
    )
    lr_schedule = LRSchedulerConfig(
        lr_scheduler_type="constant",
        warmup_steps=50,
        total_steps=run_cfg.max_steps,
    ).build(optimizer=optimizer, last_step=-1)
    return optimizer, lr_schedule


# ---------------------------------------------------------------------------
# Custom training step with canonical regulariser
# ---------------------------------------------------------------------------


def _training_step_with_reg(  # noqa: PLR0913
    model: GrokkModel,
    optimizer: AdamW,
    lr_schedule: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    run_cfg: _RunConfig,
) -> tuple[dict, float]:
    """Training step that optionally adds canonical regulariser(s) to the CE loss.

    Two independent regularisers can be combined:
      * ``lambda_canonical``   on  ``‖Π_{(V_ks)⊥} W_L‖²_F``        (readout weight)
      * ``lambda_canonical_H`` on  ``mean_i ‖Π_{(V_ks)⊥} z_i‖²``   (per-sample logit deviation)
    """
    model.train()
    x_dev, y_dev = x.to(device), y.to(device)
    optimizer.zero_grad()

    # If the H regulariser is active we need logits with grad anyway → use forward+xent path.
    z: torch.Tensor | None = None
    if run_cfg.lambda_canonical_H > 0.0:
        logits, _, _ = model(x_dev)
        z = logits[:, -1, :]
        loss = torch.nn.functional.cross_entropy(z, y_dev)
        logs = {"loss": loss.item(),
                "accuracy": (z.argmax(dim=-1) == y_dev).float().mean().item()}
    else:
        loss, logs = model.get_loss(x_dev, y_dev)

    reg_val = 0.0
    if run_cfg.lambda_canonical > 0.0:
        w = model.transformer.output.weight
        reg_w = frob_sq_off_canonical(w, run_cfg.p, run_cfg.canonical_ks)
        loss = loss + run_cfg.lambda_canonical * reg_w
        reg_val += reg_w.item()

    if run_cfg.lambda_canonical_H > 0.0 and z is not None:
        z_f = z.float()
        proj = torch.zeros_like(z_f)
        for k in run_cfg.canonical_ks:
            q = fourier_mode_subspace(run_cfg.p, k, device=z_f.device, dtype=z_f.dtype)
            proj = proj + z_f @ q @ q.T
        h_sq = (z_f - proj).pow(2).sum(dim=-1).mean()
        loss = loss + run_cfg.lambda_canonical_H * h_sq.to(loss.dtype)
        reg_val += float(h_sq.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_schedule.step()
    return logs, reg_val


# ---------------------------------------------------------------------------
# Geometry logging helper (called at each eval checkpoint)
# ---------------------------------------------------------------------------


def _log_geometry(  # noqa: PLR0913
    model: GrokkModel,
    val_loader: DataLoader,
    full_loader: DataLoader,
    perm: torch.Tensor,
    device: torch.device,
    run_cfg: _RunConfig,
    step: int,
    train_logs: dict,
    reg_val: float,
    eval_counter: int,
    wandb_run: Any,
    train_loader: DataLoader | None = None,
) -> tuple[float, float]:
    """Compute all geometry metrics and log to W&B.  Returns (val_acc, epsilon)."""
    train_cfg_dict = {
        "bsize": run_cfg.bsize,
        "eval_batches": run_cfg.eval_batches,
        "max_steps": run_cfg.max_steps,
        "eval_every": 1,
        "preferred_torch_backend": "auto",
        "global_seed": 0,
        "lr_scheduler": {"lr_scheduler_type": "constant", "warmup_steps": 10},
        "optimizer": {"lr": run_cfg.lr, "betas": [0.9, 0.98], "weight_decay": run_cfg.weight_decay,
                      "eps": 1e-6, "clip_grad_norm_max_norm": 1.0},
    }
    val_log_list = do_eval_step(
        model=model, val_dataloader=val_loader, train_cfg=train_cfg_dict,
        device=device, use_tqdm_for_eval_step=False,
        verbosity=Verbosity.QUIET, logger=log,
    )
    val_logs = combine_logs(val_log_list)
    val_acc = float(val_logs["accuracy"])
    val_loss = float(val_logs["loss"])
    train_combined = combine_logs([train_logs])

    # --- Readout weight geometry (W_L, Definition 3.3) ---
    with torch.no_grad():
        w = model.transformer.output.weight  # (p, d)
        w_spectrum = per_irrep_energy_spectrum(w, run_cfg.p)
        pred_curve = predicted_margin_curve(w, run_cfg.p)

    # --- Activation geometry (φ, class-mean last hidden states) ---
    # H[c,:] = mean last hidden state over all inputs with label c = (a+b) mod p.
    # Tests whether the pre-readout representation clusters by output class in a Fourier way.
    # Use at least 20 examples per class (20*p), capped for speed.
    activation_samples = min(20 * run_cfg.p, run_cfg.p * run_cfg.p)
    H = class_mean_activations(model, full_loader, run_cfg.p, device,
                               max_samples=activation_samples)
    h_spectrum = per_irrep_energy_spectrum(H.to(w.device), run_cfg.p)

    # --- Embedding geometry (most fundamental canonical geometry test) ---
    # Token IDs: 0='op', 1='=', 2..p+1 = elements 0..p-1.
    with torch.no_grad():
        e_spectrum = embedding_geometry(model.transformer.embeddings.weight, run_cfg.p)

    # --- K_theory: greedy minimal irrep set from paper (function of p + theory hyperparams) ---
    ks_theory = reference_k_theory(
        run_cfg.p,
        alpha=run_cfg.k_theory_alpha,
        sigma=run_cfg.k_theory_sigma,
        epsilon_target=run_cfg.k_theory_epsilon_target,
    )
    # Paper Sec. 3.2: ε = ∑‖Π_{(V*)⊥} W_L φ‖² / ∑‖W_L φ‖² over inputs (full p² by default).
    logit_dev = mean_logit_canonical_deviation(
        model, full_loader, run_cfg.p, ks_theory, device, max_examples=None
    )
    w_k_theory = spectrum_energy_on_ks(w_spectrum, ks_theory)
    phi_k_theory = spectrum_energy_on_ks(h_spectrum, ks_theory)
    embed_k_theory = spectrum_energy_on_ks(e_spectrum, ks_theory)

    # --- Per-sample deviation cost (Theorems 2 & 3) ---
    # On the full p² grid (population proxy):
    psl_full = per_sample_excess_loss(
        model, full_loader, run_cfg.p, ks_theory, device, max_examples=None,
    )
    # On the train split (empirical):
    psl_train = (
        per_sample_excess_loss(
            model, train_loader, run_cfg.p, ks_theory, device,
            max_examples=run_cfg.bsize * 8,
        )
        if train_loader is not None
        else None
    )

    # Use readout spectrum as before but rename clearly
    spectrum = w_spectrum

    log_dict: dict[str, Any] = {
        "val/accuracy": val_acc,
        "val/loss": val_loss,
        "train/loss": float(train_combined["loss"]),
        "train/accuracy": float(train_combined["accuracy"]),
        # Readout geometry (ε_W): how much of W_L lies outside the best Fourier mode
        "geometry/epsilon_W": spectrum["epsilon"],
        "geometry/epsilon_W_top3": 1.0 - spectrum["top3_energy"],
        "geometry/epsilon_W_dc": spectrum["epsilon_dc"],
        "geometry/W_top1_energy": spectrum["top1_energy"],
        "geometry/W_top3_energy": spectrum["top3_energy"],
        "geometry/W_dominant_k": spectrum["dominant_k"],
        "geometry/W_spectrum_entropy": spectrum["spectrum_entropy"],
        # Activation geometry (ε_φ): how much of mean class activations H lies outside Fourier modes
        # THIS is the proper test of the canonical representation in φ(X) ⊂ R^d
        "geometry/epsilon_phi": h_spectrum["epsilon"],
        "geometry/epsilon_phi_top3": 1.0 - h_spectrum["top3_energy"],
        "geometry/phi_top1_energy": h_spectrum["top1_energy"],
        "geometry/phi_top3_energy": h_spectrum["top3_energy"],
        "geometry/phi_dominant_k": h_spectrum["dominant_k"],
        "geometry/phi_spectrum_entropy": h_spectrum["spectrum_entropy"],
        # Primary ε: fraction of *non-DC* activation energy OUTSIDE paper's K_theory(p)
        # (small ⇒ class-mean H aligns with the irreps the theory says suffice).
        "geometry/epsilon": phi_k_theory["epsilon_off_K_nt"],
        "geometry/epsilon_single_mode_phi": h_spectrum["epsilon"],
        "geometry/epsilon_top3": 1.0 - h_spectrum["top3_energy"],
        "geometry/regulariser_term": reg_val,
        # K_theory set and K-aligned energies (W, φ, embeddings)
        "geometry/K_theory_n": len(ks_theory),
        "geometry/K_theory_ks": ",".join(str(k) for k in ks_theory),
        "geometry/phi_energy_fraction_K_theory": phi_k_theory["energy_K_nontrivial_frac"],
        "geometry/W_energy_fraction_K_theory": w_k_theory["energy_K_nontrivial_frac"],
        "geometry/embed_energy_fraction_K_theory": embed_k_theory["energy_K_nontrivial_frac"],
        "geometry/phi_epsilon_off_K_theory": phi_k_theory["epsilon_off_K_nt"],
        "geometry/W_epsilon_off_K_theory": w_k_theory["epsilon_off_K_nt"],
        "geometry/embed_epsilon_off_K_theory": embed_k_theory["epsilon_off_K_nt"],
        # Strict logit-space deviation vs V* = ⊕_{k∈K_theory} V_k (matches paper definition).
        "geometry/epsilon_logits_paper": logit_dev["epsilon_logits_off_Vstar"],
        "geometry/epsilon_logits_n_examples": logit_dev["n_logit_examples"],
        # Embedding geometry (ε_embed): most fundamental canonical geometry test.
        # Measures whether token embeddings of Z_p elements form Fourier irrep circles.
        # This is what makes the canonical mechanism POSSIBLE — if ε_embed is high,
        # the model hasn't learned the irrep input encoding at all.
        "geometry/epsilon_embed": e_spectrum["epsilon"],
        "geometry/epsilon_embed_top3": 1.0 - e_spectrum["top3_energy"],
        "geometry/embed_top1_energy": e_spectrum["top1_energy"],
        "geometry/embed_top3_energy": e_spectrum["top3_energy"],
        "geometry/embed_dominant_k": e_spectrum["dominant_k"],
        "geometry/embed_spectrum_entropy": e_spectrum["spectrum_entropy"],
    }
    # Per-irrep energies for W, φ (class-mean activations), and embeddings — log all (p-1)/2 modes.
    n_modes = (run_cfg.p - 1) // 2
    for k in range(n_modes + 1):
        w_key = f"alpha_{k}"
        if w_key in spectrum:
            log_dict[f"geometry/W_alpha_{k}"] = spectrum[w_key]
        if w_key in h_spectrum:
            log_dict[f"geometry/phi_alpha_{k}"] = h_spectrum[w_key]
        if w_key in e_spectrum:
            log_dict[f"geometry/embed_alpha_{k}"] = e_spectrum[w_key]

    # --- Per-sample deviation cost (Theorems 2 & 3) ---
    pf_pred, pf_ce, pf_obs, pf_h = (
        psl_full["predicted"], psl_full["ce_loss"],
        psl_full["observed"], psl_full["h_norm_sq"],
    )
    log_dict["geometry/d_L_full_mean"] = float(pf_pred.mean())
    log_dict["geometry/H_norm_sq_full_mean"] = float(pf_h.mean())
    log_dict["geometry/ce_full_mean"] = float(pf_ce.mean())
    # Algebraic identity check (should be ≈ 0 always, sanity for the formula)
    log_dict["geometry/theorem2_residual_max"] = float(np.abs(pf_pred - pf_obs).max())
    # Pearson R² — does deviation alone predict full per-sample CE?
    if pf_pred.std() > 1e-12 and pf_ce.std() > 1e-12:
        r = float(np.corrcoef(pf_pred, pf_ce)[0, 1])
    else:
        r = 0.0
    log_dict["geometry/R2_predicted_vs_ce"] = r * r

    if psl_train is not None:
        pt_pred = psl_train["predicted"]
        pt_h = psl_train["h_norm_sq"]
        pt_ce = psl_train["ce_loss"]
        log_dict["geometry/d_L_train_mean"] = float(pt_pred.mean())
        log_dict["geometry/H_norm_sq_train_mean"] = float(pt_h.mean())
        log_dict["geometry/ce_train_mean"] = float(pt_ce.mean())
        log_dict["geometry/d_L_gap"] = float(pf_pred.mean() - pt_pred.mean())

    # Scatter table for Figure 2 (logged at every "expensive" checkpoint to keep W&B small).
    do_scatter = run_cfg.margins_every_n_evals > 0 and (
        eval_counter == 0 or eval_counter % run_cfg.margins_every_n_evals == 0
    )
    if do_scatter:
        sub = np.random.default_rng(0).choice(
            len(pf_pred), size=min(2000, len(pf_pred)), replace=False,
        )
        scatter = wandb.Table(columns=["predicted", "ce_loss", "observed", "h_norm_sq"])
        for j in sub:
            scatter.add_data(
                float(pf_pred[j]), float(pf_ce[j]),
                float(pf_obs[j]), float(pf_h[j]),
            )
        log_dict["geometry/loss_scatter"] = scatter

    # Margins and error localization (computed every N eval checkpoints — expensive)
    do_margins = run_cfg.margins_every_n_evals > 0 and (
        eval_counter == 0 or eval_counter % run_cfg.margins_every_n_evals == 0
    )
    if do_margins:
        mg = empirical_margins(model, full_loader, run_cfg.p, device, perm,
                               max_samples=run_cfg.margin_samples)
        log_dict["geometry/margin_min"] = mg["margin_min"]
        log_dict["geometry/margin_mean"] = mg["margin_mean"]
        log_dict["geometry/margin_std"] = mg["margin_std"]

        # --- Logit Fourier decomposition: the primary canonical geometry measurement ---
        # fourier_amplitudes[k] = |DFT(gamma_curve)[k]| for k=0..p//2
        # These are the alpha_k from Equation (7) in the paper, read directly from logits.
        # The active irrep set K = {k : amp[k] > threshold * amp.max()}
        fft_amps = mg["fourier_amplitudes"]  # shape (p//2 + 1,)
        fft_amps_nontrivial = fft_amps[1:]   # drop k=0 (DC / trivial mode)
        amp_max = float(fft_amps_nontrivial.max()) if len(fft_amps_nontrivial) > 0 else 1.0
        threshold = 0.1  # mode is "active" if amplitude > 10% of dominant mode
        active_ks = [k + 1 for k, a in enumerate(fft_amps_nontrivial)
                     if float(a) > threshold * amp_max]
        dominant_k_logit = int(np.argmax(fft_amps_nontrivial)) + 1

        log_dict["geometry/logit_n_active_irreps"] = len(active_ks)
        log_dict["geometry/logit_dominant_k"] = dominant_k_logit
        log_dict["geometry/logit_amp_max"] = amp_max

        # Log every non-trivial mode (dominant k can be > 20 when p is large, e.g. p=113).
        n_log_modes = len(fft_amps_nontrivial)
        for k_idx in range(n_log_modes):
            log_dict[f"geometry/logit_amp_k{k_idx + 1}"] = float(fft_amps_nontrivial[k_idx])

        # Active irreps as a W&B Table (one row per mode, flags active/inactive)
        irrep_table = wandb.Table(columns=["k", "fourier_amplitude", "normalised_amp", "active"])
        for k_idx in range(n_log_modes):
            amp = float(fft_amps_nontrivial[k_idx])
            irrep_table.add_data(k_idx + 1, amp, amp / max(amp_max, 1e-8),
                                 int(amp > threshold * amp_max))
        log_dict["geometry/logit_irrep_spectrum"] = irrep_table

        # --- Theoretical K_min from the paper's formula (Equation 8) ---
        # Use the dominant logit amplitude as proxy for alpha.
        # Normalise by p so alpha is per-input (DFT amplitude / p ≈ per-sample contribution).
        alpha_estimated = amp_max / max(run_cfg.p, 1)
        # sigma estimated from margin std (proxy for logit noise floor)
        sigma_estimated = float(mg["margin_std"]) * 0.5

        # K_min for loss target epsilon=0.01 (near-perfect CE)
        k_min_result = compute_K_min(
            p=run_cfg.p,
            alpha=alpha_estimated,
            sigma=sigma_estimated,
            epsilon_target=0.01,
        )
        log_dict["geometry/K_min_theoretical"] = k_min_result["K_min"]
        log_dict["geometry/K_min_active_ks"] = str(k_min_result["active_ks"])
        log_dict["geometry/K_gap"] = len(active_ks) - k_min_result["K_min"]
        # Energy of φ (and W) on the *empirical* greedy K from current α̂, σ̂ (same construction as K_min)
        ks_emp = tuple(int(k) for k in k_min_result["active_ks"])
        phi_k_emp = spectrum_energy_on_ks(h_spectrum, ks_emp)
        w_k_emp = spectrum_energy_on_ks(w_spectrum, ks_emp)
        log_dict["geometry/phi_energy_fraction_K_empirical"] = phi_k_emp["energy_K_nontrivial_frac"]
        log_dict["geometry/phi_epsilon_off_K_empirical"] = phi_k_emp["epsilon_off_K_nt"]
        log_dict["geometry/W_energy_fraction_K_empirical"] = w_k_emp["energy_K_nontrivial_frac"]
        log_dict["geometry/W_epsilon_off_K_empirical"] = w_k_emp["epsilon_off_K_nt"]
        log_dict["geometry/alpha_estimated"] = alpha_estimated
        log_dict["geometry/sigma_estimated"] = sigma_estimated

        # K_min table: loss and min margin at each step of greedy selection
        kmin_table = wandb.Table(columns=["n_modes", "ce_loss", "min_margin",
                                           "loss_ratio_to_baseline", "margin_over_2sigma"])
        for i, (l, mm) in enumerate(zip(
                k_min_result["loss_per_step"], k_min_result["min_margin_per_step"])):
            kmin_table.add_data(
                i,
                float(l),
                float(mm),
                float(l) / max(k_min_result["loss_baseline"], 1e-8),
                float(mm) / max(2.0 * sigma_estimated, 1e-8),
            )
        log_dict["geometry/K_min_table"] = kmin_table

        # Margin curve: empirical vs two predictions.
        # pred_curve_W  — predicted from W_L Frobenius projections (approximate, no hidden states)
        # pred_curve_logit — reconstructed from the logit DFT amplitudes (exact by definition)
        p_val = run_cfg.p
        deltas = np.arange(1, p_val, dtype=np.float32)
        pred_curve_logit = sum(
            float(fft_amps_nontrivial[k_idx]) * (1.0 - np.cos(2.0 * np.pi * (k_idx + 1) * deltas / p_val))
            for k_idx in range(len(fft_amps_nontrivial))
        )
        margin_table = wandb.Table(
            columns=["delta", "gamma_empirical", "gamma_predicted_W", "gamma_predicted_logit"])
        for d in range(1, p_val):
            margin_table.add_data(
                d,
                float(mg["margin_curve"][d - 1]),
                float(pred_curve[d - 1]),
                float(pred_curve_logit[d - 1]),
            )
        log_dict["geometry/margin_curve"] = margin_table

        # Error localization: per-Δ error rate vs exp(−γ_Δ)
        error_table = _error_localization_table(model, full_loader, mg["margin_curve"], run_cfg.p,
                                                device, perm, max_samples=run_cfg.margin_samples)
        log_dict["geometry/error_localization"] = error_table

    # Projection intervention: project W onto top-K canonical modes, re-evaluate without retraining.
    # We test K=1 (minimal), K=3, and K=all-non-trivial to show how much each recovers.
    if run_cfg.do_projection_test:
        p_val = run_cfg.p
        # Rank all non-trivial modes by energy
        mode_energies = [(k, spectrum.get(f"alpha_{k}", 0.0)) for k in range(1, (p_val - 1) // 2 + 1)]
        mode_energies.sort(key=lambda t: t[1], reverse=True)
        top_ks_ranked = [k for k, _ in mode_energies]

        with torch.no_grad():
            w_orig = w.clone()
            for n_proj in (1, 3):
                ks_to_use = tuple(top_ks_ranked[:n_proj])
                w_proj = project_onto_canonical(w_orig, p_val, ks=ks_to_use)
                model.transformer.output.weight.copy_(w_proj)
                proj_val_list = do_eval_step(
                    model=model, val_dataloader=val_loader, train_cfg=train_cfg_dict,
                    device=device, use_tqdm_for_eval_step=False,
                    verbosity=Verbosity.QUIET, logger=log,
                )
                proj_acc = float(combine_logs(proj_val_list)["accuracy"])
                log_dict[f"geometry/proj_val_acc_top{n_proj}"] = proj_acc
                log_dict[f"geometry/proj_acc_gain_top{n_proj}"] = proj_acc - val_acc
            # Also project onto all non-trivial modes (removes only DC)
            w_proj_all = project_onto_canonical(w_orig, p_val, ks=None)
            model.transformer.output.weight.copy_(w_proj_all)
            proj_val_list = do_eval_step(
                model=model, val_dataloader=val_loader, train_cfg=train_cfg_dict,
                device=device, use_tqdm_for_eval_step=False,
                verbosity=Verbosity.QUIET, logger=log,
            )
            log_dict["geometry/proj_val_acc_all"] = float(combine_logs(proj_val_list)["accuracy"])
            model.transformer.output.weight.copy_(w_orig)  # always restore

    if wandb_run is not None:
        wandb_run.log(log_dict, step=step)

    return val_acc, float(phi_k_theory["epsilon_off_K_nt"])


def _error_localization_table(  # noqa: PLR0913
    model: GrokkModel,
    full_loader: DataLoader,
    margin_curve: np.ndarray,
    p: int,
    device: torch.device,
    perm: torch.Tensor,
    max_samples: int = 512,
) -> wandb.Table:
    """Per-Δ error rate vs predicted margin — tests error localization claim.

    Samples max_samples examples from full_loader (finite).  Vectorised delta
    matching: no inner Python loop over Δ.
    """
    was_training = model.training
    model.eval()
    perm_cpu = perm.cpu()
    error_by_delta = torch.zeros(p - 1, dtype=torch.long)
    total = 0

    with torch.no_grad():
        for x, y in full_loader:
            if total >= max_samples:
                break
            x_dev = x.to(device)
            logits, _, _ = model(x_dev)
            pred = logits[:, -1, :].argmax(dim=-1).cpu()
            y_cpu = y.cpu()
            wrong_mask = pred != y_cpu
            if wrong_mask.any():
                wrong_pred = pred[wrong_mask]
                wrong_y = y_cpu[wrong_mask]
                matches = perm_cpu[:, wrong_y] == wrong_pred.unsqueeze(0)  # (p-1, n_wrong)
                error_by_delta += matches.sum(dim=1)
            total += len(y_cpu)

    if was_training:
        model.train()

    error_rate = error_by_delta.float().numpy() / max(total, 1)
    table = wandb.Table(columns=["delta", "error_rate", "gamma_delta", "exp_neg_gamma"])
    for delta in range(1, p):
        gamma = float(margin_curve[delta - 1])
        table.add_data(delta, float(error_rate[delta - 1]), gamma, float(np.exp(-max(gamma, -500.0))))
    return table


# ---------------------------------------------------------------------------
# Core shared training loop
# ---------------------------------------------------------------------------


def _train_loop(  # noqa: PLR0913
    model: GrokkModel,
    optimizer: AdamW,
    lr_schedule: Any,
    loaders: dict[str, Any],
    device: torch.device,
    run_cfg: _RunConfig,
    perm: torch.Tensor,
) -> dict[str, Any]:
    """Run the training loop for one condition, logging geometry to the active W&B run."""
    train_loader: DataLoader = loaders["train"]
    val_loader: DataLoader = loaders["val"]
    full_loader: DataLoader = loaders["full"]
    wandb_run = wandb.run

    train_iter = iter(train_loader)
    grokking_step = -1
    eval_counter = 0
    last_train_logs: dict = {}
    last_reg_val = 0.0

    prog = tqdm(range(run_cfg.max_steps), desc="training", leave=False)
    for step in prog:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        if run_cfg.lambda_canonical > 0.0 or run_cfg.lambda_canonical_H > 0.0:
            last_train_logs, last_reg_val = _training_step_with_reg(
                model, optimizer, lr_schedule, x, y, device, run_cfg,
            )
        else:
            last_train_logs = do_training_step(
                model=model, optimizer=optimizer, clip_grad_norm_max_norm=1.0,
                lr_schedule=lr_schedule, x=x, y=y, device=device,
            )
            last_reg_val = 0.0

        if (step + 1) % run_cfg.eval_every == 0 or step == 0:
            val_acc, epsilon = _log_geometry(
                model=model, val_loader=val_loader, full_loader=full_loader,
                perm=perm, device=device, run_cfg=run_cfg, step=step + 1,
                train_logs=last_train_logs, reg_val=last_reg_val,
                eval_counter=eval_counter, wandb_run=wandb_run,
                train_loader=train_loader,
            )
            if val_acc >= 0.95 and grokking_step < 0:
                grokking_step = step + 1
            eval_counter += 1
            prog.set_postfix(val_acc=f"{val_acc:.3f}", eps=f"{epsilon:.3f}")

    # Final forced geometry eval
    val_acc, epsilon = _log_geometry(
        model=model, val_loader=val_loader, full_loader=full_loader,
        perm=perm, device=device, run_cfg=run_cfg, step=run_cfg.max_steps,
        train_logs=last_train_logs, reg_val=last_reg_val,
        eval_counter=eval_counter, wandb_run=wandb_run,
        train_loader=train_loader,
    )
    if val_acc >= 0.95 and grokking_step < 0:
        grokking_step = run_cfg.max_steps

    if wandb_run is not None:
        wandb_run.summary["grokking_step"] = grokking_step
        wandb_run.summary["final_val_accuracy"] = val_acc
        wandb_run.summary["final_epsilon"] = epsilon

    return {"grokking_step": grokking_step, "final_val_acc": val_acc, "final_epsilon": epsilon}


# ---------------------------------------------------------------------------
# W&B init helper
# ---------------------------------------------------------------------------


def _wandb_init(wandb_cfg: _WandbConfig, run_cfg: _RunConfig, extra_config: dict | None = None) -> None:
    config = dataclasses.asdict(run_cfg)
    if extra_config:
        config.update(extra_config)
    wandb_dir = pathlib.Path(GROKKING_REPOSITORY_BASE_PATH) / "wandb_dir" / wandb_cfg.project
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        name=wandb_cfg.name or None,
        mode=wandb_cfg.mode or os.environ.get("WANDB_MODE"),
        config=config,
        dir=str(wandb_dir),
    )


# ---------------------------------------------------------------------------
# Experiment A/B/C  —  geometry_phase_transition
# ---------------------------------------------------------------------------


def run_geometry_phase_transition(args: argparse.Namespace) -> None:
    """Single long training run (experiments A, B, C combined).

    Logs per-irrep spectrum, ε, empirical margins, error localisation,
    and projection intervention at every eval checkpoint.

    Covers paper claims C6 (representational deviation), C7 (phase transition),
    C3–C5 (margin decomposition), and the "remove the noise" test.
    """
    run_cfg = _RunConfig(
        p=args.p, frac_train=args.frac_train, seed=args.seed,
        max_steps=args.max_steps, eval_every=args.eval_every,
        eval_batches=args.eval_batches, bsize=args.bsize,
        lr=args.lr, weight_decay=args.weight_decay,
        margins_every_n_evals=args.margins_every_n_evals,
        margin_samples=args.margin_samples,
        do_projection_test=True,
    )
    model_cfg = _ModelConfig(
        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
        attn_dim=args.attn_dim, intermediate_dim=args.intermediate_dim,
    )
    wandb_cfg = _WandbConfig(
        project=args.wandb_project, group="geometry_phase_transition",
        name=f"p{args.p}_frac{args.frac_train}_wd{args.weight_decay}_seed{args.seed}",
        mode=args.wandb_mode,
    )
    _wandb_init(wandb_cfg, run_cfg)
    set_seed(seed=args.seed, logger=log)
    device = get_torch_device(preferred_torch_backend=PreferredTorchBackend.AUTO,
                              verbosity=Verbosity.QUIET, logger=log, cuda_device_id=None)

    dataset = _load_dataset(args.p, args.frac_train, args.seed)
    train_loader, val_loader = _make_loaders(dataset, args.bsize)
    full_loader, _, perm = _make_full_loader(dataset, args.bsize)
    model = _build_model(model_cfg, dataset, device)
    optimizer, lr_schedule = _build_optimizer(model, run_cfg)

    _train_loop(model, optimizer, lr_schedule,
                {"train": train_loader, "val": val_loader, "full": full_loader},
                device, run_cfg, perm)
    wandb.finish()


# ---------------------------------------------------------------------------
# Experiment D  —  noise_robustness
# ---------------------------------------------------------------------------


def _eval_with_logit_noise(
    model: GrokkModel,
    val_loader: DataLoader,
    device: torch.device,
    sigma: float,
    n_trials: int = 5,
) -> float:
    """Evaluate val accuracy with i.i.d. Gaussian noise N(0,σ²) added to logits.

    Averages over n_trials noise realisations for stability.
    """
    was_training = model.training
    model.eval()
    total_acc = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            correct = 0
            total = 0
            for x, y in val_loader:
                x_dev, y_dev = x.to(device), y.to(device)
                logits, _, _ = model(x_dev)
                noisy = logits[:, -1, :] + sigma * torch.randn_like(logits[:, -1, :])
                correct += (noisy.argmax(dim=-1) == y_dev).sum().item()
                total += len(y_dev)
            total_acc += correct / max(total, 1)
    if was_training:
        model.train()
    return total_acc / n_trials


def run_noise_robustness(args: argparse.Namespace) -> None:
    """Train until grokked, then sweep Gaussian noise σ on logits.

    Tests the 2σ degradation bound from Section 3.1 (claim C5):
    val accuracy should start degrading when σ > γ_min / 2.
    Paper line: "test with perturbations of the spaces".
    """
    run_cfg = _RunConfig(
        p=args.p, frac_train=args.frac_train, seed=args.seed,
        max_steps=args.max_steps, eval_every=args.eval_every,
        eval_batches=args.eval_batches, bsize=args.bsize,
        lr=args.lr, weight_decay=args.weight_decay,
        margins_every_n_evals=args.margins_every_n_evals,
        margin_samples=args.margin_samples,
        do_projection_test=False,
    )
    model_cfg = _ModelConfig(
        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
        attn_dim=args.attn_dim, intermediate_dim=args.intermediate_dim,
    )
    wandb_cfg = _WandbConfig(
        project=args.wandb_project, group="noise_robustness",
        name=f"p{args.p}_frac{args.frac_train}_seed{args.seed}",
        mode=args.wandb_mode,
    )
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    _wandb_init(wandb_cfg, run_cfg, {"sigmas": sigmas})
    set_seed(seed=args.seed, logger=log)
    device = get_torch_device(preferred_torch_backend=PreferredTorchBackend.AUTO,
                              verbosity=Verbosity.QUIET, logger=log, cuda_device_id=None)

    dataset = _load_dataset(args.p, args.frac_train, args.seed)
    train_loader, val_loader = _make_loaders(dataset, args.bsize)
    full_loader, _, perm = _make_full_loader(dataset, args.bsize)
    model = _build_model(model_cfg, dataset, device)
    optimizer, lr_schedule = _build_optimizer(model, run_cfg)

    # Phase 1: train until grokked (or max_steps)
    log.info("Phase 1: training to convergence …")
    summary = _train_loop(model, optimizer, lr_schedule,
                          {"train": train_loader, "val": val_loader, "full": full_loader},
                          device, run_cfg, perm)
    log.info("Training done. grokking_step=%s  val_acc=%.3f",
             summary["grokking_step"], summary["final_val_acc"])

    # Phase 2: noise sweep on the trained model
    log.info("Phase 2: noise robustness sweep …")
    # Measure margin_min without noise (for the theoretical 2σ threshold)
    mg = empirical_margins(model, full_loader, args.p, device, perm)
    gamma_min = mg["margin_min"]
    theoretical_sigma_threshold = gamma_min / 2.0
    wandb.run.summary["gamma_min"] = gamma_min
    wandb.run.summary["theoretical_sigma_threshold"] = theoretical_sigma_threshold

    noise_table = wandb.Table(columns=["sigma", "val_accuracy", "sigma_over_threshold"])
    for sigma in sigmas:
        acc = _eval_with_logit_noise(model, val_loader, device, sigma)
        noise_table.add_data(sigma, acc, sigma / max(theoretical_sigma_threshold, 1e-8))
        wandb.log({"noise/sigma": sigma, "noise/val_accuracy": acc,
                   "noise/sigma_over_threshold": sigma / max(theoretical_sigma_threshold, 1e-8)})

    wandb.log({"noise/robustness_table": noise_table})
    wandb.finish()


# ---------------------------------------------------------------------------
# Experiment E  —  data_threshold
# ---------------------------------------------------------------------------


def run_data_threshold(args: argparse.Namespace) -> None:
    """Sweep frac_train to find the critical data fraction n*.

    Below n*, ε never collapses and the model stays in the memorisation regime.
    Tests Section 3.3 claim C9.
    """
    fracs = [float(f) for f in args.fracs.split(",") if f.strip()]
    for frac in fracs:
        run_cfg = _RunConfig(
            p=args.p, frac_train=frac, seed=args.seed,
            max_steps=args.max_steps, eval_every=args.eval_every,
            eval_batches=args.eval_batches, bsize=args.bsize,
            lr=args.lr, weight_decay=args.weight_decay,
            margins_every_n_evals=args.margins_every_n_evals,
        margin_samples=args.margin_samples,
            do_projection_test=True,
        )
        model_cfg = _ModelConfig(
            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
            attn_dim=args.attn_dim, intermediate_dim=args.intermediate_dim,
        )
        wandb_cfg = _WandbConfig(
            project=args.wandb_project, group="data_threshold",
            name=f"p{args.p}_frac{frac:.3f}_seed{args.seed}",
            mode=args.wandb_mode,
        )
        _wandb_init(wandb_cfg, run_cfg)
        set_seed(seed=args.seed, logger=log)
        device = get_torch_device(preferred_torch_backend=PreferredTorchBackend.AUTO,
                                  verbosity=Verbosity.QUIET, logger=log, cuda_device_id=None)

        dataset = _load_dataset(args.p, frac, args.seed)
        train_loader, val_loader = _make_loaders(dataset, args.bsize)
        full_loader, _, perm = _make_full_loader(dataset, args.bsize)
        model = _build_model(model_cfg, dataset, device)
        optimizer, lr_schedule = _build_optimizer(model, run_cfg)

        _train_loop(model, optimizer, lr_schedule,
                    {"train": train_loader, "val": val_loader, "full": full_loader},
                    device, run_cfg, perm)
        wandb.finish()


# ---------------------------------------------------------------------------
# Experiment F  —  weight_decay_geometry
# ---------------------------------------------------------------------------


def run_weight_decay_geometry(args: argparse.Namespace) -> None:
    """Sweep weight_decay, tracking ε and time-to-grokking.

    Tests claim C8: weight decay is the mechanism driving ε→0.
    Maps to Section 3.3 and Section 4.4 "Regularizer".
    """
    wds = [float(w) for w in args.weight_decays.split(",") if w.strip()]
    for wd in wds:
        run_cfg = _RunConfig(
            p=args.p, frac_train=args.frac_train, seed=args.seed,
            max_steps=args.max_steps, eval_every=args.eval_every,
            eval_batches=args.eval_batches, bsize=args.bsize,
            lr=args.lr, weight_decay=wd,
            margins_every_n_evals=args.margins_every_n_evals,
        margin_samples=args.margin_samples,
            do_projection_test=False,
        )
        model_cfg = _ModelConfig(
            hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
            attn_dim=args.attn_dim, intermediate_dim=args.intermediate_dim,
        )
        wandb_cfg = _WandbConfig(
            project=args.wandb_project, group="weight_decay_geometry",
            name=f"p{args.p}_frac{args.frac_train}_wd{wd}_seed{args.seed}",
            mode=args.wandb_mode,
        )
        _wandb_init(wandb_cfg, run_cfg)
        set_seed(seed=args.seed, logger=log)
        device = get_torch_device(preferred_torch_backend=PreferredTorchBackend.AUTO,
                                  verbosity=Verbosity.QUIET, logger=log, cuda_device_id=None)

        dataset = _load_dataset(args.p, args.frac_train, args.seed)
        train_loader, val_loader = _make_loaders(dataset, args.bsize)
        full_loader, _, perm = _make_full_loader(dataset, args.bsize)
        model = _build_model(model_cfg, dataset, device)
        optimizer, lr_schedule = _build_optimizer(model, run_cfg)

        _train_loop(model, optimizer, lr_schedule,
                    {"train": train_loader, "val": val_loader, "full": full_loader},
                    device, run_cfg, perm)
        wandb.finish()


# ---------------------------------------------------------------------------
# Experiment G  —  canonical_regulariser
# ---------------------------------------------------------------------------


def run_canonical_regularizer(args: argparse.Namespace) -> None:
    """Sweep canonical regularisers (weight-side λ_W and/or logit-side λ_H = β).

    Two regularisers are supported in parallel:
        L_total = L_CE
                  + λ_W * ‖Π_{(V_ks)⊥} W_L‖²_F        (--lambdas)
                  + λ_H * mean_i ‖Π_{(V_ks)⊥} z_i‖²    (--lambdas-h, paper's β‖H‖²)

    Each (frac, λ_W, λ_H) triple becomes one W&B run.  Setting both lists to "0"
    runs the unregularised baseline only.  Run at one frac above n* and one below
    to test whether either regulariser induces grokking below the data threshold.
    """
    lambdas = [float(lam) for lam in args.lambdas.split(",") if lam.strip()]
    lambdas_h = [float(lam) for lam in args.lambdas_h.split(",") if lam.strip()]
    if not lambdas_h:
        lambdas_h = [0.0]
    fracs = [float(f) for f in args.fracs.split(",") if f.strip()]
    canonical_ks = tuple(int(k) for k in args.canonical_ks.split(",") if k.strip())

    for frac in fracs:
        for lam_w in lambdas:
            for lam_h in lambdas_h:
                run_cfg = _RunConfig(
                    p=args.p, frac_train=frac, seed=args.seed,
                    max_steps=args.max_steps, eval_every=args.eval_every,
                    eval_batches=args.eval_batches, bsize=args.bsize,
                    lr=args.lr, weight_decay=0.0,  # WD=0 to isolate regulariser effect
                    margins_every_n_evals=args.margins_every_n_evals,
                    margin_samples=args.margin_samples,
                    do_projection_test=False,
                    lambda_canonical=lam_w,
                    lambda_canonical_H=lam_h,
                    canonical_ks=canonical_ks,
                )
                model_cfg = _ModelConfig(
                    hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
                    attn_dim=args.attn_dim, intermediate_dim=args.intermediate_dim,
                )
                ks_str = "-".join(str(k) for k in canonical_ks)
                wandb_cfg = _WandbConfig(
                    project=args.wandb_project, group="canonical_regularizer",
                    name=(
                        f"p{args.p}_frac{frac:.3f}_lamW{lam_w}_lamH{lam_h}"
                        f"_ks{ks_str}_seed{args.seed}"
                    ),
                    mode=args.wandb_mode,
                )
                _wandb_init(wandb_cfg, run_cfg, {"canonical_ks": list(canonical_ks)})
                set_seed(seed=args.seed, logger=log)
                device = get_torch_device(
                    preferred_torch_backend=PreferredTorchBackend.AUTO,
                    verbosity=Verbosity.QUIET, logger=log, cuda_device_id=None,
                )

                dataset = _load_dataset(args.p, frac, args.seed)
                train_loader, val_loader = _make_loaders(dataset, args.bsize)
                full_loader, _, perm = _make_full_loader(dataset, args.bsize)
                model = _build_model(model_cfg, dataset, device)
                optimizer, lr_schedule = _build_optimizer(model, run_cfg)

                _train_loop(
                    model, optimizer, lr_schedule,
                    {"train": train_loader, "val": val_loader, "full": full_loader},
                    device, run_cfg, perm,
                )
                wandb.finish()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Canonical geometry experiments for grokking (7 experiments).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment",
        choices=[
            "geometry_phase_transition",
            "noise_robustness",
            "data_threshold",
            "weight_decay_geometry",
            "canonical_regularizer",
        ],
        default="geometry_phase_transition",
        help="Which experiment to run.",
    )
    # W&B
    p.add_argument(
        "--wandb-project",
        default="canonical_repr_grokking",
        help="W&B project name (create it once in the UI or it is created on first log).",
    )
    p.add_argument("--wandb-mode", default=None, help="e.g. offline, disabled")
    # Task
    p.add_argument("--p", type=int, default=113, help="Prime modulus for Z_p.")
    p.add_argument("--seed", type=int, default=42)
    # Architecture
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--attn-dim", type=int, default=32)
    p.add_argument("--intermediate-dim", type=int, default=512)
    # Training
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--bsize", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="Weight decay (used as default for experiments A/D/E).")
    p.add_argument("--frac-train", type=float, default=0.3,
                   help="Training fraction (used as default for single-run experiments).")
    p.add_argument("--margins-every-n-evals", type=int, default=5,
                   help="Compute margin curve every N eval steps (0=only first+last).")
    p.add_argument("--margin-samples", type=int, default=512,
                   help="Number of examples to sample for margin/error-localization estimates.")
    # Sweep arguments
    p.add_argument("--fracs", type=str, default="0.1,0.15,0.2,0.25,0.3,0.4,0.5",
                   help="Comma-separated frac_train values (experiments E and G).")
    p.add_argument("--weight-decays", type=str, default="0,0.0001,0.001,0.01,0.1,1.0",
                   help="Comma-separated weight decay values (experiment F).")
    p.add_argument("--sigmas", type=str, default="0,0.5,1,2,5,10,20",
                   help="Comma-separated noise σ values (experiment D).")
    p.add_argument(
        "--lambdas",
        type=str,
        default="0,0.001,0.01,0.1,1.0,10.0,30.0,100.0",
        help="Comma-separated λ_W values (weight-side ‖Π_⊥ W_L‖² penalty), experiment G.",
    )
    p.add_argument(
        "--lambdas-h",
        type=str,
        default="0",
        help="Comma-separated λ_H = β values (logit-side ‖H‖² penalty, paper Sec. 4.5), experiment G.",
    )
    p.add_argument("--canonical-ks", type=str, default="1",
                   help="Comma-separated Fourier mode indices for V* in experiment G.")
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args()

    dispatch = {
        "geometry_phase_transition": run_geometry_phase_transition,
        "noise_robustness": run_noise_robustness,
        "data_threshold": run_data_threshold,
        "weight_decay_geometry": run_weight_decay_geometry,
        "canonical_regularizer": run_canonical_regularizer,
    }
    dispatch[args.experiment](args)


if __name__ == "__main__":
    main()
