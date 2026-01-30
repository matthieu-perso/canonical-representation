# How to reconstruct and investigate the geometry

Concrete steps to get geometry data and look for geometric changes (e.g. at grokking).

---

## 1. Where the data comes from

- **During training:** Geometry is computed every `compute_geometry_every` steps (e.g. 500) and logged to **wandb** (`geometry/curvature_proxy`, `geometry/fisher_max_eig`, etc.) and to the **console**.
- **After a run:** You can **recompute** geometry from saved checkpoints and write a single CSV. That gives you **path length** (segment + cumulative) as well, which is only available offline.

---

## 2. Reconstruct geometry from a run (offline)

You need a **run directory** that contains checkpoints, e.g.:

`outputs/run/train_grokk/2026-01-30/12-00-00.123456/`  
or, for multirun, one of the sweep subdirs:

`outputs/multirun/train_grokk/2026-01-29/17-31-47.080439/dataset.frac_train=0.4,.../seed=1234/`

**Step 1 – Produce the geometry CSV**

From the repo root:

```bash
cd /Users/matthieu/feature_grokking/grokking-via-lid

# RUN_DIR = path to one run (must contain checkpoints/step=500, step=1000, ...)
uv run python grokking/scripts/run_geometry_analysis.py RUN_DIR -o RUN_DIR/geometry_metrics.csv --step-interval 500
```

- **RUN_DIR:** Replace with the full path to one run directory (e.g. `outputs/run/train_grokk/2026-01-30/12-00-00.123456`).
- This loads each checkpoint at 500, 1000, 1500, …, computes Fisher, curvature proxy, sharpness, representation spectrum, and **path length** between consecutive checkpoints, and writes **geometry_metrics.csv** inside that run dir.

**Step 2 – Plot and inspect**

```bash
uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR/geometry_metrics.csv -o RUN_DIR/geometry_plots/
```

- Reads **geometry_metrics.csv**, plots curvature_proxy, fisher_max_eig, sharpness_hessian_trace, cumulative_natural_path_length, and representation_spectrum vs **step**, and saves PDFs in `RUN_DIR/geometry_plots/`.

**Optional – Mark the grokking step**

If you know the step \(t^*\) where val accuracy crossed 0.9 (or your threshold), pass it so a vertical line is drawn:

```bash
uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR/geometry_metrics.csv -o RUN_DIR/geometry_plots/ --grokking-step 120000
```

**One-shot: run dir → CSV + plots**

If you pass the **run dir** and `--from-run-dir`, the script will run the geometry analysis (if the CSV is missing) then plot:

```bash
uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR --from-run-dir --step-interval 500 -o RUN_DIR/geometry_plots/
```

---

## 3. What to look for (investigation)

1. **Open the PDFs** in `RUN_DIR/geometry_plots/`:
   - **geometry_parameter_space.pdf:** curvature_proxy, fisher_max_eig, sharpness vs step.
   - **geometry_path_length.pdf:** cumulative natural path length vs step.
   - **geometry_representation_spectrum.pdf:** top representation eigenvalues vs step.

2. **Compare with generalization:** Get **val accuracy vs step** from wandb (or from your training log). Find the step \(t^*\) where val accuracy crosses 0.9 (grokking). Re-run the plot script with `--grokking-step t^*` so the red vertical line marks that step.

3. **Interpret:**
   - **Slope or level change** in curvature_proxy, fisher_max_eig, or sharpness **near \(t^*\)** → geometry of the parameter manifold changes around the transition.
   - **Inflection** in cumulative path length near \(t^*\) → training path “speeds up” or “slows down” in information distance.
   - **Change** in representation_spectrum near \(t^*\) → representation-space geometry (shape of hidden states) changes at grokking.

4. **Use wandb for many runs:** If you have many runs (e.g. multirun over frac_train), use wandb’s charts to plot `geometry/curvature_proxy`, `geometry/sharpness_hessian_trace`, etc. vs step and group by run. The offline CSV + script above is for **one run** at a time.

---

## 4. Summary

| Goal | Command / where |
|------|------------------|
| Reconstruct geometry (CSV) from one run | `uv run python grokking/scripts/run_geometry_analysis.py RUN_DIR -o RUN_DIR/geometry_metrics.csv` |
| Plot geometry vs step | `uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR/geometry_metrics.csv -o RUN_DIR/geometry_plots/` |
| Mark grokking step on plots | Add `--grokking-step 120000` (replace with your step) |
| Do both (CSV + plots) from run dir | `uv run python grokking/scripts/plot_geometry_investigation.py RUN_DIR --from-run-dir -o RUN_DIR/geometry_plots/` |
| Inspect many runs | Use wandb (geometry/* metrics vs step, grouped by run) |

**RUN_DIR** = path to a single run directory that contains `checkpoints/step=500`, `step=1000`, etc. (e.g. one subdir from `outputs/run/` or one sweep subdir from `outputs/multirun/`).
