"""
Visualize all training metrics from RAFI++ training history.

Usage:
    python visualize_metrics.py
    python visualize_metrics.py --history ./logs/rafipp_run/history.json --out_dir ./outputs/metrics_plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ──────────────────────────── colour palette ────────────────────────────
STAGE_COLORS = {1: "#6366f1", 2: "#f59e0b", 3: "#10b981"}
STAGE_BG = {1: "#eef2ff", 2: "#fffbeb", 3: "#ecfdf5"}
LINE_PALETTE = [
    "#6366f1", "#ef4444", "#f59e0b", "#10b981",
    "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6",
    "#f97316", "#64748b", "#a855f7", "#06b6d4",
]


# ──────────────────────────── helpers ───────────────────────────────────
def load_history(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stage_boundaries(history: list) -> list:
    """Return list of (start_epoch, end_epoch, stage)."""
    if not history:
        return []
    bounds = []
    cur_stage = history[0]["stage"]
    start = history[0]["epoch"]
    for rec in history:
        if rec["stage"] != cur_stage:
            bounds.append((start, rec["epoch"] - 1, cur_stage))
            cur_stage = rec["stage"]
            start = rec["epoch"]
    bounds.append((start, history[-1]["epoch"], cur_stage))
    return bounds


def shade_stages(ax, bounds: list, epochs: list):
    """Add coloured background bands for each training stage."""
    for start, end, stage in bounds:
        ax.axvspan(
            start - 0.5, end + 0.5,
            color=STAGE_BG.get(stage, "#f8fafc"),
            alpha=0.55, zorder=0,
        )
    # stage labels
    for start, end, stage in bounds:
        mid = (start + end) / 2
        ax.text(
            mid, 1.02, f"Stage {stage}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=STAGE_COLORS.get(stage, "#64748b"),
        )


def apply_style(ax, title: str, xlabel: str = "Epoch", ylabel: str = "Value"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────── plot functions ─────────────────────────────
def plot_segmentation_losses(history, bounds, out_dir):
    """Plot mask / boundary / confidence / total seg loss."""
    keys = ["loss_mask", "loss_boundary", "loss_conf", "loss_seg"]
    data = defaultdict(list)
    epochs = []
    for rec in history:
        epochs.append(rec["epoch"])
        for k in keys:
            data[k].append(rec["train"].get(k, float("nan")))

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds, epochs)
    for i, k in enumerate(keys):
        vals = data[k]
        if all(np.isnan(v) for v in vals):
            continue
        lw = 2.2 if k == "loss_seg" else 1.5
        ls = "-" if k == "loss_seg" else "--"
        ax.plot(epochs, vals, color=LINE_PALETTE[i], linewidth=lw,
                linestyle=ls, marker="o", markersize=3, label=k)
    apply_style(ax, "Segmentation Losses", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(out_dir / "segmentation_losses.png", dpi=180)
    plt.close(fig)


def plot_reconstruction_losses(history, bounds, out_dir):
    """Plot reconstruction-related losses (rec, ssim, perc, style, id, edge)."""
    keys = ["loss_rec", "loss_ssim", "loss_perc", "loss_style", "loss_id", "loss_edge"]
    data = defaultdict(list)
    epochs = []
    for rec in history:
        epochs.append(rec["epoch"])
        for k in keys:
            data[k].append(rec["train"].get(k, float("nan")))

    has_data = any(not all(np.isnan(v) for v in data[k]) for k in keys)
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds, epochs)
    for i, k in enumerate(keys):
        vals = data[k]
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(epochs, vals, color=LINE_PALETTE[i], linewidth=1.5,
                marker="o", markersize=3, label=k)
    apply_style(ax, "Reconstruction Losses", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_losses.png", dpi=180)
    plt.close(fig)


def plot_adversarial_losses(history, bounds, out_dir):
    """Plot adversarial losses (adv, dp, df, d_total)."""
    keys = ["loss_adv", "loss_dp", "loss_df", "loss_d_total"]
    data = defaultdict(list)
    epochs = []
    for rec in history:
        epochs.append(rec["epoch"])
        for k in keys:
            data[k].append(rec["train"].get(k, float("nan")))

    has_data = any(not all(np.isnan(v) for v in data[k]) for k in keys)
    if not has_data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds, epochs)
    for i, k in enumerate(keys):
        vals = data[k]
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(epochs, vals, color=LINE_PALETTE[i + 4], linewidth=1.5,
                marker="o", markersize=3, label=k)
    apply_style(ax, "Adversarial Losses", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(out_dir / "adversarial_losses.png", dpi=180)
    plt.close(fig)


def plot_total_loss(history, bounds, out_dir):
    """Plot the overall loss_total (or loss_seg for stage-1-only epochs)."""
    epochs = []
    totals = []
    for rec in history:
        epochs.append(rec["epoch"])
        t = rec["train"]
        totals.append(t.get("loss_total", t.get("loss_seg", float("nan"))))

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds, epochs)
    ax.plot(epochs, totals, color="#ef4444", linewidth=2.2,
            marker="o", markersize=4, label="loss_total")
    apply_style(ax, "Total Loss per Epoch", ylabel="Loss")
    fig.tight_layout()
    fig.savefig(out_dir / "total_loss.png", dpi=180)
    plt.close(fig)


def plot_val_segmentation(history, bounds, out_dir):
    """Plot validation Dice & IoU."""
    epochs, dice_vals, iou_vals = [], [], []
    for rec in history:
        d = rec["val"].get("dice", None)
        io = rec["val"].get("iou", None)
        if d is not None or io is not None:
            epochs.append(rec["epoch"])
            dice_vals.append(d if d is not None else float("nan"))
            iou_vals.append(io if io is not None else float("nan"))

    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds, epochs)
    ax.plot(epochs, dice_vals, color="#6366f1", linewidth=2, marker="s",
            markersize=5, label="Dice")
    ax.plot(epochs, iou_vals, color="#10b981", linewidth=2, marker="^",
            markersize=5, label="IoU")
    ax.set_ylim(bottom=max(0, min(dice_vals + iou_vals) - 0.05), top=1.02)
    apply_style(ax, "Validation — Segmentation Metrics (Dice & IoU)", ylabel="Score")
    fig.tight_layout()
    fig.savefig(out_dir / "val_segmentation.png", dpi=180)
    plt.close(fig)


def plot_val_restoration(history, bounds, out_dir):
    """Plot validation L1, PSNR, SSIM in separate subplots."""
    epochs = []
    l1_vals, psnr_vals, ssim_vals = [], [], []
    for rec in history:
        v = rec["val"]
        if "l1" in v or "psnr" in v or "ssim" in v:
            epochs.append(rec["epoch"])
            l1_vals.append(v.get("l1", float("nan")))
            psnr_vals.append(v.get("psnr", float("nan")))
            ssim_vals.append(v.get("ssim", float("nan")))

    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    pairs = [
        (axes[0], l1_vals, "L1 ↓", "#ef4444", "L1"),
        (axes[1], psnr_vals, "PSNR (dB) ↑", "#3b82f6", "PSNR"),
        (axes[2], ssim_vals, "SSIM ↑", "#10b981", "SSIM"),
    ]
    for ax, vals, title, color, lbl in pairs:
        shade_stages(ax, bounds, epochs)
        ax.plot(epochs, vals, color=color, linewidth=2, marker="o", markersize=4, label=lbl)
        apply_style(ax, f"Validation — {title}", ylabel=lbl)

    fig.suptitle("Validation — Restoration Metrics", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / "val_restoration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_dashboard(history, bounds, out_dir):
    """Single overview figure with 6 subplots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    epochs = [r["epoch"] for r in history]

    # ── (0,0) Total loss ──
    ax = axes[0, 0]
    totals = [r["train"].get("loss_total", r["train"].get("loss_seg", float("nan"))) for r in history]
    shade_stages(ax, bounds, epochs)
    ax.plot(epochs, totals, color="#ef4444", linewidth=2, marker="o", markersize=3)
    apply_style(ax, "Total Loss")

    # ── (0,1) Segmentation losses ──
    ax = axes[0, 1]
    shade_stages(ax, bounds, epochs)
    for i, k in enumerate(["loss_mask", "loss_boundary", "loss_conf"]):
        vals = [r["train"].get(k, float("nan")) for r in history]
        if not all(np.isnan(v) for v in vals):
            ax.plot(epochs, vals, color=LINE_PALETTE[i], linewidth=1.4,
                    marker="o", markersize=2, label=k)
    apply_style(ax, "Segmentation Losses")

    # ── (0,2) Reconstruction losses ──
    ax = axes[0, 2]
    shade_stages(ax, bounds, epochs)
    for i, k in enumerate(["loss_rec", "loss_ssim", "loss_perc", "loss_style"]):
        vals = [r["train"].get(k, float("nan")) for r in history]
        if not all(np.isnan(v) for v in vals):
            ax.plot(epochs, vals, color=LINE_PALETTE[i + 3], linewidth=1.4,
                    marker="o", markersize=2, label=k)
    apply_style(ax, "Reconstruction Losses")

    # ── (1,0) Adversarial losses ──
    ax = axes[1, 0]
    shade_stages(ax, bounds, epochs)
    for i, k in enumerate(["loss_adv", "loss_dp", "loss_df"]):
        vals = [r["train"].get(k, float("nan")) for r in history]
        if not all(np.isnan(v) for v in vals):
            ax.plot(epochs, vals, color=LINE_PALETTE[i + 7], linewidth=1.4,
                    marker="o", markersize=2, label=k)
    apply_style(ax, "Adversarial Losses")

    # ── (1,1) Val segmentation ──
    ax = axes[1, 1]
    shade_stages(ax, bounds, epochs)
    dice = [r["val"].get("dice", float("nan")) for r in history]
    iou = [r["val"].get("iou", float("nan")) for r in history]
    if not all(np.isnan(v) for v in dice):
        ax.plot(epochs, dice, color="#6366f1", linewidth=2, marker="s", markersize=4, label="Dice")
    if not all(np.isnan(v) for v in iou):
        ax.plot(epochs, iou, color="#10b981", linewidth=2, marker="^", markersize=4, label="IoU")
    apply_style(ax, "Val — Dice & IoU", ylabel="Score")

    # ── (1,2) Val restoration ──
    ax = axes[1, 2]
    shade_stages(ax, bounds, epochs)
    psnr = [r["val"].get("psnr", float("nan")) for r in history]
    ssim = [r["val"].get("ssim", float("nan")) for r in history]
    if not all(np.isnan(v) for v in psnr):
        ax.plot(epochs, psnr, color="#3b82f6", linewidth=2, marker="o", markersize=4, label="PSNR")
    ax2 = ax.twinx()
    if not all(np.isnan(v) for v in ssim):
        ax2.plot(epochs, ssim, color="#10b981", linewidth=2, marker="^", markersize=4, label="SSIM")
        ax2.set_ylabel("SSIM", fontsize=10, color="#10b981")
        ax2.spines["top"].set_visible(False)
    apply_style(ax, "Val — PSNR & SSIM", ylabel="PSNR (dB)")
    # merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best", framealpha=0.85)

    fig.suptitle("RAFI++ Training Dashboard", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────── summary table ─────────────────────────────
def print_summary_table(history):
    """Print a formatted summary of the final metrics."""
    if not history:
        print("No training history found.")
        return

    last = history[-1]
    print("\n" + "=" * 60)
    print(f"  RAFI++ Training Summary — {last['epoch']} epoch(s)")
    print("=" * 60)

    # best metrics per category
    best_dice = max((r["val"].get("dice", 0) for r in history), default=0)
    best_iou = max((r["val"].get("iou", 0) for r in history), default=0)
    best_psnr = max((r["val"].get("psnr", 0) for r in history), default=0)
    best_ssim = max((r["val"].get("ssim", 0) for r in history), default=0)
    best_l1 = min((r["val"].get("l1", float("inf")) for r in history), default=0)

    print(f"\n  {'Metric':<20} {'Best':>12} {'Last Epoch':>12}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 12}")

    rows = [
        ("Dice", best_dice, last["val"].get("dice")),
        ("IoU", best_iou, last["val"].get("iou")),
        ("PSNR (dB)", best_psnr, last["val"].get("psnr")),
        ("SSIM", best_ssim, last["val"].get("ssim")),
        ("L1", best_l1 if best_l1 != float("inf") else None, last["val"].get("l1")),
    ]
    for name, best, cur in rows:
        b = f"{best:.6f}" if best and best != float("inf") else "—"
        c = f"{cur:.6f}" if cur is not None else "—"
        print(f"  {name:<20} {b:>12} {c:>12}")

    print(f"\n  Final train loss: {last['train'].get('loss_total', last['train'].get('loss_seg', '—')):.6f}")
    print("=" * 60 + "\n")


# ──────────────────────────── main ──────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RAFI++ training metrics")
    parser.add_argument(
        "--history", type=str, default="./logs/rafipp_run/history.json",
        help="Path to history.json produced by train.py",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./outputs/metrics_plots",
        help="Directory to save plot images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    history_path = Path(args.history)
    out_dir = Path(args.out_dir)

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}\n"
            "Make sure you have trained the model first (python train.py)."
        )

    history = load_history(history_path)
    if not history:
        print("History is empty. Nothing to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = stage_boundaries(history)

    print(f"Loaded {len(history)} epoch(s) from {history_path}")
    print(f"Saving plots to {out_dir}/\n")

    # Individual detailed plots
    plot_total_loss(history, bounds, out_dir)
    print("  ✓ total_loss.png")

    plot_segmentation_losses(history, bounds, out_dir)
    print("  ✓ segmentation_losses.png")

    plot_reconstruction_losses(history, bounds, out_dir)
    print("  ✓ reconstruction_losses.png")

    plot_adversarial_losses(history, bounds, out_dir)
    print("  ✓ adversarial_losses.png")

    plot_val_segmentation(history, bounds, out_dir)
    print("  ✓ val_segmentation.png")

    plot_val_restoration(history, bounds, out_dir)
    print("  ✓ val_restoration.png")

    # Combined dashboard
    plot_combined_dashboard(history, bounds, out_dir)
    print("  ✓ dashboard.png")

    print_summary_table(history)
    print(f"All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
