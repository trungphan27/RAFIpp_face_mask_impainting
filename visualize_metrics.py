"""
Visualize RAFI++ training metrics from metrics.csv and history.json.

Examples:
    python visualize_metrics.py
    python visualize_metrics.py --metrics logs/rafipp_run/metrics.csv --history logs/rafipp_run/history.json
    python visualize_metrics.py --source json --history logs/rafipp_run/history.json
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
STAGE_COLORS = {1: "#6366f1", 2: "#f59e0b", 3: "#10b981"}
STAGE_BG = {1: "#eef2ff", 2: "#fffbeb", 3: "#ecfdf5"}
LINE_PALETTE = [
    "#6366f1",
    "#ef4444",
    "#f59e0b",
    "#10b981",
    "#3b82f6",
    "#ec4899",
    "#8b5cf6",
    "#14b8a6",
    "#f97316",
    "#64748b",
    "#a855f7",
    "#06b6d4",
]


def parse_float(value):
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def parse_int(value, default=0):
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def has_series_data(records, keys):
    for key in keys:
        if any(is_finite(record.get(key)) for record in records):
            return True
    return False


def clean_records(records):
    cleaned = []
    for record in records:
        epoch = parse_int(record.get("epoch"), default=0)
        if epoch <= 0:
            continue

        row = {"epoch": epoch, "stage": parse_int(record.get("stage"), default=0)}
        for key, value in record.items():
            if key in {"epoch", "stage"}:
                continue
            row[key] = parse_float(value)
        cleaned.append(row)
    return cleaned


def dedupe_records(records, policy):
    if policy == "none":
        return sorted(records, key=lambda row: (row["epoch"], row.get("stage", 0)))
    if policy != "last":
        raise ValueError(f"Unsupported dedupe policy: {policy}")

    by_epoch = {}
    order = []
    for record in records:
        epoch = record["epoch"]
        if epoch not in by_epoch:
            order.append(epoch)
        by_epoch[epoch] = record
    return [by_epoch[epoch] for epoch in sorted(order)]


def flatten_history_record(record):
    flat = {
        "epoch": record.get("epoch"),
        "stage": record.get("stage"),
    }
    for key, value in record.get("train", {}).items():
        flat[f"train_{key}"] = value
    for key, value in record.get("val", {}).items():
        flat[f"val_{key}"] = value
    return flat


def load_csv_records(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return clean_records(list(reader))


def load_json_records(path):
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    return clean_records([flatten_history_record(record) for record in history])


def load_records(source, metrics_path, history_path, dedupe_policy):
    errors = []

    if source in {"auto", "csv"} and metrics_path.exists():
        try:
            records = load_csv_records(metrics_path)
            if records:
                return dedupe_records(records, dedupe_policy), "csv"
            errors.append(f"CSV has no usable records: {metrics_path}")
        except Exception as exc:
            errors.append(f"CSV load failed: {exc}")

    if source in {"auto", "json"} and history_path.exists():
        try:
            records = load_json_records(history_path)
            if records:
                return dedupe_records(records, dedupe_policy), "json"
            errors.append(f"JSON has no usable records: {history_path}")
        except Exception as exc:
            errors.append(f"JSON load failed: {exc}")

    if source == "csv" and not metrics_path.exists():
        errors.append(f"CSV file not found: {metrics_path}")
    if source == "json" and not history_path.exists():
        errors.append(f"JSON file not found: {history_path}")
    if source == "auto" and not metrics_path.exists() and not history_path.exists():
        errors.append(f"No input found: {metrics_path} or {history_path}")

    detail = "\n".join(f"- {error}" for error in errors)
    raise FileNotFoundError(f"Could not load training metrics.\n{detail}")


def stage_boundaries(records):
    if not records:
        return []

    bounds = []
    current_stage = records[0].get("stage", 0)
    start_epoch = records[0]["epoch"]
    previous_epoch = records[0]["epoch"]

    for record in records[1:]:
        stage = record.get("stage", 0)
        epoch = record["epoch"]
        if stage != current_stage:
            bounds.append((start_epoch, previous_epoch, current_stage))
            current_stage = stage
            start_epoch = epoch
        previous_epoch = epoch

    bounds.append((start_epoch, previous_epoch, current_stage))
    return bounds


def shade_stages(ax, bounds):
    for start, end, stage in bounds:
        if stage <= 0:
            continue
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=STAGE_BG.get(stage, "#f8fafc"),
            alpha=0.55,
            zorder=0,
        )

    for start, end, stage in bounds:
        if stage <= 0:
            continue
        mid = (start + end) / 2
        ax.text(
            mid,
            1.02,
            f"Stage {stage}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=STAGE_COLORS.get(stage, "#64748b"),
        )


def apply_style(ax, title, xlabel="Epoch", ylabel="Value", show_legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def series(records, key):
    return [record.get(key, float("nan")) for record in records]


def stage_filtered_series(records, key, allowed_stages):
    values = []
    for record in records:
        if record.get("stage") in allowed_stages:
            values.append(record.get(key, float("nan")))
        else:
            values.append(float("nan"))
    return values


def save_figure(fig, out_dir, filename):
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_total_loss(records, bounds, out_dir):
    epochs = series(records, "epoch")
    values = []
    for record in records:
        total = record.get("train_loss_total", float("nan"))
        if not is_finite(total):
            total = record.get("train_loss_seg", float("nan"))
        values.append(total)

    if not any(is_finite(value) for value in values):
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds)
    ax.plot(epochs, values, color="#ef4444", linewidth=2.2, marker="o", markersize=4, label="total loss")
    apply_style(ax, "Total Loss per Epoch", ylabel="Loss")
    return save_figure(fig, out_dir, "total_loss.png")


def plot_loss_group(records, bounds, out_dir, keys, title, filename, ylabel="Loss", start_color=0, allowed_stages=None):
    filtered_records = records
    if allowed_stages is not None:
        filtered_records = [record for record in records if record.get("stage") in allowed_stages]

    if not has_series_data(filtered_records, keys):
        return None

    epochs = series(records, "epoch")
    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds)

    for index, key in enumerate(keys):
        if allowed_stages is None:
            values = series(records, key)
        else:
            values = stage_filtered_series(records, key, allowed_stages)
        if not any(is_finite(value) for value in values):
            continue
        linewidth = 2.2 if key.endswith("_total") or key.endswith("_seg") else 1.5
        linestyle = "-" if linewidth > 2 else "--"
        color = LINE_PALETTE[(start_color + index) % len(LINE_PALETTE)]
        ax.plot(epochs, values, color=color, linewidth=linewidth, linestyle=linestyle, marker="o", markersize=3, label=key)

    apply_style(ax, title, ylabel=ylabel)
    return save_figure(fig, out_dir, filename)


def plot_val_segmentation(records, bounds, out_dir):
    keys = ["val_dice", "val_iou"]
    if not has_series_data(records, keys):
        return None

    epochs = series(records, "epoch")
    fig, ax = plt.subplots(figsize=(10, 5))
    shade_stages(ax, bounds)

    dice = series(records, "val_dice")
    iou = series(records, "val_iou")
    if any(is_finite(value) for value in dice):
        ax.plot(epochs, dice, color="#6366f1", linewidth=2, marker="s", markersize=5, label="Dice")
    if any(is_finite(value) for value in iou):
        ax.plot(epochs, iou, color="#10b981", linewidth=2, marker="^", markersize=5, label="IoU")

    finite_values = [value for value in dice + iou if is_finite(value)]
    if finite_values:
        ax.set_ylim(bottom=max(0, min(finite_values) - 0.05), top=min(1.02, max(1.0, max(finite_values) + 0.01)))
    apply_style(ax, "Validation Segmentation Metrics", ylabel="Score")
    return save_figure(fig, out_dir, "val_segmentation.png")


def plot_val_restoration(records, bounds, out_dir):
    keys = ["val_l1", "val_psnr", "val_ssim"]
    if not has_series_data(records, keys):
        return None

    epochs = series(records, "epoch")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plots = [
        (axes[0], "val_l1", "Validation L1", "#ef4444", "L1"),
        (axes[1], "val_psnr", "Validation PSNR", "#3b82f6", "PSNR (dB)"),
        (axes[2], "val_ssim", "Validation SSIM", "#10b981", "SSIM"),
    ]

    for ax, key, title, color, ylabel in plots:
        shade_stages(ax, bounds)
        values = series(records, key)
        if any(is_finite(value) for value in values):
            ax.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4, label=key)
        apply_style(ax, title, ylabel=ylabel)

    fig.suptitle("Validation Restoration Metrics", fontsize=14, fontweight="bold", y=1.03)
    return save_figure(fig, out_dir, "val_restoration.png")


def plot_combined_dashboard(records, bounds, out_dir):
    epochs = series(records, "epoch")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    total = []
    for record in records:
        value = record.get("train_loss_total", float("nan"))
        if not is_finite(value):
            value = record.get("train_loss_seg", float("nan"))
        total.append(value)
    ax = axes[0, 0]
    shade_stages(ax, bounds)
    if any(is_finite(value) for value in total):
        ax.plot(epochs, total, color="#ef4444", linewidth=2, marker="o", markersize=3, label="total loss")
    apply_style(ax, "Total Loss", ylabel="Loss")

    dashboard_groups = [
        (
            axes[0, 1],
            ["train_loss_mask", "train_loss_boundary", "train_loss_conf", "train_loss_seg"],
            "Segmentation Losses",
            "Loss",
            0,
            {1, 3},
        ),
        (
            axes[0, 2],
            ["train_loss_rec", "train_loss_ssim", "train_loss_perc", "train_loss_style"],
            "Reconstruction Losses",
            "Loss",
            3,
            {2, 3},
        ),
        (
            axes[1, 0],
            ["train_loss_adv", "train_loss_dp", "train_loss_df", "train_loss_d_total"],
            "Adversarial Losses",
            "Loss",
            7,
            {2, 3},
        ),
    ]

    for ax, keys, title, ylabel, color_start, allowed_stages in dashboard_groups:
        shade_stages(ax, bounds)
        for index, key in enumerate(keys):
            values = stage_filtered_series(records, key, allowed_stages)
            if any(is_finite(value) for value in values):
                ax.plot(
                    epochs,
                    values,
                    color=LINE_PALETTE[(color_start + index) % len(LINE_PALETTE)],
                    linewidth=1.4,
                    marker="o",
                    markersize=2,
                    label=key,
                )
        apply_style(ax, title, ylabel=ylabel)

    ax = axes[1, 1]
    shade_stages(ax, bounds)
    dice = series(records, "val_dice")
    iou = series(records, "val_iou")
    if any(is_finite(value) for value in dice):
        ax.plot(epochs, dice, color="#6366f1", linewidth=2, marker="s", markersize=4, label="Dice")
    if any(is_finite(value) for value in iou):
        ax.plot(epochs, iou, color="#10b981", linewidth=2, marker="^", markersize=4, label="IoU")
    apply_style(ax, "Val Dice and IoU", ylabel="Score")

    ax = axes[1, 2]
    shade_stages(ax, bounds)
    psnr = series(records, "val_psnr")
    ssim = series(records, "val_ssim")
    if any(is_finite(value) for value in psnr):
        ax.plot(epochs, psnr, color="#3b82f6", linewidth=2, marker="o", markersize=4, label="PSNR")
    ax2 = ax.twinx()
    if any(is_finite(value) for value in ssim):
        ax2.plot(epochs, ssim, color="#10b981", linewidth=2, marker="^", markersize=4, label="SSIM")
        ax2.set_ylabel("SSIM", fontsize=10, color="#10b981")
        ax2.spines["top"].set_visible(False)
    apply_style(ax, "Val PSNR and SSIM", ylabel="PSNR (dB)")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles1 or handles2:
        ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="best", framealpha=0.85)

    fig.suptitle("RAFI++ Training Dashboard", fontsize=16, fontweight="bold", y=1.01)
    return save_figure(fig, out_dir, "dashboard.png")


def best_value(records, key, mode):
    values = [record.get(key) for record in records if is_finite(record.get(key))]
    if not values:
        return None
    return min(values) if mode == "min" else max(values)


def last_value(records, key):
    for record in reversed(records):
        value = record.get(key)
        if is_finite(value):
            return value
    return None


def format_value(value):
    return f"{value:.6f}" if value is not None and is_finite(value) else "-"


def print_summary(records, source_name, out_dir, saved_paths):
    print("\n" + "=" * 64)
    print("RAFI++ Training Metrics Summary")
    print("=" * 64)
    print(f"Source: {source_name}")
    print(f"Records after dedupe: {len(records)}")
    print(f"Epoch range: {records[0]['epoch']} - {records[-1]['epoch']}")
    print(f"Plots saved: {out_dir.resolve()}")

    rows = [
        ("Dice", best_value(records, "val_dice", "max"), last_value(records, "val_dice")),
        ("IoU", best_value(records, "val_iou", "max"), last_value(records, "val_iou")),
        ("PSNR (dB)", best_value(records, "val_psnr", "max"), last_value(records, "val_psnr")),
        ("SSIM", best_value(records, "val_ssim", "max"), last_value(records, "val_ssim")),
        ("L1", best_value(records, "val_l1", "min"), last_value(records, "val_l1")),
    ]

    print("\n  Metric             Best        Last")
    print("  ----------------  ----------  ----------")
    for name, best, last in rows:
        print(f"  {name:<16}  {format_value(best):>10}  {format_value(last):>10}")

    final_loss = last_value(records, "train_loss_total")
    if final_loss is None:
        final_loss = last_value(records, "train_loss_seg")
    print(f"\nFinal train loss: {format_value(final_loss)}")

    print("\nGenerated files:")
    for path in saved_paths:
        print(f"  - {path.name}")
    print("=" * 64 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RAFI++ training metrics")
    parser.add_argument("--metrics", type=str, default="./logs/rafipp_run/metrics.csv", help="Path to flattened metrics.csv")
    parser.add_argument("--history", type=str, default="./logs/rafipp_run/history.json", help="Path to history.json fallback")
    parser.add_argument("--out_dir", type=str, default="./outputs/metrics_plots", help="Directory to save plot images")
    parser.add_argument("--dedupe", choices=["last", "none"], default="last", help="How to handle duplicate epochs")
    parser.add_argument("--source", choices=["auto", "csv", "json"], default="auto", help="Input source priority")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = Path(args.metrics)
    history_path = Path(args.history)
    out_dir = Path(args.out_dir)

    records, source_name = load_records(args.source, metrics_path, history_path, args.dedupe)
    if not records:
        print("No metrics to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = stage_boundaries(records)

    saved_paths = []
    plot_jobs = [
        lambda: plot_total_loss(records, bounds, out_dir),
        lambda: plot_loss_group(
            records,
            bounds,
            out_dir,
            ["train_loss_mask", "train_loss_boundary", "train_loss_conf", "train_loss_seg"],
            "Segmentation Losses",
            "segmentation_losses.png",
            allowed_stages={1, 3},
        ),
        lambda: plot_loss_group(
            records,
            bounds,
            out_dir,
            ["train_loss_rec", "train_loss_ssim", "train_loss_perc", "train_loss_style", "train_loss_id", "train_loss_edge"],
            "Reconstruction Losses",
            "reconstruction_losses.png",
            start_color=3,
            allowed_stages={2, 3},
        ),
        lambda: plot_loss_group(
            records,
            bounds,
            out_dir,
            ["train_loss_adv", "train_loss_dp", "train_loss_df", "train_loss_d_total"],
            "Adversarial Losses",
            "adversarial_losses.png",
            start_color=7,
            allowed_stages={2, 3},
        ),
        lambda: plot_val_segmentation(records, bounds, out_dir),
        lambda: plot_val_restoration(records, bounds, out_dir),
        lambda: plot_combined_dashboard(records, bounds, out_dir),
    ]

    for job in plot_jobs:
        path = job()
        if path is not None:
            saved_paths.append(path)

    print_summary(records, source_name, out_dir, saved_paths)


if __name__ == "__main__":
    main()
