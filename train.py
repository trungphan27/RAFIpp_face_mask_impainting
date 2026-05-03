import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.datasets import RAFIppCelebA
from Experiments.configs import get_args, save_args
from Model import RAFIppSystem
from Utils.metrics import dice_score, iou_score, summarize_restoration
from Utils.seed import seed_everything
from Utils.visualization import save_training_grid


@torch.no_grad()
def validate_seg(system: RAFIppSystem, loader: DataLoader, device: torch.device):
    system.eval()
    dice_list = []
    iou_list = []
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = system.inference(batch)
        dice_list.append(float(dice_score(outputs["mask_pred"], batch["mask"]).item()))
        iou_list.append(float(iou_score(outputs["mask_pred"], batch["mask"]).item()))
    return {
        "dice": sum(dice_list) / max(len(dice_list), 1),
        "iou": sum(iou_list) / max(len(iou_list), 1),
    }


@torch.no_grad()
def validate_restore(system: RAFIppSystem, loader: DataLoader, device: torch.device):
    system.eval()
    sums = defaultdict(float)
    count = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = system.inference(batch)
        metrics = summarize_restoration(outputs["isyn"], batch["gt"])
        for k, v in metrics.items():
            sums[k] += float(v)
        count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}


def epoch_to_stage(epoch: int, args) -> int:
    if epoch <= args.stage1_epochs:
        return 1
    if epoch <= args.stage1_epochs + args.stage2_epochs:
        return 2
    return 3


def flatten_record(record: dict) -> dict:
    row = {
        "epoch": record["epoch"],
        "stage": record["stage"],
    }
    for k, v in record.get("train", {}).items():
        row[f"train_{k}"] = v
    for k, v in record.get("val", {}).items():
        row[f"val_{k}"] = v
    return row


def write_metrics_csv(history: list, csv_path: Path):
    rows = [flatten_record(r) for r in history]
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = get_args()
    seed_everything(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_args(args, run_dir / "config.yaml")

    history_path = run_dir / "history.json"
    csv_path = run_dir / "metrics.csv"

    train_set = RAFIppCelebA(
        args.data_root, split="train", image_size=args.image_size, augment=True
    )
    val_set = RAFIppCelebA(
        args.data_root, split="val", image_size=args.image_size, augment=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    system = RAFIppSystem(args).to(device)

    start_epoch = 1
    best_score = None
    history = []

    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            history = []

    if args.resume:
        state = system.load_checkpoint(args.resume, map_location=device)
        start_epoch = state["epoch"] + 1
        best_score = state.get("best_score", None)

        # Nếu resume mà history.json chưa load được, vẫn cố đồng bộ số epoch cũ
        if not history_path.exists() and start_epoch > 1:
            history = []

    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs

    for epoch in range(start_epoch, total_epochs + 1):
        stage = epoch_to_stage(epoch, args)
        system.train()
        meter = defaultdict(float)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} - stage {stage}")

        for step, batch in enumerate(pbar, start=1):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            stats = system.train_step(batch, stage)

            for k, v in stats.items():
                meter[k] += float(v)

            avg_total = meter.get("loss_total", meter.get("loss_seg", 0.0)) / step
            pbar.set_postfix(loss=f"{avg_total:.4f}")

            if step == 1:
                outputs = system.inference(batch)
                sample_path = (
                    Path(args.sample_dir) / args.run_name / f"epoch_{epoch:03d}.png"
                )
                save_training_grid(
                    batch, outputs, sample_path, nrow=batch["gt"].shape[0]
                )

        epoch_stats = {k: v / max(len(train_loader), 1) for k, v in meter.items()}

        if stage == 1:
            val_metrics = validate_seg(system, val_loader, device)
            current_score = val_metrics["dice"]
            better = best_score is None or current_score > best_score
        else:
            val_metrics = validate_restore(system, val_loader, device)
            current_score = -val_metrics["l1"]
            better = best_score is None or current_score > best_score

        record = {
            "epoch": epoch,
            "stage": stage,
            "train": epoch_stats,
            "val": val_metrics,
        }
        history.append(record)

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        write_metrics_csv(history, csv_path)

        latest_ckpt = Path(args.checkpoint_dir) / args.run_name / "latest.pt"
        system.save_checkpoint(
            latest_ckpt, epoch=epoch, stage=stage, best_score=current_score
        )

        if epoch % args.save_every == 0:
            system.save_checkpoint(
                Path(args.checkpoint_dir) / args.run_name / f"epoch_{epoch:03d}.pt",
                epoch=epoch,
                stage=stage,
                best_score=current_score,
            )

        if better:
            best_score = current_score
            system.save_checkpoint(
                Path(args.checkpoint_dir) / args.run_name / f"best_stage{stage}.pt",
                epoch=epoch,
                stage=stage,
                best_score=current_score,
            )

        print(json.dumps(record, indent=2))
        print(f"[INFO] Metrics CSV updated: {csv_path}")


if __name__ == "__main__":
    main()
