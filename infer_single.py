import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from Model.networks import RAFIpp
from Utils.visualization import save_tensor_image, tensor_to_uint8_image


def load_rgb(path: Path, image_size: int) -> Tuple[Image.Image, Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    image = image.resize((image_size, image_size), Image.BICUBIC)
    return image, original_size


def load_mask(path: Path, image_size: int, threshold: int, invert: bool) -> Image.Image:
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)
    arr = np.array(mask)
    if invert:
        arr = 255 - arr
    arr = (arr >= threshold).astype(np.uint8) * 255
    return Image.fromarray(arr)


def rgb_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def apply_black_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_arr = np.array(image).copy()
    mask_arr = np.array(mask)
    image_arr[mask_arr > 0] = 0
    return Image.fromarray(image_arr)


def black_out_tensor(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return image * (1.0 - mask) - mask


def compute_boundary(mask: Image.Image, thickness: int = 3) -> Image.Image:
    mask_arr = np.array(mask).astype(np.uint8)
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(mask_arr, kernel, iterations=1)
    eroded = cv2.erode(mask_arr, kernel, iterations=1)
    boundary = cv2.absdiff(dilated, eroded)
    boundary = (boundary > 0).astype(np.uint8) * 255
    return Image.fromarray(boundary)


def load_model(checkpoint: Path, device: torch.device) -> Dict[str, object]:
    model = RAFIpp().to(device)
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return {
        "model": model,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "stage": ckpt.get("stage") if isinstance(ckpt, dict) else None,
        "best_score": ckpt.get("best_score") if isinstance(ckpt, dict) else None,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def run_inference(
    model: RAFIpp,
    masked: torch.Tensor,
    guidance: str,
    provided_mask: Optional[torch.Tensor],
    provided_boundary: Optional[torch.Tensor],
    blackout_predicted_mask: bool,
) -> Dict[str, torch.Tensor]:
    if guidance == "predicted":
        seg = model.segnet(masked)
        restore_input = (
            black_out_tensor(masked, seg["mask_pred"]) if blackout_predicted_mask else masked
        )
        restore = model.restore(
            restore_input,
            seg["mask_pred"],
            seg["boundary_pred"],
            seg["confidence_pred"],
        )
        return {
            **seg,
            **restore,
            "restore_input": restore_input,
            "mask_used": seg["mask_pred"],
            "boundary_used": seg["boundary_pred"],
            "confidence_used": seg["confidence_pred"],
        }

    if provided_mask is None or provided_boundary is None:
        raise ValueError("Provided-mask guidance requires --mask.")

    seg = model.segnet(masked)
    confidence = torch.ones_like(provided_mask)
    restore = model.restore(masked, provided_mask, provided_boundary, confidence)
    return {
        **seg,
        **restore,
        "mask_used": provided_mask,
        "boundary_used": provided_boundary,
        "confidence_used": confidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAFI++ inference on one masked face image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image.")
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional binary mask. White pixels are the region to inpaint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./checkpoints/rafipp_run/epoch_040.pt"),
        help="Path to RAFI++ checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./outputs/single_inference"),
        help="Directory for output images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional direct path for the final blended output image.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--guidance",
        choices=("auto", "predicted", "provided"),
        default="auto",
        help=(
            "Segmentation maps used by RestoreNet. "
            "'auto' uses --mask when present, otherwise model prediction."
        ),
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=128,
        help="Threshold for binarizing --mask.",
    )
    parser.add_argument(
        "--invert_mask",
        action="store_true",
        help="Use this when black pixels, not white pixels, mark the inpaint region.",
    )
    parser.add_argument(
        "--no_apply_mask",
        action="store_true",
        help="Do not black out --image with --mask before inference.",
    )
    parser.add_argument(
        "--no_blackout_predicted_mask",
        action="store_true",
        help=(
            "When no --mask is provided, keep the original input pixels for RestoreNet "
            "instead of blacking out the model-predicted mask region."
        ),
    )
    parser.add_argument(
        "--boundary_thickness",
        type=int,
        default=3,
        help="Morphological boundary thickness derived from --mask.",
    )
    parser.add_argument(
        "--save_original_size",
        action="store_true",
        help="Also save the final output resized back to the input image size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if args.mask is not None and not args.mask.exists():
        raise FileNotFoundError(f"Input mask not found: {args.mask}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    guidance = args.guidance
    if guidance == "auto":
        guidance = "provided" if args.mask is not None else "predicted"
    if guidance == "provided" and args.mask is None:
        raise ValueError("--guidance provided requires --mask.")

    image, original_size = load_rgb(args.image, args.image_size)
    mask = None
    boundary = None
    if args.mask is not None:
        mask = load_mask(args.mask, args.image_size, args.mask_threshold, args.invert_mask)
        boundary = compute_boundary(mask, args.boundary_thickness)
        if not args.no_apply_mask:
            image = apply_black_mask(image, mask)

    device = resolve_device(args.device)
    ckpt_info = load_model(args.checkpoint, device)
    model = ckpt_info["model"]

    batch = rgb_to_tensor(image).unsqueeze(0).to(device)
    mask_tensor = mask_to_tensor(mask).unsqueeze(0).to(device) if mask is not None else None
    boundary_tensor = (
        mask_to_tensor(boundary).unsqueeze(0).to(device) if boundary is not None else None
    )

    outputs = run_inference(
        model,
        batch,
        guidance,
        mask_tensor,
        boundary_tensor,
        blackout_predicted_mask=guidance == "predicted" and not args.no_blackout_predicted_mask,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    save_tensor_image(batch[0], args.out_dir / f"{stem}_source.png")
    save_tensor_image(outputs.get("restore_input", batch)[0], args.out_dir / f"{stem}_input.png")
    save_tensor_image(outputs["mask_pred"][0], args.out_dir / f"{stem}_mask_pred.png", is_mask=True)
    save_tensor_image(outputs["restored"][0], args.out_dir / f"{stem}_restored.png")
    save_tensor_image(outputs["isyn"][0], args.out_dir / f"{stem}_isyn.png")

    if mask_tensor is not None:
        save_tensor_image(mask_tensor[0], args.out_dir / f"{stem}_mask_provided.png", is_mask=True)
    if "mask_used" in outputs:
        save_tensor_image(outputs["mask_used"][0], args.out_dir / f"{stem}_mask_used.png", is_mask=True)
    if "boundary_used" in outputs:
        save_tensor_image(outputs["boundary_used"][0], args.out_dir / f"{stem}_boundary_used.png", is_mask=True)

    if args.output is not None:
        save_tensor_image(outputs["isyn"][0], args.output)

    if args.save_original_size:
        final_arr = tensor_to_uint8_image(outputs["isyn"][0])
        final_image = Image.fromarray(final_arr).resize(original_size, Image.BICUBIC)
        final_image.save(args.out_dir / f"{stem}_isyn_original_size.png")
        if args.output is not None:
            final_image.save(args.output)

    meta = {
        "image": str(args.image),
        "mask": str(args.mask) if args.mask is not None else None,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt_info["epoch"],
        "checkpoint_stage": ckpt_info["stage"],
        "guidance": guidance,
        "image_size": args.image_size,
        "device": str(device),
        "applied_input_mask": args.mask is not None and not args.no_apply_mask,
        "blackout_predicted_mask": guidance == "predicted"
        and not args.no_blackout_predicted_mask,
    }
    (args.out_dir / f"{stem}_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Saved final output: {args.out_dir / f'{stem}_isyn.png'}")


if __name__ == "__main__":
    main()
