import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for Dataset/prepare_celeba_hq_masks.py. "
        "Install it with: pip install mediapipe"
    ) from exc


class FaceMaskPolygonGenerator:
    """Create lower-face mask polygons from MediaPipe FaceMesh landmarks.

    The goal here is not to simulate a photo-realistic physical mask, but to
    produce a stable lower-face occlusion region covering the area below the
    nose, around the mouth, and down to the chin, as requested.
    """

    # A compact set of points around cheeks, mouth, nose-bottom and chin.
    HULL_INDICES = [
        234,
        93,
        132,
        58,
        172,
        136,
        150,
        149,
        176,
        148,
        152,
        377,
        400,
        378,
        379,
        365,
        397,
        288,
        361,
        323,
        454,
        98,
        2,
        327,
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
    ]

    TOP_GUIDE_INDICES = [98, 2, 327, 205, 425]

    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def close(self) -> None:
        self.face_mesh.close()

    def _landmarks_to_pixels(
        self, results, width: int, height: int
    ) -> Optional[np.ndarray]:
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array(
            [[int(p.x * width), int(p.y * height)] for p in lm], dtype=np.int32
        )
        return pts

    def _expand_polygon(
        self,
        polygon: np.ndarray,
        center: np.ndarray,
        scale_x: float = 1.08,
        scale_y: float = 1.18,
    ) -> np.ndarray:
        shifted = polygon.astype(np.float32) - center[None, :]
        shifted[:, 0] *= scale_x
        shifted[:, 1] *= scale_y
        expanded = shifted + center[None, :]
        return np.round(expanded).astype(np.int32)

    def build_mask(
        self, image_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        pts = self._landmarks_to_pixels(results, image_bgr.shape[1], image_bgr.shape[0])
        if pts is None:
            return None

        hull_points = pts[self.HULL_INDICES]
        hull = cv2.convexHull(hull_points)
        hull = hull.squeeze(1)
        if hull.ndim != 2 or len(hull) < 3:
            return None

        # Keep only the lower part using nose-bottom as a soft top anchor.
        top_guide = pts[self.TOP_GUIDE_INDICES]
        top_y = int(np.percentile(top_guide[:, 1], 60))
        hull[:, 1] = np.maximum(hull[:, 1], top_y)

        center = hull.mean(axis=0)
        hull = self._expand_polygon(hull, center)
        hull[:, 0] = np.clip(hull[:, 0], 0, image_bgr.shape[1] - 1)
        hull[:, 1] = np.clip(hull[:, 1], 0, image_bgr.shape[0] - 1)

        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = (mask > 32).astype(np.uint8) * 255
        return mask, hull


def compute_boundary(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = cv2.absdiff(dilated, eroded)
    return (boundary > 0).astype(np.uint8) * 255


def mask_to_black(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    out[mask > 0] = 0
    return out


def read_image(path: Path, img_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def write_rgb(path: Path, image_bgr: np.ndarray) -> None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_rgb).save(path)


def split_names(
    names: List[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[str]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1."
    )
    rng = random.Random(seed)
    names = sorted(names)
    rng.shuffle(names)
    n = len(names)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = names[:n_train]
    val = names[n_train : n_train + n_val]
    test = names[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    img_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    gt_dir = out_dir / "gt"
    masked_dir = out_dir / "masked"
    mask_dir = out_dir / "masks"
    boundary_dir = out_dir / "boundaries"
    splits_dir = out_dir / "splits"
    meta_dir = out_dir / "meta"
    for d in [gt_dir, masked_dir, mask_dir, boundary_dir, splits_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    valid_ext = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = [
        p for p in sorted(raw_dir.iterdir()) if p.suffix.lower() in valid_ext
    ]
    generator = FaceMaskPolygonGenerator(img_size=img_size)

    kept_names: List[str] = []
    skipped: Dict[str, str] = {}

    try:
        for path in tqdm(image_paths, desc="Preparing RAFI++ dataset"):
            try:
                image_bgr = read_image(path, img_size)
                result = generator.build_mask(image_bgr)
                if result is None:
                    skipped[path.name] = "no_face_landmarks"
                    continue
                mask, hull = result
                boundary = compute_boundary(mask)
                masked = mask_to_black(image_bgr, mask)

                stem = path.stem
                out_name = f"{stem}.png"
                write_rgb(gt_dir / out_name, image_bgr)
                write_rgb(masked_dir / out_name, masked)
                Image.fromarray(mask).save(mask_dir / out_name)
                Image.fromarray(boundary).save(boundary_dir / out_name)
                (meta_dir / f"{stem}.json").write_text(
                    json.dumps({"hull": hull.tolist()}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                kept_names.append(out_name)
            except Exception as exc:
                skipped[path.name] = str(exc)
    finally:
        generator.close()

    splits = split_names(kept_names, train_ratio, val_ratio, test_ratio, seed)
    for split_name, split_items in splits.items():
        (splits_dir / f"{split_name}.txt").write_text(
            "\n".join(split_items) + ("\n" if split_items else ""), encoding="utf-8"
        )

    summary = {
        "raw_images": len(image_paths),
        "kept_images": len(kept_names),
        "skipped_images": len(skipped),
        "splits": {k: len(v) for k, v in splits.items()},
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "img_size": img_size,
    }
    (out_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "skipped.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare RAFI++ dataset from celeba_hq_256 using lower-face polygon masks."
    )
    parser.add_argument("--raw_dir", type=Path, default=Path("./celeba_hq_256"))
    parser.add_argument("--out_dir", type=Path, default=Path("./Dataset/CelebA/rafipp"))
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        img_size=args.img_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
