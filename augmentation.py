"""
Offline augmentation pipeline for FireSightNet 4-Class Dataset construction.
Implements Algorithm 1 from the paper — class balancing to 6,000 images per class.
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


TARGET_PER_CLASS = 6000
CLASSES = ['No_Fire', 'Smoke', 'Fire', 'Fire_Smoke']


def augment_image(img: Image.Image) -> Image.Image:
    """Apply one random augmentation from the paper's augmentation pool."""
    ops = [
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(random.uniform(-15, 15), expand=False),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: ImageEnhance.Saturation(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))),
        lambda x: _random_crop_resize(x),
    ]
    n_ops = random.randint(1, 3)
    chosen = random.sample(ops, n_ops)
    for op in chosen:
        img = op(img)
    return img


def _random_crop_resize(img: Image.Image, scale=(0.8, 1.0)) -> Image.Image:
    w, h = img.size
    s = random.uniform(*scale)
    nw, nh = int(w * s), int(h * s)
    left = random.randint(0, w - nw)
    top  = random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)


def balance_class(src_dir: str, dst_dir: str, target: int = TARGET_PER_CLASS,
                  seed: int = 42):
    """
    Balance one class folder to `target` images using offline augmentation.
    Only operates on training partition images — never val/test.

    Args:
        src_dir: Source folder with original images
        dst_dir: Destination folder for balanced images
        target:  Target number of images (default 6,000)
        seed:    Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(dst_dir, exist_ok=True)

    images = [f for f in os.listdir(src_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n = len(images)

    # Copy originals first
    for fname in images:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            Image.open(src).convert('RGB').save(dst)

    # Generate augmented images to reach target
    aug_idx = 0
    while len(os.listdir(dst_dir)) < target:
        src_fname = random.choice(images)
        img = Image.open(os.path.join(src_dir, src_fname)).convert('RGB')
        aug = augment_image(img)
        out_name = f"aug_{aug_idx:06d}.jpg"
        aug.save(os.path.join(dst_dir, out_name), quality=95)
        aug_idx += 1

    print(f"  {os.path.basename(dst_dir)}: {n} → {len(os.listdir(dst_dir))} images")


def build_balanced_dataset(raw_root: str, out_root: str,
                            target: int = TARGET_PER_CLASS):
    """
    Build the complete balanced FireSightNet 4-Class Dataset.
    Only balances the training partition.

    Args:
        raw_root: Root of the fused raw dataset (train/val/test splits already done)
        out_root: Output root for the balanced dataset
    """
    print(f"\nBuilding balanced dataset (target: {target} per class)")
    print(f"Source: {raw_root}")
    print(f"Output: {out_root}\n")

    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            src = os.path.join(raw_root, split, cls)
            dst = os.path.join(out_root, split, cls)
            if not os.path.exists(src):
                print(f"  WARNING: {src} not found, skipping.")
                continue
            if split == 'train':
                balance_class(src, dst, target)
            else:
                # Val/test: copy originals only — no augmentation
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        Image.open(os.path.join(src, f)).convert('RGB').save(
                            os.path.join(dst, f)
                        )
                print(f"  {split}/{cls}: copied {len(os.listdir(dst))} images (no augmentation)")

    print("\nDataset construction complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',    required=True, help='Raw fused dataset root')
    parser.add_argument('--out',    required=True, help='Output balanced dataset root')
    parser.add_argument('--target', type=int, default=TARGET_PER_CLASS)
    args = parser.parse_args()
    build_balanced_dataset(args.raw, args.out, args.target)
