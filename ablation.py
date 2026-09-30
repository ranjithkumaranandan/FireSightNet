"""
Ablation study script for FireSightNet.
Trains and evaluates all 7 architectural variants (A–G).
Reproduces Table 6 from the paper.
"""

import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.firesightnet import build_variant, FocalCrossEntropyLoss
from utils.dataset import get_dataloaders
from utils.metrics import compute_all_metrics, print_metrics, fire_spread_index

VARIANTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
VARIANT_NAMES = {
    'A': 'EfficientNet-B4 + GAP + FC (Baseline)',
    'B': 'A + CBAM Attention',
    'C': 'A + Transformer Self-Attention (8 heads)',
    'D': 'A + FPN (Feature Pyramid Network)',
    'E': 'A + CBAM + FPN',
    'F': 'A + Transformer + CBAM (no FPN)',
    'G': 'All components + Triple Pooling + Focal Loss + TTA',
}


def train_variant(variant, loaders, device, epochs=25, lr=1e-4, seed=42):
    torch.manual_seed(seed)
    model = build_variant(variant, pretrained=True).to(device)

    criterion = FocalCrossEntropyLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for imgs, labels in tqdm(loaders['train'],
                                  desc=f'  Variant {variant} Epoch {epoch}/{epochs}',
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Val
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in loaders['val']:
                imgs = imgs.to(device)
                preds += model(imgs).argmax(1).cpu().tolist()
                gts   += labels.tolist()
        acc = sum(p == g for p, g in zip(preds, gts)) / len(gts) * 100
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test with best weights
    model.load_state_dict(best_state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in loaders['test']:
            preds += model(imgs.to(device)).argmax(1).cpu().tolist()
            gts   += labels.tolist()

    metrics = compute_all_metrics(gts, preds)
    return metrics, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    required=True, help='Dataset root')
    parser.add_argument('--epochs',  type=int, default=25)
    parser.add_argument('--batch',   type=int, default=32)
    parser.add_argument('--lr',      type=float, default=1e-4)
    parser.add_argument('--save',    default='ablation_results/')
    parser.add_argument('--device',  default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    device = torch.device(args.device)

    loaders = get_dataloaders(args.data, batch_size=args.batch)

    results = []
    baseline_acc = None

    for variant in VARIANTS:
        print(f"\n{'='*60}")
        print(f"  Variant {variant}: {VARIANT_NAMES[variant]}")
        print(f"{'='*60}")

        metrics, model = train_variant(
            variant, loaders, device, args.epochs, args.lr
        )

        if baseline_acc is None:
            baseline_acc = metrics['accuracy']

        gain = metrics['accuracy'] - baseline_acc
        metrics['variant'] = variant
        metrics['name'] = VARIANT_NAMES[variant]
        metrics['gain'] = gain
        results.append(metrics)

        # Save checkpoint
        torch.save(model.state_dict(),
                   os.path.join(args.save, f'ablation_{variant}.pth'))

        print(f"\n  Results — Variant {variant}:")
        print(f"    Accuracy : {metrics['accuracy']:.2f}%")
        print(f"    F1-Score : {metrics['f1_score']:.2f}%")
        print(f"    FSI      : {metrics['fsi']:.2f}%")
        print(f"    Gain     : {gain:+.2f}%")

    # Save CSV (reproduces Table 6)
    csv_path = os.path.join(args.save, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant', 'Components', 'Accuracy(%)', 'Precision(%)',
                         'Recall(%)', 'F1-Score(%)', 'FSI(%)', 'Gain(%)'])
        for r in results:
            writer.writerow([
                r['variant'], r['name'],
                f"{r['accuracy']:.2f}", f"{r['precision']:.2f}",
                f"{r['recall']:.2f}",   f"{r['f1_score']:.2f}",
                f"{r['fsi']:.2f}",      f"{r['gain']:+.2f}",
            ])
    print(f"\nAblation results saved to: {csv_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY SUMMARY (Table 6)")
    print(f"{'='*80}")
    print(f"  {'Var':<4} {'Accuracy':>10} {'F1':>8} {'FSI':>8} {'Gain':>8}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['variant']:<4} {r['accuracy']:>9.2f}% {r['f1_score']:>7.2f}% "
              f"{r['fsi']:>7.2f}% {r['gain']:>+8.2f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
