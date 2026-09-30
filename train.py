"""
Training script for FireSightNet.
Paper: Expert Systems with Applications
Author: Ranjith Kumar Anandan, SRM Institute of Science and Technology
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.firesightnet import FireSightNet, FocalCrossEntropyLoss
from utils.dataset import get_dataloaders
from utils.metrics import compute_all_metrics, print_metrics


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in tqdm(loader, desc='Eval ', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss  += loss.item() * imgs.size(0)
        all_preds   += logits.argmax(1).cpu().tolist()
        all_labels  += labels.cpu().tolist()
    metrics = compute_all_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(all_labels)
    return metrics


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(cfg['training']['seed'])

    # ── Data ──────────────────────────────────────────────
    loaders = get_dataloaders(
        cfg['dataset']['root'],
        batch_size=cfg['training']['batch_size'],
        image_size=cfg['dataset']['image_size'],
        num_workers=cfg['training']['num_workers'],
    )

    # ── Model ─────────────────────────────────────────────
    model = FireSightNet(
        num_classes=cfg['model']['num_classes'],
        transformer_heads=cfg['model']['transformer_heads'],
        use_transformer=True,
        use_cbam=cfg['model']['cbam'],
        use_fpn=cfg['model']['fpn'],
        use_triple_pooling=cfg['model']['triple_pooling'],
        dropout1=cfg['model']['dropout1'],
        dropout2=cfg['model']['dropout2'],
        pretrained=cfg['model']['pretrained'],
    ).to(device)

    # ── Loss ──────────────────────────────────────────────
    criterion = FocalCrossEntropyLoss(
        gamma=cfg['loss']['gamma'],
        label_smoothing=cfg['loss']['label_smoothing'],
        num_classes=cfg['model']['num_classes'],
    )

    # ── Optimizer + Scheduler ─────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['optimizer']['lr'],
        weight_decay=cfg['optimizer']['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs'],
        eta_min=cfg['scheduler']['min_lr'],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────
    save_dir = cfg['checkpointing']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders['train'], optimizer, criterion, device, scaler
        )
        val_metrics = evaluate(model, loaders['val'], criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.2f}% "
              f"FSI: {val_metrics['fsi']:.2f}%")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_acc,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, 'firesightnet_best.pth'))
            print(f"  ✓ Saved best model (Acc={best_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Accuracy: {best_acc:.2f}%")

    # ── Final test evaluation ──────────────────────────────
    ckpt = torch.load(os.path.join(save_dir, 'firesightnet_best.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, loaders['test'], criterion, device)
    print("\n=== Test Set Results ===")
    print_metrics(test_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/firesightnet_config.yaml')
    args = parser.parse_args()
    main(args.config)
