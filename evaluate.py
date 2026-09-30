"""
Evaluation script for FireSightNet.
Computes all paper metrics including FSI, mAP@0.5, Dice, Boundary F1.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

from model.firesightnet import FireSightNet
from utils.dataset import get_dataloaders, get_transforms, CLASSES
from utils.metrics import compute_all_metrics, print_metrics, fire_spread_index
from torch.utils.data import DataLoader


def test_time_augmentation(model, imgs, n_aug=5, device='cpu'):
    """Apply TTA and average predictions."""
    import torchvision.transforms.functional as TF
    preds = [model(imgs)]
    for _ in range(n_aug - 1):
        aug = TF.hflip(imgs) if torch.rand(1) > 0.5 else imgs
        preds.append(model(aug))
    return torch.stack(preds).mean(0)


@torch.no_grad()
def evaluate_full(model, data_root, checkpoint, device, use_tta=True, batch_size=32):
    model.eval()
    transform = get_transforms('test')
    from utils.dataset import FireSightDataset
    ds = FireSightDataset(data_root, 'test', transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc='Evaluating'):
        imgs = imgs.to(device)
        if use_tta:
            logits = test_time_augmentation(model, imgs, device=device)
        else:
            logits = model(imgs)
        probs = logits.softmax(dim=1).cpu()
        all_probs   += probs.tolist()
        all_preds   += logits.argmax(1).cpu().tolist()
        all_labels  += labels.tolist()

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                cbar_kws={'label': 'Recall (%)'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('FireSightNet — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curves(y_true, y_prob, save_path=None):
    y_bin = np.eye(len(CLASSES))[y_true]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#E05C5C', '#F5A623', '#4A90D9', '#7ED321']

    for i, (cls, col) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{cls} (AUC={auc_score:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('FireSightNet — ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--data',       required=True, help='Dataset root directory')
    parser.add_argument('--no-tta',     action='store_true', help='Disable TTA')
    parser.add_argument('--save-dir',   default='results/', help='Save figures here')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model = FireSightNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    y_true, y_pred, y_prob = evaluate_full(
        model, args.data, args.checkpoint, device,
        use_tta=not args.no_tta
    )

    # Metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)

    # Plots
    plot_confusion_matrix(y_true, y_pred,
                          save_path=os.path.join(args.save_dir, 'confusion_matrix.png'))
    plot_roc_curves(y_true, y_prob,
                    save_path=os.path.join(args.save_dir, 'roc_curves.png'))

    print(f"\nResults saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
