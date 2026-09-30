"""
Single image inference for FireSightNet.
Usage: python predict.py --image path/to/image.jpg --checkpoint checkpoints/firesightnet_best.pth
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms

from model.firesightnet import FireSightNet

CLASSES = ['No_Fire', 'Smoke', 'Fire', 'Fire_Smoke']
LABELS  = {
    'No_Fire':    '🟢 No Fire     — Scene is safe',
    'Smoke':      '🟡 Smoke       — Early-stage smoke detected. Monitor closely.',
    'Fire':       '🔴 Fire        — Active fire detected. Alert emergency services.',
    'Fire_Smoke': '🔴 Fire+Smoke  — Combined fire and smoke. Immediate action required.',
}


def predict(image_path: str, checkpoint: str, device: str = 'cpu',
            use_tta: bool = True) -> dict:
    """
    Run FireSightNet inference on a single image.

    Args:
        image_path: Path to input image
        checkpoint: Path to .pth model checkpoint
        device: 'cpu' or 'cuda'
        use_tta: Apply test-time augmentation

    Returns:
        dict with predicted class, confidence, and all class probabilities
    """
    device = torch.device(device)

    # Load model
    model = FireSightNet(pretrained=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Preprocess
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_tta:
            import torchvision.transforms.functional as TF
            logits = (model(x) + model(TF.hflip(x)) + model(TF.vflip(x))) / 3
        else:
            logits = model(x)

        probs = logits.softmax(dim=1)[0].cpu()
        pred_idx = probs.argmax().item()

    return {
        'predicted_class': CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]) * 100,
        'probabilities': {cls: float(p) * 100 for cls, p in zip(CLASSES, probs)},
    }


def main():
    parser = argparse.ArgumentParser(description='FireSightNet single image prediction')
    parser.add_argument('--image',      required=True, help='Input image path')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint (.pth)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-tta',     action='store_true', help='Disable TTA')
    args = parser.parse_args()

    result = predict(args.image, args.checkpoint, args.device, use_tta=not args.no_tta)

    print("\n" + "=" * 55)
    print("  FireSightNet — Prediction Result")
    print("=" * 55)
    print(f"  Image : {args.image}")
    print(f"  {LABELS[result['predicted_class']]}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print("\n  Class Probabilities:")
    for cls, prob in result['probabilities'].items():
        bar = '█' * int(prob / 5)
        print(f"    {cls:<15} {prob:6.2f}%  {bar}")
    print("=" * 55)


if __name__ == '__main__':
    main()
