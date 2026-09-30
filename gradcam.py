"""
GradCAM visualization for FireSightNet.
Generates class activation maps to interpret model decisions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


CLASSES = ['No_Fire', 'Smoke', 'Fire', 'Fire_Smoke']


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients across channels
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx, output.softmax(dim=1)[0].detach().cpu().numpy()


def visualize_gradcam(model, image_path, target_layer, save_path=None,
                      device='cpu', class_idx=None):
    """
    Generate and display GradCAM for a single image.

    Args:
        model: FireSightNet model
        image_path: Path to input image
        target_layer: Target layer for GradCAM (e.g. model.cbam)
        save_path: Optional path to save the figure
        device: 'cpu' or 'cuda'
        class_idx: Class to visualize (None = predicted class)
    """
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    model = model.to(device)
    gradcam = GradCAM(model, target_layer)
    cam, pred_class, probs = gradcam.generate(img_tensor, class_idx)

    # Overlay
    heatmap = cm.jet(cam)[:, :, :3]
    img_np = np.array(img_pil) / 255.0
    overlay = 0.5 * img_np + 0.5 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_pil);             axes[0].set_title('Original Image');   axes[0].axis('off')
    axes[1].imshow(heatmap);             axes[1].set_title('GradCAM Heatmap');  axes[1].axis('off')
    axes[2].imshow(np.clip(overlay, 0, 1)); axes[2].set_title(f'Overlay\nPred: {CLASSES[pred_class]} ({probs[pred_class]*100:.1f}%)'); axes[2].axis('off')

    plt.suptitle(f'FireSightNet GradCAM — {CLASSES[pred_class]}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()

    return pred_class, probs


if __name__ == '__main__':
    import argparse
    from model.firesightnet import FireSightNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save',       default=None)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = FireSightNet()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    pred, probs = visualize_gradcam(
        model, args.image,
        target_layer=model.cbam,
        save_path=args.save,
        device=args.device
    )
    print(f"\nPrediction: {CLASSES[pred]}")
    for i, (cls, p) in enumerate(zip(CLASSES, probs)):
        print(f"  {cls}: {p*100:.2f}%")
