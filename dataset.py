"""
Dataset loading and preprocessing for FireSightNet.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASSES = ['No_Fire', 'Smoke', 'Fire', 'Fire_Smoke']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class FireSightDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        for cls in CLASSES:
            cls_dir = os.path.join(self.root, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(split='train', image_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloaders(root, batch_size=32, image_size=224, num_workers=4):
    loaders = {}
    for split in ['train', 'val', 'test']:
        ds = FireSightDataset(root, split, get_transforms(split, image_size))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    return loaders
