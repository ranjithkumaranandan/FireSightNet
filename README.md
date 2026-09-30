# 🔥 FireSightNet: A Novel Deep Learning Framework for Multi-Class Forest Fire and Smoke Detection

[![Paper](https://img.shields.io/badge/Paper-Expert%20Systems%20with%20Applications-blue)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/datasets/ranjithkumaranandan/forestfire)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

> **A Novel Deep Learning Framework for Multi-Class Forest Fire and Smoke Detection Using Dual Attention and Multi-Scale Feature Fusion with Fire Spread Index Evaluation**
>
> Ranjith Kumar Anandan — Department of Computer Science and Engineering, SRM Institute of Science and Technology, Ramapuram, Chennai, India

---

## 📌 Overview

**FireSightNet** is an end-to-end deep learning framework for **four-class** forest fire and smoke detection, capable of simultaneously classifying:

| Class | Description |
|-------|-------------|
| 🟢 No Fire | Background / safe scenes |
| 🟡 Smoke | Early-stage smoke without visible flame |
| 🔴 Fire | Active fire without dense smoke |
| 🔴🟡 Fire+Smoke | Combined fire and smoke scenarios |

Most existing approaches treat fire detection as a **binary problem**. FireSightNet addresses the full fire development spectrum, enabling more actionable emergency response.

---

## 🏆 Key Results

| Metric | FireSightNet | Best Baseline |
|--------|-------------|---------------|
| Accuracy | **96.96%** | 95.67% (InceptionV3) |
| Precision | **97.06%** | 95.78% |
| F1-Score | **96.95%** | 95.67% |
| mAP@0.5 | **99.83%** | 99.53% |
| FSI | **96.89%** | 96.39% |
| Mean IoU | **94.21%** | 93.15% |

---

## 🧠 Architecture

FireSightNet integrates three complementary architectural innovations on top of an **EfficientNet-B4** backbone:

```
Input Image (224×224×3)
        ↓
EfficientNet-B4 Backbone (ImageNet pretrained)
        ↓
Multi-Head Transformer Self-Attention (8 heads)   ← Global context
        ↓
CBAM (Channel + Spatial Attention)                ← Local refinement
        ↓
Feature Pyramid Network (FPN)                     ← Multi-scale fusion
        ↓
Triple Pooling: GeM + GAP + GMP → 768-d          ← Rich representation
        ↓
FC layers + Focal CE Loss (γ=2, ε=0.1)
        ↓
4-Class Output
```

### Novel Components
- **Dual Attention**: Multi-Head Transformer Self-Attention + CBAM for global-to-local feature refinement
- **FPN**: Multi-scale feature fusion across fire densities and distances
- **Triple Pooling**: GeM + GAP + GMP concatenation producing a 768-dimensional feature vector
- **Fire Spread Index (FSI)**: Novel evaluation metric — geometric mean of per-class recall, penalizing any missed fire class

---

## 📊 Fire Spread Index (FSI)

FSI is a novel domain-specific metric introduced in this paper:

$$FSI = \left(\prod_{c=1}^{C} R_c\right)^{1/C}$$

where $R_c$ is the recall of class $c$. FSI → 0 if any single class recall → 0, ensuring safety-critical reliability that standard accuracy/F1 cannot guarantee.

---

## 📁 Repository Structure

```
firesightnet/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── firesightnet_config.yaml      # All hyperparameters
├── model/
│   └── firesightnet.py               # Full model architecture
├── train.py                          # Training script
├── evaluate.py                       # Evaluation + FSI computation
├── predict.py                        # Single image inference
├── ablation.py                       # Ablation study script
└── utils/
    ├── dataset.py                    # Dataset loading
    ├── augmentation.py               # Augmentation pipeline
    ├── metrics.py                    # FSI + all metrics
    └── gradcam.py                    # GradCAM visualization
```

---

## 🗃️ Dataset

The **FireSightNet 4-Class Dataset** is publicly available on Kaggle:

🔗 [https://www.kaggle.com/datasets/ranjithkumaranandan/forestfire](https://www.kaggle.com/datasets/ranjithkumaranandan/forestfire)

| Source Dataset | Images Used | Classes |
|---------------|-------------|---------|
| FLAME [28] | 1,900 | Fire, No Fire |
| ForestFireImages [29] | 5,050 | Fire, No Fire |
| D-Fire [30] | 999 | Fire, Smoke, No Fire |
| Forest Fire C4 [31] | 4,823 | All 4 classes |
| **Total (after augmentation)** | **24,000** | **4 classes (6,000 each)** |

- Resolution: 224×224×3
- License: CC BY 4.0
- Split: 70% train / 15% val / 15% test

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/ranjithkumaranandan/FireSightNet.git
cd FireSightNet
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
# Download from Kaggle
kaggle datasets download ranjithkumaranandan/forestfire
unzip forestfire.zip -d data/
```

### 4. Train
```bash
python train.py --config configs/firesightnet_config.yaml
```

### 5. Evaluate
```bash
python evaluate.py --checkpoint checkpoints/firesightnet_best.pth --data data/test
```

### 6. Predict on a single image
```bash
python predict.py --image path/to/image.jpg --checkpoint checkpoints/firesightnet_best.pth
```

---

## 📈 Ablation Study

| Variant | Components | Accuracy | F1-Score | Gain |
|---------|-----------|----------|----------|------|
| A | EfficientNet-B4 baseline | 95.09% | 95.08% | — |
| B | A + CBAM | 94.88% | 94.87% | −0.21% |
| C | A + Transformer | 95.17% | 95.15% | +0.08% |
| D | A + FPN | 94.93% | 94.90% | −0.16% |
| E | A + CBAM + FPN | 94.93% | 94.90% | −0.16% |
| F | A + Transformer + CBAM | 94.93% | 94.91% | −0.16% |
| **G (Proposed)** | **All components** | **97.42%** | **97.42%** | **+2.33%** |

> The plateau at D/E/F confirms that no two-component combination is sufficient — synergy requires architectural completeness.

---

## ⚙️ Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
grad-cam>=1.4.8
pyyaml>=6.0
tqdm>=4.65.0
```

---

## 📝 Citation

If you use FireSightNet or the dataset in your research, please cite:

```bibtex
@article{anandan2024firesightnet,
  title={A Novel Deep Learning Framework for Multi-Class Forest Fire and Smoke Detection Using Dual Attention and Multi-Scale Feature Fusion with Fire Spread Index Evaluation},
  author={Anandan, Ranjith Kumar},
  journal={Expert Systems with Applications},
  year={2024},
  publisher={Elsevier}
}
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- SRM Institute of Science and Technology, Ramapuram, Chennai
- FLAME, ForestFireImages, D-Fire, and Forest Fire C4 dataset creators
- PyTorch, timm, and EfficientNet communities
