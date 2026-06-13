# 🔍 Distorted Visual Sequence Pattern Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-✓-yellow?style=flat)
![IIT Roorkee](https://img.shields.io/badge/IIT-Roorkee-9C1A1A?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

### 🏛️ CIG AI/ML Open Challenge 2026 — IIT Roorkee
**Sameer Modi · Roll No. 23410030 · Integrated M.Tech, Geological Technology**

</div>

---

## 🎯 Objective

> We are given **grayscale images of heavily distorted text** (e.g. `AXU323`).
> Our job is to train a deep learning model that correctly reads the hidden text
> from noisy, blurry, overlapping-character images — essentially an **intelligent CAPTCHA solver**.

```
📷 Distorted Image  --->  🧠 CNN (sees features)  --->  📖 BiLSTM (reads sequence)  --->  ✅ CTC (text)
```

- 👁️ **CNN** : Detects edges, curves, and character stroke patterns
- 📖 **BiLSTM** : Reads feature columns left-to-right AND right-to-left for better context
- 🎯 **CTC** : Decodes output without needing to know exact character positions

---

## 🗺️ Project Roadmap

| # | Phase | Goal |
|---|-------|------|
| 1️⃣ | **Setup** | Install libraries, configure GPU |
| 2️⃣ | **Data Loading + EDA** | Understand the data, visualise images |
| 3️⃣ | **Preprocessing + Dataset** | Resize, augment, encode labels |
| 4️⃣ | **CRNN Model** | CNN + BiLSTM + CTC output layer |
| 5️⃣ | **Training** | Train, validate, save best model |
| 6️⃣ | **Evaluation + Submission** | Predict on test, export CSV |

---

## 🏗️ Model Architecture

Two models are implemented — both trained **entirely from scratch** on the provided dataset.

### 🔵 Model 4A — Custom CRNN (Baseline)

Built entirely from scratch: 4-stage CNN + BiLSTM + CTC.

```
Input (B, 1, 64, 256)
  --> CNN 4 stages  --> (64, B, 512)   time-first
  --> BiLSTM        --> (64, B, 512)
  --> FC + logsmax  --> (64, B, num_classes)
  --> CTC decode    --> predicted text
```

### 🟢 Model 4B — ResNet-18 Style CRNN ⭐ Selected

Uses the ResNet-18 **architecture** but with:
- `pretrained=False` → no ImageNet weights — **random initialisation**
- **NO layers frozen** → every layer trains from scratch
- **NO differential learning rate** → uniform LR for all layers

```
Input (B, 1, 64, 256)
  --> Channel Adapter   1ch -> 3ch  (learned 1x1 conv, random init)
  --> ResNet-18 backbone (conv1 + maxpool + layer1 + layer2)
      ALL weights random, ALL trainable
      Output: (B, 128, 8, 32)
  --> AdaptiveAvgPool(1, 64)         -> (B, 128, 1, 64)
  --> Channel Projection 128->512    -> (B, 512, 1, 64)
  --> Squeeze + Permute              -> (64, B, 512)
  --> BiLSTM                         -> (64, B, 512)
  --> FC + log_softmax               -> (64, B, num_classes)
  --> CTC decode                     -> predicted text
```

> ⚠️ **Competition Rule Compliance:** Standard architectures are allowed but pretrained weights are NOT.
> `pretrained=False` means the network starts with random numbers and must learn everything
> from the provided dataset alone. This is **fully rule-compliant**.

---

## ⚙️ Configuration

```python
IMAGE_HEIGHT = 64      # all images resized to this height
IMAGE_WIDTH  = 256     # IMAGE_WIDTH / 4 = 64 LSTM time-steps after CNN
BATCH_SIZE   = 64
VAL_SPLIT    = 0.10    # 10% of training data used for validation
NUM_WORKERS  = 2

NUM_EPOCHS   = 60      # from-scratch needs more epochs than fine-tuning
LR           = 1e-3    # uniform LR — no differential rate
SEED         = 42
```

---

## 🔤 Character Vocabulary (CharVocab)

The model works with **numbers, not text**. A lookup table maps every character to an integer:

```
'<B>' (blank) -> 0    ← REQUIRED by CTC loss
'2'           -> 1
'3'           -> 2
'A'           -> 3    ...and so on (sorted)
```

Two jobs — **not** perfect inverses:

```
encode()  :  "BU522X"  ->  [9, 26, 4, 1, 1, 29]   ← used DURING TRAINING
decode()  :  model output (64 time-steps)  ->  "BU522X"   ← used DURING PREDICTION
```

> **Note:** `decode()` collapses consecutive duplicates (CTC rule).
> `encode → decode` is NOT a round-trip for labels with repeated chars like `"522"`.
> This is correct CTC behaviour, not a bug.

---

## 🎛️ Augmentation Pipelines

| Pipeline | Transforms Applied |
|----------|-------------------|
| 🏋️ **Training** | Resize → Random blur (40%) → GaussNoise (35%) → ShiftScaleRotate (40%) → BrightnessContrast (30%) → Normalise → ToTensor |
| ✅ **Val / Test** | Resize → Normalise → ToTensor only |

```python
# Training augmentation (from notebook)
A.OneOf([
    A.GaussianBlur(blur_limit=(3, 5)),   # soft smooth blur
    A.MotionBlur(blur_limit=5),          # directional smear blur
    A.MedianBlur(blur_limit=3),          # removes salt & pepper noise
], p=0.4)
A.GaussNoise(var_limit=(5.0, 30.0), p=0.35)
A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=3, p=0.4)
A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)
A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0)
```

---

## 🚀 Training Setup

```python
MODEL_CHOICE = 'resnet'   # 'resnet' or 'custom'

# Optimizer — single uniform LR (no pretrained layers to protect)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

# Scheduler — OneCycleLR (warm-up then cosine decay)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS, pct_start=0.1, anneal_strategy='cos'
)

# Loss
ctc_loss_fn = nn.CTCLoss(blank=vocab.BLANK_IDX, zero_infinity=True)
```

| Model | `MODEL_CHOICE` | Notes |
|-------|---------------|-------|
| ResNet-18 CRNN | `'resnet'` | Higher capacity, random weights |
| Custom CRNN | `'custom'` | Fully custom CNN, random weights |

---

## 📊 Evaluation

### Metric — Character Error Rate (CER)

```
CER = sum(edit_distance(pred, truth)) / sum(len(truth))

CER = 0.0  →  perfect predictions
CER = 1.0  →  every character is wrong
```

### Error Categories (from notebook Step 6.4)

```python
same_length_wrong_chars  # char confusion errors
prediction_too_short     # missed characters
prediction_too_long      # hallucinated characters
```

### Visual Prediction Grid (Step 6.3)

```
🟢 Green border  = exact match
🟠 Orange border = right length, wrong chars
🔴 Red border    = wrong length
```

---

## 📁 Repository Structure

```
📦 distorted-text-recognition
 ┣ 📓 final_notebook.ipynb                   ← Complete 6-phase pipeline
 ┣ 📄 submission_SameerModi_23410030.csv     ← Test set predictions
 ┣ 📋 CIG_AI_ML_PS.pdf                       ← Original problem statement
 ┗ 📖 README.md                              ← You are here
```

---

## 🚀 How to Run

```python
# 1. Upload cig_ps.zip to Google Colab
# 2. Open final_notebook.ipynb
# 3. Runtime → Change runtime type → T4 GPU
# 4. Runtime → Run all

# Dataset extraction is handled automatically inside the notebook:
import zipfile
with zipfile.ZipFile('/content/cig_ps.zip', 'r') as z:
    z.extractall('/content/')
```

### Install Dependencies

```bash
pip install torch torchvision albumentations editdistance tqdm pandas matplotlib opencv-python
```

---

## 📤 Submission Format

```csv
image,prediction
test-0.png,AXU323
test-1.png,BT91KD
test-2.png,QWER12
```

**File name:** `submission_SameerModi_23410030.csv`

---

## 📚 Libraries Used

```python
import torch, torch.nn, torch.optim          # deep learning
import torchvision.models                     # ResNet-18 architecture
import albumentations                         # image augmentation
import editdistance                           # CER metric
import cv2                                    # image loading
import pandas, numpy, matplotlib              # data handling & plots
from tqdm import tqdm                         # progress bars
```

---

## 👤 About

| Field | Detail |
|-------|--------|
| **Name** | Sameer Modi |
| **Roll No.** | 23410030 |
| **Programme** | Integrated M.Tech — Geological Technology |
| **Institute** | IIT Roorkee |
| **Challenge** | CIG AI/ML Open Challenge 2026 |

---

<div align="center">


*Built with ❤️ using PyTorch · All weights trained from scratch · No pretrained weights used*

</div>
