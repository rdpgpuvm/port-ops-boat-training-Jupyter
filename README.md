# ⛵ Port Operations Boat Type Classification — Jupyter Notebook

Transfer Learning CNN (MobileNetV2) for automated vessel classification at port facilities.  
This repo contains the **commented Jupyter training notebook** — no Flask/Heroku deployment.  
All results, confusion matrices, sensitivity plots, and per-class metrics are displayed inline.

---

## 📓 Notebook

**`port_ops_boat_classification.ipynb`**

### Sections Covered

| # | Section | Description |
|---|---------|-------------|
| 1 | Install & Import | Libraries, TF version check, reproducibility seeds |
| 2 | Configuration | Image size, class names, paths, batch size |
| 3 | Class Distribution | Bar chart + pie chart of per-class image counts |
| 4 | Class Weights | Formula + rationale for compensating 24.3:1 imbalance |
| 5 | Data Pipeline | `image_dataset_from_directory`, augmentation layers, prefetch |
| 6 | Augmentation Preview | Side-by-side original vs augmented sample images |
| 7 | Build Model | MobileNetV2 backbone + custom Dense/BN/Dropout head |
| 8 | Phase 1 Training | Feature extraction — frozen backbone, lr=1e-3 |
| 9 | Phase 2 Training | Fine-tuning — top 20 layers unfrozen, lr=1e-4 |
| 10 | Evaluation | Top-1/Top-3 accuracy, classification report |
| 11 | Confusion Matrix | Normalised heatmap + per-class accuracy bar chart |
| 12 | Training Curves | Loss & accuracy for both phases |
| 13 | Confidence Analysis | Correct vs incorrect confidence distributions |
| 14 | Sensitivity Tests | Gaussian noise (σ=0–0.4) + brightness (×0.3–×2.0) |
| 15 | Save Metrics | metrics.json + model.keras |

---

## 🚢 Dataset — 9 Vessel Classes

| Class | Images | Imbalance vs Sailboat |
|-------|--------|-----------------------|
| sailboat | 389 | 1.0× (reference) |
| kayak | 203 | 1.9× |
| gondola | 193 | 2.0× |
| cruise_ship | 191 | 2.0× |
| ferry_boat | 63 | 6.2× |
| buoy | 53 | 7.3× |
| paper_boat | 31 | 12.5× |
| freight_boat | 23 | 16.9× |
| inflatable_boat | 16 | **24.3×** (rarest) |

**Total: 1,162 images**

Imbalance is handled via **class-weighted loss** (not SMOTE — SMOTE is designed for tabular
feature vectors, not image pixels; augmentation is used instead for image variety).

---

## 🧠 Model Architecture

```
Input (96×96×3)
    │
    ▼
MobileNetV2 backbone          ← pretrained on ImageNet (1.28M images)
(frozen in Phase 1,           ← 155 layers, ~2.25M parameters
 top 20 unfrozen in Phase 2)
    │
    ▼
GlobalAveragePooling2D        ← collapses H×W×C feature maps → flat vector (C,)
    │                            avoids Flatten() which would cause 32k-dim overfitting
    ▼
Dense(256) + BatchNorm + ReLU + Dropout(0.5)   ← custom classification head
    │
    ▼
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Dense(9, Softmax)             ← one probability per vessel class
```

### Architecture Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **Backbone** | MobileNetV2 | Lightweight, optimised for edge/embedded deployment at port |
| **Pooling** | GlobalAveragePooling2D | Less prone to overfitting vs Flatten on small datasets |
| **Dropout** | 0.5 → 0.3 | Heavier near input of head; lighter near output |
| **L2 Regularisation** | λ=1e-4 | Penalises large weights in classification head |
| **Imbalance** | Class weights | More appropriate than SMOTE for image data |

---

## ⚙️ Training Configuration

### Two-Phase Transfer Learning

| | Phase 1: Feature Extraction | Phase 2: Fine-Tuning |
|--|----------------------------|---------------------|
| **Backbone** | Fully frozen | Last 20 layers unfrozen |
| **Learning Rate** | 1e-3 | 1e-4 (10× smaller) |
| **Max Epochs** | 25 | 35 |
| **Early Stopping patience** | 8 epochs | 10 epochs |
| **Purpose** | Stabilise new head | Adapt high-level features to vessel domain |

### Why Two Phases?
The classification head starts with **random weights**. If we unfreeze the backbone immediately,
the large random gradients would corrupt the carefully pretrained ImageNet weights.
Phase 1 stabilises the head first; Phase 2 then gently nudges the backbone.

### Data Augmentation (Training Only)

| Transform | Range | Rationale |
|-----------|-------|-----------|
| RandomFlip | horizontal | Vessels look the same mirrored |
| RandomRotation | ±20% | Port cameras may be angled |
| RandomZoom | ±20% | Varying distance from camera |
| RandomContrast | ±20% | Lighting variation |
| RandomBrightness | ±15% | Time of day / weather |
| RandomTranslation | ±10% | Vessel not always centred in frame |

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| **Top-1 Val Accuracy** | ~78–82% |
| **Top-3 Val Accuracy** | ~93–96% |
| **Mean Confidence (correct)** | ~0.80+ |
| **Noise robustness (σ=0.2)** | ≤15% accuracy drop |

### Per-Class Difficulty
- **Easiest:** `sailboat`, `kayak`, `gondola` — many samples, visually distinct
- **Hardest:** `inflatable_boat`, `freight_boat` — fewest samples (16 and 23 images)
- **Common confusions:** `ferry_boat` ↔ `cruise_ship` (similar hull shape)

---

## ▶️ Run on Kaggle

Upload `port_ops_boat_classification.ipynb` to a Kaggle notebook with:
- Dataset: `bitbot1/boat-type-classification` (boat_type_classification_dataset folder)
- **GPU T4 accelerator recommended** — MobileNetV2 fine-tuning is significantly faster with GPU
- Runtime: ~10–15 min on GPU T4

Dataset path in notebook: `/kaggle/input/boat-type-classification/boat_type_classification_dataset`

---

## 💻 Run Locally

```bash
git clone https://github.com/rdpgpuvm/port-ops-boat-training-Jupyter.git
cd port-ops-boat-training-Jupyter

pip install -r requirements.txt
jupyter notebook port_ops_boat_classification.ipynb
```

Update `DATA_DIR` in the Configuration cell to point to your local dataset folder.

---

## 📁 Project Structure

```
port-ops-boat-training-Jupyter/
├── port_ops_boat_classification.ipynb   # Main Jupyter training notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

Output files generated after running (saved to /kaggle/working/model/ on Kaggle):
```
model/
├── model.keras          # Best model weights (Phase 2)
├── phase1_best.keras    # Best model weights (Phase 1)
└── metrics.json         # Full evaluation metrics JSON
```

---
