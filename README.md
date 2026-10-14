# CerebraScan

Brain tumor segmentation from MRI using **ResU-Net** (PyTorch) + **Streamlit** web app.

---

## 🎯 Project Overview

- **Goal:** Automatically segment brain tumors from MRI scans.
- **Model:** ResU-Net (ResNet34 encoder + U-Net decoder).
- **Framework:** PyTorch + segmentation-models-pytorch.
- **UI:** Streamlit.
- **Dataset:** LGG MRI Segmentation (Kaggle / TCGA).

---

## 📂 Project Structure

CerebraScan/
├── data/
│ ├── raw/ # dataset (optional sample)
│ └── splits/ # train.txt, val.txt, test.txt
├── models/
│ ├── unet.py # ResU-Net model
│ └── weights/
│ ├── best_model.pth # trained weights
│ └── curves.png # training curves
├── src/
│ ├── dataset.py # PyTorch Dataset + DataLoader
│ ├── utils.py # Loss + metrics
│ ├── train.py # training script
│ └── predict.py # inference (upcoming)
├── app/
│ └── streamlit_app.py # Streamlit UI (upcoming)
├── notebook/
│ └── eda.ipynb
├── venv/
├── .env
├── .gitignore
├── requirements.txt
└── README.md



---

## 📊 Dataset

- **Name:** LGG MRI Segmentation
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (TCGA LGG collection)
- **Patients:** 110
- **Slices:** ~3,929 image-mask pairs
- **Image shape:** (256, 256, 3)
- **Mask shape:** (256, 256) — binary (0 or 1)

### Image Channels
| Channel | Sequence |
|---|---|
| 1 | Pre-contrast |
| 2 | FLAIR |
| 3 | Post-contrast |

### File Naming


---

## ✅ Progress

### Stage 1 — Project Setup
- [x] Folder structure created
- [x] Virtual environment (`venv/`)
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `README.md`

### Stage 2 — Data Exploration (EDA)
- [x] Dataset loaded via `kagglehub`
- [x] 110 patient folders verified
- [x] Image shape confirmed `(256, 256, 3)`
- [x] Mask shape confirmed `(256, 256)`, binary
- [x] 3 channels visualized (Pre / FLAIR / Post)
- [x] Image + mask overlay verified
- [x] Class balance: **~33.6% tumor / ~66% empty**
- [x] Tumor area distribution plotted

**Key findings:**
| Item | Value |
|---|---|
| Image shape | (256, 256, 3) |
| Mask shape | (256, 256) |
| Mask values | 0 / 1 |
| Tumor slices | ~33.6% |
| Empty slices | ~66% |
| Best channel | FLAIR |
| Main issue | Class imbalance |
| Loss choice | Dice + BCE |
| Split strategy | By patient |

### Stage 3 — Patient-wise Data Splitting
- [x] 110 patients shuffled (seed=42)
- [x] Split: 80 train / 15 val / 15 test
- [x] No overlap check passed
- [x] Saved as `train.txt`, `val.txt`, `test.txt`

**Split stats:**
| Set | Patients | Slices |
|---|---|---|
| Train | 80 | ~2,800 |
| Val | 15 | ~550 |
| Test | 15 | ~550 |

### Stage 4 — Data Pipeline (`dataset.py`)
- [x] `build_pairs()` flattens folders → `(image, mask)` pairs
- [x] `BrainMRIDataset` custom PyTorch Dataset
- [x] Augmentations (flip, rotate, shift-scale) using albumentations
- [x] Normalize with ImageNet stats
- [x] `DataLoader` with batch size 16
- [x] Verified batch shapes: `(16, 3, 256, 256)` + `(16, 1, 256, 256)`

### Stage 5 — Model (`unet.py`)
- [x] Built ResU-Net via `segmentation-models-pytorch`
- [x] Encoder: **ResNet34** (ImageNet pretrained)
- [x] Decoder: U-Net style with skip connections
- [x] Input: (3, 256, 256) → Output: (1, 256, 256)
- [x] Total params: ~24.4M

### Stage 6 — Loss + Optimizer (`utils.py`)
- [x] Dice Loss
- [x] BCEWithLogitsLoss
- [x] Combined Loss (0.5 Dice + 0.5 BCE)
- [x] Metrics: Dice coefficient, IoU
- [x] Optimizer: Adam (lr=1e-4)
- [x] Scheduler: ReduceLROnPlateau

### Stage 7 — Training (`train.py`)
- [x] Trained on Kaggle GPU (T4 x2)
- [x] Epochs: 30
- [x] Batch size: 16
- [x] Saved best model by validation Dice → `best_model.pth`
- [x] Plotted training curves → `curves.png`
- [x] Best Val Dice: **0.6480** (updating as training continues)

---

## 🚧 Upcoming Stages

- [ ] Stage 8 — Inference (`predict.py`)
- [ ] Stage 9 — Streamlit web app
- [ ] Stage 10 — Deployment

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| DL Framework | PyTorch, segmentation-models-pytorch |
| Data | NumPy, Pandas, PIL, OpenCV |
| Augmentation | Albumentations |
| Visualization | Matplotlib |
| UI | Streamlit |
| Training | Kaggle Notebooks (free GPU) |

