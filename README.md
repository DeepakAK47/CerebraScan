# CerebraScan

Brain tumor segmentation from MRI using U-Net (PyTorch) + Streamlit.

## Dataset
- LGG MRI Segmentation (Kaggle / TCGA)
- 110 patients, ~3,929 slices
- Image: (256,256,3) | Mask: (256,256) binary
- Channels: Pre-contrast, FLAIR, Post-contrast  
- Split: by patient (not slice)

## Stage 1:
- [x] Folders, venv, requirements, .gitignore

## Stage 2:
**Stage 2 — EDA**
- [x] Dataset loaded, shapes verified
- [x] 3 channels visualized
- [x] Mask overlay checked
- [x] Class balance: ~33.6% tumor / 65% empty
- [x] Tumor area distribution plotted
- [x] Loss choice: Dice + BCE

## stage 3:
