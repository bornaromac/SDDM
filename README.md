# SDDM
# SDDM — Class-Conditional DDPM for Synthetic Dermatoscopic Images (HAM10000)

This repository contains a proof-of-concept pipeline for generating synthetic dermatoscopic images using a **class-conditional Denoising Diffusion Probabilistic Model (DDPM)** trained on **HAM10000**.  
In addition to training and sampling, the project includes a **post-hoc data analysis module**: export of real/synthetic embeddings, **UMAP** projection, and an **interactive D3.js visualization** with nearest-neighbor inspection to help assess distribution overlap and potential memorization.

## Project structure

- `.venv/` — local virtual environment *(ignored in Git)*
- `configs/` — experiment configuration (dataset, model, training, sampling parameters)
- `src/` — training, sampling, and export scripts
- `d3_viz/` — interactive visualization (HTML/CSS/JS) for UMAP + nearest-neighbor analysis
- `viz_export/` — exported artifacts for visualization (e.g., JSON files; optionally images depending on your setup)
- `data/` — dataset location (expected: `data/archive.zip`) *(ignored in Git)*
- `ckpts/` — checkpoints *(ignored in Git)*
- `runs/` — logs/outputs *(ignored in Git)*
- `debug_roundtrip/` — debug artifacts *(ignored in Git)*
- `requirements.txt` — Python dependencies

## Requirements

- Python **3.10+** (recommended 3.11)
- GPU environment recommended (Kaggle Notebooks used for training/sampling due to Colab GPU restrictions)
- Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
Install dependencies:

```bash
pip install -r requirements.txt
