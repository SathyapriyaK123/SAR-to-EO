# SAR-to-EO Image Translation (GalaxEye Technical Assignment)

This repository contains the complete PyTorch implementation for translating Synthetic Aperture Radar (SAR) imagery into Electro-Optical (EO/RGB) optical imagery. This is an ill-posed cross-modal image translation problem because SAR lacks direct color/spectral information.

Our approach uses a conditional Generative Adversarial Network (cGAN) based on the **Pix2Pix** architecture. We also implement a baseline U-Net model trained with $L_1$ loss only to serve as a controlled ablation study.

---

## Requirements

The code is verified on **Python 3.10+**. All package dependencies are listed in [requirements.txt](requirements.txt):
* `torch` and `torchvision` (standard PyTorch libraries)
* `pyyaml` (for configuration parsing)
* `matplotlib` (for loss-curve plotting)
* `scikit-image` (for SSIM and PSNR calculations)
* `lpips` (for Perceptual Similarity evaluation)
* `pytorch-fid` (for Fréchet Inception Distance calculation)
* `tqdm` (progress logging)

To install dependencies locally or in a cloud environment:
```bash
pip install -r requirements.txt
```

---

## Environment Setup

### Local Setup (Virtual Environment)
To set up a local virtual environment:
```bash
# Create environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### Cloud Setup (Kaggle or Google Colab)
Since satellite datasets are extremely large, we recommend training on **Kaggle**. Kaggle provides 30 hours of free GPU (Nvidia T4) per week.
1. Create a new notebook on Kaggle.
2. In the right-hand panel, set the accelerator to **GPU T4 x1** or **GPU T4 x2**.
3. In the settings panel, ensure **Internet** is toggled **On**.
4. In the first cell of your notebook, run:
```bash
# Clone the repository
!git clone https://github.com/<your-username>/<your-repo-name>.git project
%cd project

# Install packages
!pip install -r requirements.txt
```

---

## Dataset Structure

This project is configured to run out-of-the-box with the **Sentinel 1&2 Image Pairs (SAR & Optical)** dataset on Kaggle (hosted as `requiemonk/sentinel12-image-pairs-segregated-by-terrain`).

### Adding the Dataset in Kaggle
In your Kaggle notebook, click **Add Data** in the right-hand sidebar. Search for `sentinel12-image-pairs-segregated-by-terrain` and click **Add**.

### Expected Directory Layout
The dataset path should be configured in `config.yaml`. The folder structure must match:
```text
root_dir/
├── agri/
│   ├── s1/   <- Sentinel-1 SAR patches (e.g. ROIs1970_fall_s1_11_p427.png)
│   └── s2/   <- Sentinel-2 Optical RGB patches (e.g. ROIs1970_fall_s1_11_p427.png)
├── barrenland/
│   ├── s1/
│   └── s2/
├── grassland/
│   ├── s1/
│   └── s2/
└── urban/
    ├── s1/
    └── s2/
```

### Preventing Spatial Data Leakage
Adjacent satellite patches often represent overlapping or near-duplicate frames. To prevent artificial inflation of performance metrics, `dataset.py` extracts the **ROI (Region of Interest)** ID from filenames (e.g. `ROIs1970`). Patches are grouped by their ROI, and splits are made at the ROI level rather than the patch level. This ensures validation and test scenes represent completely unseen geographical passes.

---

## Training

To run training, specify the configuration file and choose between the baseline and full model.

### 1. Run Ablation Baseline (U-Net with $L_1$ Loss Only)
```bash
python train.py --config config.yaml --ablation_mode l1_only --epochs 15
```

### 2. Run Full Pix2Pix Model (U-Net Generator + PatchGAN Discriminator)
```bash
python train.py --config config.yaml --ablation_mode pix2pix --epochs 15
```
Training logs, checkpoints (e.g. `outputs/pix2pix_best.pth`), loss curves, and validation image grids will be saved to the `outputs/` folder.

---

## Evaluation

To evaluate a trained checkpoint on the validation/test split and calculate metrics (**LPIPS**, **FID**, **SSIM**, **PSNR**):
```bash
python eval.py --config config.yaml --weights outputs/pix2pix_best.pth
```
This script:
1. Runs inference on the test split.
2. Computes **SSIM** and **PSNR** (pixel-level) and **LPIPS** (perceptual similarity).
3. Computes the **FID** score by generating Inception representations of real and fake test image folders.
4. Saves 10 side-by-side qualitative comparison triplets in `outputs/qualitative_triplets/`.

---

## Inference I/O Contract

To run inference on a custom directory of unseen SAR images conforming to the assignment's contract:
```bash
python infer.py --input_dir <path_to_sar_pngs> --output_dir <path_to_save_results> --weights <path_to_checkpoint.pth>
```

### Input/Output Requirements
*   **Input**: Directory containing single-channel Sentinel-1 SAR (VV) patches, 256x256, 8-bit PNG, dB-scaled and min-max normalized to `[0, 255]`.
*   **Output**: Directory with generated 256x256 RGB PNG images, using the exact same filenames.
*   **VRAM**: Runs on a single GPU with $\le 16\text{ GB}$ VRAM.
*   **Internet**: Operates entirely offline; weights are loaded locally.

---

## Model Weights

The weights of our trained models can be downloaded using the links below:
*   **Full Pix2Pix Model Checkpoint**: `[Insert public link here - e.g. Hugging Face/Google Drive]`
*   **Ablation Baseline ($L_1$ only) Checkpoint**: `[Insert public link here - e.g. Hugging Face/Google Drive]`

---

## Results

### Metrics Summary

| Model Config | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ (dB) |
| :--- | :---: | :---: | :---: | :---: |
| **Ablation Baseline ($L_1$ only)** | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| **Full Pix2Pix ($L_1$ + GAN)** | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |

### Qualitative Samples
Qualitative comparison grids (SAR Input $\rightarrow$ Generated Optical $\rightarrow$ Ground-Truth Optical) are saved in the `outputs/samples/` and `outputs/qualitative_triplets/` directories.

---

## Citation / References

1. **SEN1-2 Dataset**: Schmitt, M., Hughes, L. H., & Zhu, X. X. (2018). *SEN1-2 — A Dataset for Deep Learning in SAR-Optical Data Fusion*. arXiv preprint arXiv:1807.00248.
2. **Pix2Pix Paper**: Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks*. CVPR 2017.
3. **Kaggle Version**: Sentinel 1&2 Image Pairs (SAR & Optical), hosted by Kaggle user `requiemonk`.
