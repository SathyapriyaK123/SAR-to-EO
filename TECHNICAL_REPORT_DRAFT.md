# TECHNICAL ASSIGNMENT REPORT
## Satellite AI Research Intern: SAR-to-EO Image Translation
**Author:** Kanapuram Sathyapriya 
**Date:** July 2026  

---

## 1. Abstract
This report presents an end-to-end applied research project on Synthetic Aperture Radar (SAR) to Electro-Optical (EO) image translation. Using paired Sentinel-1 and Sentinel-2 satellite imagery, we translate single-channel, log-scaled SAR amplitude inputs into 3-channel optical RGB outputs. We design and implement a conditional Generative Adversarial Network (cGAN) based on the Pix2Pix framework (comprising a U-Net-256 Generator and a 70x70 PatchGAN Discriminator). To evaluate the impact of adversarial training, we conduct a controlled ablation study comparing the cGAN against a baseline U-Net trained strictly with pixel-wise $L_1$ reconstruction loss. 

While the $L_1$ baseline yields mathematically higher pixel-level metrics ($\text{SSIM} = 0.31989$, $\text{PSNR} = 13.7808\text{ dB}$), it produces structurally blurry, mean-seeking predictions. In contrast, the full Pix2Pix model significantly improves perceptual realism, reducing the Learned Perceptual Image Patch Similarity (LPIPS) from $0.76737$ to $0.67672$ and lowering the Fréchet Inception Distance (FID) from $375.0559$ to $339.2502$. This "pixel-vs-perceptual" gap reveals the fundamental information asymmetry between radar and optical sensors, demonstrating that adversarial loss is crucial for generating visually convincing textures.

---

## 2. Literature Survey
SAR-to-EO translation sits at the intersection of remote sensing, sensor fusion, and cross-modal image translation. SAR sensors utilize active microwave signals that penetrate clouds, smoke, and darkness, but the resulting imagery suffers from speckle noise, geometric distortions (layover, foreshortening), and lack of spectral/color info. EO sensors capture passive solar reflection in the visible spectrum (RGB) which is highly interpretable but limited by cloud cover.

Early methods for cross-modal satellite translation relied on traditional regression, radiometric normalization, or shallow dictionary learning. The state-of-the-art was revolutionized by Deep Convolutional Networks, specifically Image-to-Image translation models. Isola et al. (2017) introduced **Pix2Pix**, a conditional GAN that pairs reconstruction losses ($L_1$) with PatchGAN discriminators to penalize high-frequency artifacts. For unpaired cross-domain translation, CycleGAN (Zhu et al., 2017) utilized cycle-consistency constraints. 

In remote sensing literature, Schmitt et al. (2018) released the **SEN1-2** dataset, establishing a benchmark for deep learning-based SAR-optical data fusion. Subsequent work (e.g., Ley et al., 2020) demonstrated that while pure $L_1$ or $L_2$ losses converge to a statistical mean (creating blurry, "safe" predictions), adding adversarial loss forces the network to generate plausible high-frequency details (such as roads, urban blocks, and field boundaries). Recently, Denoising Diffusion Probabilistic Models (DDPMs) have shown promise for image translation, but they require substantial computational resources at inference time compared to feed-forward GAN generators. Our work builds upon the Pix2Pix baseline, adapting it to handle single-channel VV inputs and evaluating it using perceptual metrics (LPIPS, FID) alongside pixel metrics (SSIM, PSNR).

---

## 3. Methodology

### 3.1 Preprocessing and Normalization
1.  **SAR Inputs (Sentinel-1 VV)**: Images are loaded as single-channel grayscale tensors. The pixel amplitudes, already log/dB-scaled and min-max normalized to $[0, 255]$, are converted to PyTorch float tensors and scaled to $[-1, 1]$ via the transform $x_{\text{norm}} = (x / 127.5) - 1.0$.
2.  **Optical Targets (Sentinel-2 RGB)**: Images are loaded as 3-channel RGB tensors and normalized to $[-1, 1]$ to match the output activation of the generator's final layer.
3.  **Augmentations**: During training, horizontal and vertical flips are applied randomly and simultaneously to both the input and target tensors to improve generalization.

### 3.2 Spatial Data Leakage Prevention (ROI Splitting)
Because satellite patches are cropped from overlapping flight passes, adjacent patches are often near-duplicates. A naive random train/val split leads to severe spatial data leakage, where validation patches are nearly identical to training patches, artificially inflating performance scores.
To prevent this, we group patches by their **Region of Interest (ROI)** ID extracted from the file names (e.g., `ROIs1970`). We perform the split at the ROI level: the training set contains all patches from $80\%$ of unique ROIs, while the validation/test set holds out the remaining $20\%$ of unique ROIs. This guarantees that the evaluation is performed on completely unseen geographic regions.

### 3.3 Network Architectures
1.  **Generator**: We implement a U-Net-256 architecture. It contains 8 downsampling blocks (encoder) and 8 upsampling blocks (decoder) with skip connections. Skip connections concatenate the encoder's activation maps directly with the decoder's activation maps at each resolution level, preserving low-level spatial geometry (like shorelines and roads). The output layer uses a `Tanh` activation function.
2.  **Discriminator**: We use a 70x70 PatchGAN. It takes the concatenation of the input SAR image and the target/generated EO image (4 channels total) and outputs a grid of prediction logits. Each value in the grid represents whether a corresponding 70x70 patch in the image is real or fake. This focuses the discriminator on local texture and contrast rather than global shapes.

### 3.4 Loss Functions & Training Strategy
We optimize the model using two configurations:

#### Configuration A: Ablation Baseline ($L_1$ Only)
The generator is trained strictly to minimize the pixel-wise mean absolute error ($L_1$ loss) between the generated EO image $G(x)$ and the target EO image $y$:
$$\mathcal{L}_{L_1}(G) = \mathbb{E}_{x, y} \left[ \| y - G(x) \|_1 \right]$$

#### Configuration B: Full Pix2Pix cGAN
We optimize a minimax objective where the generator $G$ tries to minimize a combined loss, while the discriminator $D$ tries to maximize it:
$$\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{x, y} \left[ \log D(x, y) \right] + \mathbb{E}_{x} \left[ \log(1 - D(x, G(x))) \right]$$
$$\mathcal{L}_{\text{total}}(G) = \mathcal{L}_{\text{cGAN}}(G, D) + \lambda_{L_1} \mathcal{L}_{L_1}(G)$$
We set $\lambda_{L_1} = 100.0$ to balance the structural correctness of $L_1$ with the realistic texture generation of the GAN loss. Both models are trained for 15 epochs using the Adam optimizer with a learning rate of $0.0002$, $\beta_1=0.5$, and $\beta_2=0.999$.

---

## 4. Results

### 4.1 Quantitative Evaluation
We evaluate both models on the held-out validation split. Results are summarized below:

| Model Configuration | LPIPS ↓ (Primary) | FID ↓ (Primary) | SSIM ↑ (Secondary) | PSNR ↑ (Secondary) |
| :--- | :---: | :---: | :---: | :---: |
| **Ablation Baseline ($L_1$ only)** | 0.76737 | 375.0559 | **0.31989** | **13.7808 dB** |
| **Full Pix2Pix ($L_1$ + GAN)** | **0.67672** | **339.2502** | 0.13924 | 12.8365 dB |

### 4.2 Discussion of the Pixel-vs-Perceptual Gap
The quantitative results exhibit a classic phenomenon in generative modeling:
*   The **L1 Baseline** achieves higher SSIM ($0.31989$) and PSNR ($13.7808\text{ dB}$). Since $L_1$ loss penalizes the mathematical distance from the target, the network minimizes error by predicting the statistical median of the training distribution (a blurry, average representation). This results in high pixel-wise agreement but visually uninformative, blurry images.
*   The **Pix2Pix** model degrades in pixel-wise metrics (SSIM drops to $0.13924$, PSNR drops to $12.8365\text{ dB}$) but drastically improves perceptual metrics (**LPIPS** drops by $11.8\%$ to $0.67672$, **FID** drops by $35.8$ points to $339.2502$). The GAN loss penalizes blurriness, forcing the generator to create sharp textures and distinct color boundaries (e.g. painting urban structures grey and fields green). If a sharp boundary is shifted by even a single pixel, it receives a large pixel penalty (worse PSNR/SSIM), but it looks significantly more realistic to human eyes and deep feature extractors (better LPIPS/FID).

### 4.3 Training Convergence & Loss Curves
*   **L1 Baseline**: The validation $L_1$ loss steadily converged, dropping from `0.4031` (Epoch 1) to `0.3328` (Epoch 5). Beyond Epoch 5, validation loss fluctuated between `0.33` and `0.39`, indicating convergence.
*   **Pix2Pix**: The Discriminator loss converged quickly and stabilized between `0.20` and `0.50`, showing a healthy balance where neither generator nor discriminator dominated the game. The validation loss reached its minimum at Epoch 2 (`0.3634`) and fluctuated slightly thereafter, which is typical for adversarial setups due to the dynamic minimax objective.

---

## 5. Future Work
If joining GalaxEye as an intern, I would explore the following technical directions:
1.  **Dual-Channel Input Adaptation (VV + VH)**: Sentinel-1 provides both VV and VH polarizations. Extending the generator to accept a 2-channel input (VV + VH) would provide crucial structural and geometric context (as VH polarization behaves differently over vegetation and urban areas), likely improving translation quality.
2.  **Self-Supervised Pre-Training**: Pre-training the U-Net encoder on a large unlabeled set of SAR images using contrastive learning (e.g., SimCLR or MoCo) would build robust structural representations before fine-tuning on the paired EO dataset.
3.  **Diffusion Models (DDPM)**: Explore conditional diffusion translation (like Palette) as an alternative to GANs, evaluating whether the iterative denoising process produces sharper and more diverse terrain textures, albeit at the cost of slower inference.

---

## 6. Conclusion
This project successfully implemented a deep-learning pipeline for cross-modal SAR-to-EO image translation. We demonstrated that although pixel-wise losses ($L_1$) optimize numerical metrics like PSNR and SSIM, they generate blurry outputs that lack real-world utility. By integrating adversarial loss (cGAN), we generated images with significantly better perceptual realism, as validated by an 11.8% improvement in LPIPS and a 35.8-point drop in FID. Our ROI-based split strategy successfully prevented spatial leakage, proving the model generalizes to completely unseen scenes.

---

## 7. Time and Resource Log

*   **Machine Used for Training:** Kaggle Notebook, Free GPU Tier.
    *   **GPU Model:** Nvidia Tesla T4 (16 GB VRAM).
    *   **CPU / RAM:** Intel Xeon (2 vCPUs), 13 GB RAM.
*   **Training Time details:**
    *   *Baseline Model:* ~3 min 55 sec per epoch. Total training time: ~58 minutes.
    *   *Pix2Pix Model:* ~9 min 05 sec per epoch. Total training time: ~136 minutes.
    *   *Evaluation:* ~2 minutes per run.
    *   *Total wall-clock training & evaluation time:* ~3.5 hours.
*   **Activity Log Breakdown:**
    1.  *Data Exploration & Setup:* 2 hours (analyzing image formats, verifying folders on Kaggle, setting up data loaders).
    2.  *Literature Reading:* 3 hours (reading Pix2Pix and SEN1-2 dataset papers).
    3.  *Code Implementation:* 4 hours (writing dataset loaders, model blocks, training loops, evaluation metrics).
    4.  *Training & Troubleshooting:* 4 hours (setting up Kaggle notebook, fixing dataset name matching, running epochs).
    5.  *Report Writing & Documentation:* 3 hours (analyzing outputs, drafting report and README).
    *   **Total Time Spent:** 16 hours.
