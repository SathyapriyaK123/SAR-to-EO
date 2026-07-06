import os
import shutil
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Import metrics libraries
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
import lpips
from pytorch_fid import fid_score

from dataset import get_data_loaders
from models import Generator
from utils import load_checkpoint, denormalize

def evaluate(config, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # 1. Initialize DataLoader
    _, val_loader = get_data_loaders(
        root_dir=config['dataset']['root_dir'],
        image_size=config['dataset']['image_size'],
        batch_size=1, # batch size 1 is best for image-by-image evaluation
        train_split_ratio=config['dataset']['train_split_ratio'],
        seed=config['dataset']['seed'],
        num_workers=config['dataset']['num_workers']
    )
    
    # 2. Load Generator
    gen = Generator(
        in_channels=config['model']['generator']['input_channels'],
        out_channels=config['model']['generator']['output_channels'],
        features=config['model']['generator']['init_features']
    ).to(device)
    
    # Load weights
    load_checkpoint(weights_path, gen, device=device)
    gen.eval()
    
    # 3. Setup Temp Folders for FID computation
    save_dir = config['training']['save_dir']
    temp_real_dir = os.path.join(save_dir, 'eval_real')
    temp_gen_dir = os.path.join(save_dir, 'eval_gen')
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_gen_dir, exist_ok=True)
    
    # Setup LPIPS loss function (AlexNet-based is the standard)
    # LPIPS expects input tensors normalized to [-1, 1], which matches our generator output range!
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    lpips_vals = []
    ssim_vals = []
    psnr_vals = []
    
    # To save qualitative triplets
    triplet_count = 0
    triplets_save_dir = os.path.join(save_dir, 'qualitative_triplets')
    os.makedirs(triplets_save_dir, exist_ok=True)
    
    print("Running inference and calculating pixel-level metrics...")
    
    with torch.no_grad():
        for i, (sar, eo) in enumerate(tqdm(val_loader)):
            sar = sar.to(device)
            eo = eo.to(device)
            
            # Generate EO image
            gen_eo = gen(sar)
            
            # 1. Calculate LPIPS (tensors are already in [-1, 1])
            lpips_score = lpips_fn(gen_eo, eo).item()
            lpips_vals.append(lpips_score)
            
            # 2. Convert to numpy [0, 255] uint8 for SSIM, PSNR, and saving
            gen_eo_denorm = denormalize(gen_eo).clamp(0.0, 1.0)
            eo_denorm = denormalize(eo).clamp(0.0, 1.0)
            
            # Convert to numpy (channels last: HxWxC)
            gen_np = (gen_eo_denorm.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            eo_np = (eo_denorm.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            sar_np = (denormalize(sar).squeeze(0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
            
            # Calculate SSIM (multichannel=True)
            ssim_score = ssim_fn(eo_np, gen_np, channel_axis=2, data_range=255)
            ssim_vals.append(ssim_score)
            
            # Calculate PSNR
            psnr_score = psnr_fn(eo_np, gen_np, data_range=255)
            psnr_vals.append(psnr_score)
            
            # Save files for FID scoring (using standard 8-bit PNGs)
            real_img = Image.fromarray(eo_np)
            gen_img = Image.fromarray(gen_np)
            
            real_img.save(os.path.join(temp_real_dir, f"patch_{i:05d}.png"))
            gen_img.save(os.path.join(temp_gen_dir, f"patch_{i:05d}.png"))
            
            # Save qualitative triplets (SAR | Generated | Ground-Truth)
            if triplet_count < 10:
                # Concatenate horizontally
                sar_rgb = np.stack([sar_np] * 3, axis=2)
                triplet = np.hstack([sar_rgb, gen_np, eo_np])
                Image.fromarray(triplet).save(os.path.join(triplets_save_dir, f"triplet_{triplet_count:02d}.png"))
                triplet_count += 1
                
    # Calculate Average Metrics
    mean_lpips = np.mean(lpips_vals)
    mean_ssim = np.mean(ssim_vals)
    mean_psnr = np.mean(psnr_vals)
    
    print("\nComputing FID score (this runs real and generated directories through Inception)...")
    try:
        fid_val = fid_score.calculate_fid_given_paths(
            [temp_real_dir, temp_gen_dir],
            batch_size=16,
            device=device,
            dims=2048
        )
    except Exception as e:
        print(f"Error computing FID: {e}. Defaulting to NaN.")
        fid_val = float('nan')
        
    # Clean up temp FID dirs to save space
    shutil.rmtree(temp_real_dir)
    shutil.rmtree(temp_gen_dir)
    
    # Save metrics to a text file
    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== SAR-to-EO Translation Evaluation Results ===\n")
        f.write(f"Weights Evaluated: {weights_path}\n")
        f.write(f"Number of test patches: {len(val_loader)}\n\n")
        f.write(f"Primary Perceptual Metrics:\n")
        f.write(f"  LPIPS: {mean_lpips:.5f}\n")
        f.write(f"  FID:   {fid_val:.4f}\n\n")
        f.write(f"Secondary Pixel-level Metrics:\n")
        f.write(f"  SSIM:  {mean_ssim:.5f}\n")
        f.write(f"  PSNR:  {mean_psnr:.4f} dB\n")
        
    print("\n================ EVALUATION METRICS ================")
    print(f"Primary (Perceptual):")
    print(f"  LPIPS: {mean_lpips:.5f} (lower is better)")
    print(f"  FID:   {fid_val:.4f} (lower is better)")
    print(f"Secondary (Pixel-level):")
    print(f"  SSIM:  {mean_ssim:.5f} (higher is better)")
    print(f"  PSNR:  {mean_psnr:.4f} dB (higher is better)")
    print("====================================================")
    print(f"Qualitative triplets saved to: {triplets_save_dir}")
    print(f"Report saved to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained SAR-to-EO models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--weights", type=str, required=True, help="Path to generator model weights (.pth).")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate(config, args.weights)
