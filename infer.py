import os
import argparse
import glob
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from models import Generator
from utils import load_checkpoint, denormalize

def run_inference(input_dir, output_dir, weights_path):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Initialize Generator with configuration matching training
    # Note: U-Net takes 1 input channel (VV) and outputs 3 channels (RGB)
    gen = Generator(in_channels=1, out_channels=3, features=64).to(device)
    
    # 3. Load model weights
    # Setting model_d=None, opt_g=None, opt_d=None as we only need generator state
    load_checkpoint(weights_path, gen, device=device)
    gen.eval()
    
    # 4. Find all PNG files in the input directory
    input_paths = glob.glob(os.path.join(input_dir, '*.png'))
    if len(input_paths) == 0:
        print(f"Warning: No PNG images found in input directory: {input_dir}")
        return
        
    print(f"Running inference on {len(input_paths)} images...")
    
    with torch.no_grad():
        for path in input_paths:
            filename = os.path.basename(path)
            
            # Load input image
            # convert('L') ensures it is treated as single-channel grayscale (VV channel)
            img = Image.open(path).convert('L')
            
            # Ensure it is exactly 256x256
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.Resampling.BILINEAR)
                
            # Convert to PyTorch Tensor [0.0, 1.0]
            img_tensor = TF.to_tensor(img).to(device)
            
            # Normalize to [-1, 1] range matching training preprocessing
            img_tensor = (img_tensor - 0.5) / 0.5
            
            # Add batch dimension: [1, 1, 256, 256]
            img_tensor = img_tensor.unsqueeze(0)
            
            # Generate translated EO (RGB) image
            output_tensor = gen(img_tensor)
            
            # Denormalize output from [-1, 1] to [0.0, 1.0]
            output_denorm = denormalize(output_tensor).clamp(0.0, 1.0)
            
            # Remove batch dimension and permute to (H, W, C)
            output_np = output_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Convert to [0, 255] uint8 RGB image
            output_img = Image.fromarray((output_np * 255.0).astype('uint8'), 'RGB')
            
            # Save output with identical filename
            output_path = os.path.join(output_dir, filename)
            output_img.save(output_path)
            
    print(f"Inference completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for SAR-to-EO translation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input SAR patches.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated EO images.")
    parser.add_argument("--weights", type=str, required=True, help="Path to generator model weights (.pth).")
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir, args.weights)
