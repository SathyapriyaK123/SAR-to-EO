import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def denormalize(tensor):
    """
    Denormalizes image tensor from range [-1, 1] to [0, 1].
    """
    return (tensor + 1.0) / 2.0

def save_checkpoint(model_g, model_d, opt_g, opt_d, epoch, filepath):
    """
    Saves a checkpoint containing model states and optimizer states.
    """
    state = {
        'epoch': epoch,
        'generator_state_dict': model_g.state_dict(),
        'generator_opt_state_dict': opt_g.state_dict(),
    }
    if model_d is not None and opt_d is not None:
        state['discriminator_state_dict'] = model_d.state_dict()
        state['discriminator_opt_state_dict'] = opt_d.state_dict()
        
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model_g, model_d=None, opt_g=None, opt_d=None, device='cuda'):
    """
    Loads a checkpoint and updates model and optimizer states.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device)
    model_g.load_state_dict(checkpoint['generator_state_dict'])
    if opt_g is not None:
        opt_g.load_state_dict(checkpoint['generator_opt_state_dict'])
        
    if model_d is not None and 'discriminator_state_dict' in checkpoint:
        model_d.load_state_dict(checkpoint['discriminator_state_dict'])
        if opt_d is not None:
            opt_d.load_state_dict(checkpoint['discriminator_opt_state_dict'])
            
    print(f"Checkpoint loaded from {filepath} (Epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint.get('epoch', 0)

def plot_loss_curves(history, save_path):
    """
    Plots the training and validation curves.
    Supports both GAN (G & D losses) and L1-only baseline training.
    """
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(history['train_g_loss']) + 1)
    
    # Plot Generator Losses
    plt.plot(epochs, history['train_g_loss'], label='Train Gen Loss', color='blue', linestyle='-')
    if 'val_g_loss' in history and len(history['val_g_loss']) > 0:
        plt.plot(epochs, history['val_g_loss'], label='Val Gen Loss', color='cyan', linestyle='--')
        
    # Plot Discriminator Losses (if present)
    if 'train_d_loss' in history and len(history['train_d_loss']) > 0:
        plt.plot(epochs, history['train_d_loss'], label='Train Disc Loss', color='red', linestyle='-')
        
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")

def save_sample_images(generator, val_loader, epoch, device, save_dir, num_samples=5):
    """
    Generates predictions on validation samples and saves a grid showing
    SAR Input (grayscale) | Generated EO (RGB) | Ground-Truth EO (RGB).
    """
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    
    # Get a batch
    with torch.no_grad():
        sar_batch, eo_batch = next(iter(val_loader))
        sar_batch = sar_batch[:num_samples].to(device)
        eo_batch = eo_batch[:num_samples].to(device)
        
        # Generate translations
        generated_eo = generator(sar_batch)
        
        # Denormalize images to [0, 1] range for visualization
        sar_denorm = denormalize(sar_batch)
        eo_denorm = denormalize(eo_batch)
        gen_denorm = denormalize(generated_eo)
        
        # Convert SAR single-channel to 3-channel (RGB) by repeating channels
        sar_rgb = sar_denorm.repeat(1, 3, 1, 1)
        
        # List to hold the columns
        comparison_images = []
        for i in range(num_samples):
            # Concatenate horizontally: SAR, Generated, Target
            row = torch.cat([sar_rgb[i], gen_denorm[i], eo_denorm[i]], dim=2)
            comparison_images.append(row)
            
        # Stack vertically
        grid = torch.stack(comparison_images, dim=0)
        grid_image = vutils.make_grid(grid, nrow=1, padding=2, normalize=False)
        
        # Save grid image
        save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_val_samples.png")
        vutils.save_image(grid_image, save_path)
        print(f"Saved validation samples comparison to {save_path}")
        
    generator.train()
