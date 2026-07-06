import os
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loaders
from models import Generator, Discriminator, init_weights
from utils import save_checkpoint, plot_loss_curves, save_sample_images

def train(config, ablation_mode, epochs=None):
    # Overwrite config epochs if CLI arg passed
    num_epochs = epochs if epochs is not None else config['training']['epochs']
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training Mode: {ablation_mode.upper()}")
    
    # Set seed for reproducibility
    torch.manual_seed(config['dataset']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['dataset']['seed'])
        
    # Get DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_data_loaders(
        root_dir=config['dataset']['root_dir'],
        image_size=config['dataset']['image_size'],
        batch_size=config['dataset']['batch_size'],
        train_split_ratio=config['dataset']['train_split_ratio'],
        seed=config['dataset']['seed'],
        num_workers=config['dataset']['num_workers']
    )
    
    # Initialize Generator
    gen = Generator(
        in_channels=config['model']['generator']['input_channels'],
        out_channels=config['model']['generator']['output_channels'],
        features=config['model']['generator']['init_features']
    ).to(device)
    gen.apply(init_weights)
    
    opt_g = optim.Adam(
        gen.parameters(),
        lr=config['training']['lr_g'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    # Initialize Discriminator (only for Pix2Pix)
    disc = None
    opt_d = None
    if ablation_mode == "pix2pix":
        disc = Discriminator(
            in_channels=config['model']['discriminator']['input_channels'],
            features=config['model']['discriminator']['init_features']
        ).to(device)
        disc.apply(init_weights)
        
        opt_d = optim.Adam(
            disc.parameters(),
            lr=config['training']['lr_d'],
            betas=(config['training']['beta1'], config['training']['beta2'])
        )
        
    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    
    # Training logs history
    history = {
        'train_g_loss': [],
        'train_d_loss': [],
        'val_g_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Training loop
        gen.train()
        if disc is not None:
            disc.train()
            
        epoch_g_losses = []
        epoch_d_losses = []
        
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (sar, eo) in enumerate(loop):
            sar = sar.to(device)
            eo = eo.to(device)
            
            # --- TRAIN GENERATOR (in l1_only baseline) OR GENERATOR + DISCRIMINATOR (in pix2pix) ---
            
            if ablation_mode == "l1_only":
                # Baseline U-Net with L1 loss only
                fake_eo = gen(sar)
                loss_g = criterion_l1(fake_eo, eo)
                
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                
                epoch_g_losses.append(loss_g.item())
                loop.set_description(f"G_L1: {loss_g.item():.4f}")
                
            elif ablation_mode == "pix2pix":
                # --- Train Discriminator ---
                fake_eo = gen(sar)
                
                # Discriminator loss for real images
                pred_real = disc(sar, eo)
                loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
                
                # Discriminator loss for generated/fake images
                pred_fake = disc(sar, fake_eo.detach())
                loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                
                # Combined D loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                
                epoch_d_losses.append(loss_d.item())
                
                # --- Train Generator ---
                pred_fake_g = disc(sar, fake_eo)
                # GAN loss (tricking the discriminator)
                loss_g_gan = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
                # L1 pixel-wise reconstruction loss
                loss_g_l1 = criterion_l1(fake_eo, eo)
                
                # Combined G loss
                loss_g = loss_g_gan + config['training']['lambda_l1'] * loss_g_l1
                
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                
                epoch_g_losses.append(loss_g.item())
                loop.set_description(f"D_loss: {loss_d.item():.4f} | G_loss: {loss_g.item():.4f}")
                
        # Average epoch training losses
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        history['train_g_loss'].append(avg_g_loss)
        
        if ablation_mode == "pix2pix":
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            history['train_d_loss'].append(avg_d_loss)
        else:
            history['train_d_loss'].append(0.0) # Placeholder
            
        # Validation loop
        if epoch % config['training']['val_interval'] == 0:
            gen.eval()
            val_losses = []
            
            with torch.no_grad():
                for sar, eo in val_loader:
                    sar = sar.to(device)
                    eo = eo.to(device)
                    
                    fake_eo = gen(sar)
                    val_l1 = criterion_l1(fake_eo, eo)
                    val_losses.append(val_l1.item())
                    
            avg_val_loss = sum(val_losses) / len(val_losses)
            history['val_g_loss'].append(avg_val_loss)
            print(f"Validation Generator L1 Loss: {avg_val_loss:.4f}")
            
            # Save sample generated outputs
            save_sample_images(gen, val_loader, epoch, device, os.path.join(save_dir, 'samples'))
            
            # Checkpoint saving
            # We track the best generator model based on validation reconstruction (L1) loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_name = f"{ablation_mode}_best.pth"
                save_checkpoint(gen, disc, opt_g, opt_d, epoch, os.path.join(save_dir, checkpoint_name))
                
        # Save last checkpoint
        checkpoint_last = f"{ablation_mode}_last.pth"
        save_checkpoint(gen, disc, opt_g, opt_d, epoch, os.path.join(save_dir, checkpoint_last))
        
    # Plot final loss curves
    plot_loss_curves(
        history, 
        os.path.join(save_dir, f"loss_curve_{ablation_mode}.png")
    )
    
    # Save loss values to CSV
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    df.index += 1
    df.to_csv(os.path.join(save_dir, f"loss_history_{ablation_mode}.csv"))
    print(f"Saved history CSV to {save_dir}")
    print("Training Completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAR-to-EO translation models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--ablation_mode", type=str, choices=["pix2pix", "l1_only"], default="pix2pix", 
                        help="Choose 'pix2pix' (full GAN) or 'l1_only' (U-Net with L1 loss).")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    args = parser.parse_args()
    
    # Read config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    train(config, args.ablation_mode, args.epochs)
