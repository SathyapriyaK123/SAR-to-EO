import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

def get_roi_id(filename):
    """
    Extracts the Region of Interest (ROI) identifier from a filename.
    Typically, filenames follow the structure 'ROIsXXXX_...' or 'ROI_XXXX_...'.
    By grouping by ROI, we prevent spatial data leakage between train and val splits.
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) > 0:
        return parts[0]
    return basename[:10]  # Fallback

class SARTargetDataset(Dataset):
    def __init__(self, pairs, image_size=256, is_train=True):
        """
        Args:
            pairs: List of dicts, each with 'sar_path' and 'eo_path' keys.
            image_size: Target size for resizing images.
            is_train: Boolean, whether to apply training augmentations.
        """
        self.pairs = pairs
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sar_path = pair['sar_path']
        eo_path = pair['eo_path']

        # Load SAR image in grayscale ('L')
        sar_img = Image.open(sar_path).convert('L')
        
        # Load EO image in RGB
        eo_img = Image.open(eo_path).convert('RGB')

        # Resize
        sar_img = TF.resize(sar_img, [self.image_size, self.image_size], antialias=True)
        eo_img = TF.resize(eo_img, [self.image_size, self.image_size], antialias=True)

        # Convert to Tensor (scales [0, 255] to [0.0, 1.0])
        sar_tensor = TF.to_tensor(sar_img)
        eo_tensor = TF.to_tensor(eo_img)

        # Apply Augmentations (Horizontal and Vertical Flips) during training
        if self.is_train:
            # Random horizontal flipping
            if random.random() > 0.5:
                sar_tensor = TF.hflip(sar_tensor)
                eo_tensor = TF.hflip(eo_tensor)
            
            # Random vertical flipping
            if random.random() > 0.5:
                sar_tensor = TF.vflip(sar_tensor)
                eo_tensor = TF.vflip(eo_tensor)

        # Normalize to [-1, 1] range (standard for GANs / Tanh output)
        sar_tensor = (sar_tensor - 0.5) / 0.5
        eo_tensor = (eo_tensor - 0.5) / 0.5

        return sar_tensor, eo_tensor


def get_data_loaders(root_dir, image_size=256, batch_size=16, train_split_ratio=0.8, seed=42, num_workers=2):
    """
    Scans the terrain-segregated directory, pairs up S1 and S2 images,
    splits them by ROI to prevent data leakage, and returns DataLoader instances.
    """
    # Categories in the requiemonk/sentinel12-image-pairs-segregated-by-terrain dataset
    categories = ['agri', 'barrenland', 'grassland', 'urban']
    
    all_pairs = []
    
    for cat in categories:
        s1_dir = os.path.join(root_dir, cat, 's1')
        s2_dir = os.path.join(root_dir, cat, 's2')
        
        if not (os.path.exists(s1_dir) and os.path.exists(s2_dir)):
            print(f"Warning: Directories for category '{cat}' not found. Skipping.")
            continue
            
        s1_files = glob.glob(os.path.join(s1_dir, '*.png'))
        
        for s1_path in s1_files:
            filename = os.path.basename(s1_path)
            # Find the corresponding S2 file by replacing '_s1_' with '_s2_' in the filename
            s2_filename = filename.replace('_s1_', '_s2_')
            s2_path = os.path.join(s2_dir, s2_filename)
            
            if os.path.exists(s2_path):
                all_pairs.append({
                    'sar_path': s1_path,
                    'eo_path': s2_path,
                    'category': cat,
                    'roi_id': get_roi_id(filename)
                })
                
    if len(all_pairs) == 0:
        raise ValueError(f"No matching S1/S2 image pairs found in root directory: {root_dir}")
        
    print(f"Total pairs discovered: {len(all_pairs)}")

    # Extract unique ROIs
    unique_rois = list(set([p['roi_id'] for p in all_pairs]))
    print(f"Found {len(unique_rois)} unique ROIs/Scenes.")
    
    # Shuffle unique ROIs deterministically
    unique_rois.sort()
    random.seed(seed)
    random.shuffle(unique_rois)
    
    # Split ROIs
    num_train_rois = int(len(unique_rois) * train_split_ratio)
    train_rois = set(unique_rois[:num_train_rois])
    val_rois = set(unique_rois[num_train_rois:])
    
    train_pairs = [p for p in all_pairs if p['roi_id'] in train_rois]
    val_pairs = [p for p in all_pairs if p['roi_id'] in val_rois]
    
    print(f"Train pairs: {len(train_pairs)} (from {len(train_rois)} ROIs)")
    print(f"Validation pairs: {len(val_pairs)} (from {len(val_rois)} ROIs)")

    # Create PyTorch datasets
    train_dataset = SARTargetDataset(train_pairs, image_size=image_size, is_train=True)
    val_dataset = SARTargetDataset(val_pairs, image_size=image_size, is_train=False)

    # Create PyTorch dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
