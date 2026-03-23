"""ViT patch features → HDF5 per image."""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from tqdm import tqdm
import argparse
from typing import List, Tuple

# Increase PIL's image size limit for high-resolution images
Image.MAX_IMAGE_PIXELS = None

# Model dictionary from MIL model
MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}


class ViTFeatureExtractor(nn.Module):
    """ViT-based feature extractor for patches"""
    
    def __init__(self, model_name: str = "vit-b16-224-in21k"):
        super().__init__()
        self.model_name = model_name
        
        try:
            model_path = MODEL_DICT[model_name]
        except KeyError:
            raise ValueError(f"{model_name} is not available. Choose from {list(MODEL_DICT.keys())}")
        
        # Load pretrained ViT model
        self.backbone = AutoModel.from_pretrained(model_path)
        config = self.backbone.config
        self.feature_dim = config.hidden_size
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, H, W] (batch of patches)
        Returns:
            features: Tensor of shape [B, feature_dim]
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x)
            # Use CLS token representation
            features = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
        
        return features


def extract_patches_with_coordinates(image: Image.Image, patch_size: int = 224, stride: int = 224) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
    """Extract patches from image with their coordinates"""
    w, h = image.size
    patches = []
    coordinates = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            coordinates.append((x, y))  # Top-left corner coordinates
    
    return patches, coordinates


def process_image(image_path: str, output_path: str, feature_extractor: ViTFeatureExtractor, 
                 patch_size: int = 224, stride: int = 224, batch_size: int = 32):
    """Process a single image and save embeddings to .h5 file"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Extract patches and coordinates
    patches, coordinates = extract_patches_with_coordinates(image, patch_size, stride)
    
    if len(patches) == 0:
        tqdm.write(f"Warning: No patches extracted from {image_path}")
        return
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Process patches in batches
    embeddings = []
    device = next(feature_extractor.parameters()).device
    
    # Progress bar for patches within this image
    patch_desc = f"Processing patches in {os.path.basename(image_path)}"
    patch_pbar = tqdm(range(0, len(patches), batch_size), desc=patch_desc, leave=False)
    
    for i in patch_pbar:
        batch_patches = patches[i:i + batch_size]
        
        # Transform patches
        batch_tensors = []
        for patch in batch_patches:
            patch_tensor = transform(patch)
            batch_tensors.append(patch_tensor)
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Extract features
        with torch.no_grad():
            batch_features = feature_extractor(batch_tensor)
            embeddings.append(batch_features.cpu().numpy())
        
        # Update patch progress
        patch_pbar.set_postfix({
            'patches': f"{min(i + batch_size, len(patches))}/{len(patches)}"
        })
    
    patch_pbar.close()
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)  # [num_patches, embedding_dim]
    coordinates = np.array(coordinates)  # [num_patches, 2]
    
    # Save to .h5 file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        f.create_dataset('coordinates', data=coordinates, compression='gzip')
        f.attrs['patch_size'] = patch_size
        f.attrs['stride'] = stride
        f.attrs['image_size'] = image.size
        f.attrs['num_patches'] = len(patches)
        f.attrs['embedding_dim'] = embeddings.shape[1]
    
    tqdm.write(f"Saved {len(patches)} patches from {os.path.basename(image_path)} to {output_path}")


def process_dataset(data_dir: str, model_name: str = "vit-b16-224-in21k", 
                   patch_size: int = 224, stride: int = 224, batch_size: int = 32):
    """Process entire dataset"""
    
    # Initialize feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = ViTFeatureExtractor(model_name).to(device)
    
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Patch size: {patch_size}, Stride: {stride}")
    
    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images with progress bar
    failed_count = 0
    skipped_count = 0
    
    for image_path in tqdm(image_files, desc="Processing images", unit="img"):
        # Create output path
        relative_path = os.path.relpath(image_path, data_dir)
        output_path = os.path.join(os.path.dirname(image_path),
                                   os.path.splitext(os.path.basename(relative_path))[0].replace('.png', '') + ".h5")
        
        # Skip if already processed
        if os.path.exists(output_path):
            skipped_count += 1
            tqdm.write(f"Skipping {os.path.basename(image_path)} (already processed)")
            continue
        
        try:
            process_image(image_path, output_path, feature_extractor, 
                        patch_size, stride, batch_size)
        except Exception as e:
            failed_count += 1
            tqdm.write(f"Error processing {image_path}: {e}")
    
    print(f"\nProcessing completed!")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {len(image_files) - skipped_count - failed_count}")
    print(f"Skipped (already processed): {skipped_count}")
    print(f"Failed: {failed_count}")


def main():
    parser = argparse.ArgumentParser(description="ViT patch features → HDF5 per image.")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Image directory (recursive)")
    parser.add_argument("--model_name", type=str, default="vit-b16-224-in21k",
                       choices=list(MODEL_DICT.keys()),
                       help="ViT model to use for feature extraction")
    parser.add_argument("--patch_size", type=int, default=224,
                       help="Size of patches to extract")
    parser.add_argument("--stride", type=int, default=224,
                       help="Stride for patch extraction")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing patches")
    
    args = parser.parse_args()
    
    process_dataset(
        data_dir=args.data_dir,
        model_name=args.model_name,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main() 