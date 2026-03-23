"""MIL data module: HDF5 patch embeddings, fold CSVs, and PyTorch Lightning ``MILDataModule``."""

import os
import pandas as pd
import numpy as np
from functools import partial
from typing import Optional, Sequence, Tuple, List
import shutil
import uuid
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import h5py

# Increase PIL's image size limit for large BCR-map images (high-resolution MIL inputs)
Image.MAX_IMAGE_PIXELS = None


class MILEmbeddingDataset(Dataset):
    """Dataset for Multiple Instance Learning that loads pre-computed embeddings from .h5 files"""
    
    def __init__(self, 
                 root: str, 
                 max_patches: int = 10000,
                 random_sampling: bool = True,
                 **kwargs):
        """
        Args:
            root: Root directory containing class subdirectories with .h5 files
            max_patches: Maximum number of patches per image
            random_sampling: Whether to randomly sample patches if there are more than max_patches
        """
        self.root = root
        self.max_patches = max_patches
        self.random_sampling = random_sampling
        
        # Get all .h5 file paths and their corresponding labels
        self.samples = []
        self.class_to_idx = {}
        
        # Scan directories to get samples and class mapping
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for class_name in classes:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for item_name in os.listdir(class_dir):
                item_path = os.path.join(class_dir, item_name)
                if os.path.isfile(item_path):
                    # Check if there's a corresponding .h5 file
                    h5_path = os.path.join(class_dir, os.path.splitext(item_name)[0] + ".h5")
                    if os.path.exists(h5_path):
                        self.samples.append((h5_path, class_idx, item_path))
                    else:
                        print(f"Warning: No .h5 file found for {item_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        h5_path, label, original_path = self.samples[idx]
        
        # Load embeddings and coordinates from .h5 file
        with h5py.File(h5_path, 'r') as f:
            embeddings = f['embeddings'][:]  # [num_patches, embedding_dim]
            coordinates = f['coordinates'][:]  # [num_patches, 2]
            
        # Convert to tensors
        embeddings = torch.from_numpy(embeddings).float()
        coordinates = torch.from_numpy(coordinates).long()
        
        # Sample patches if there are too many
        num_patches = embeddings.shape[0]
        if num_patches > self.max_patches:
            if self.random_sampling:
                # Random sampling
                indices = torch.randperm(num_patches)[:self.max_patches]
            else:
                # Take first max_patches
                indices = torch.arange(self.max_patches)
            
            embeddings = embeddings[indices]
            coordinates = coordinates[indices]
        
        return embeddings, label, original_path


def mil_embedding_collate_fn(batch):
    """
    Custom collate function for MIL embedding datasets
    Returns:
        - List of embedding tensors (each element is [num_patches, embedding_dim])
        - Tensor of labels
        - List of image paths
    """
    embeddings_list, labels, paths = zip(*batch)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return list(embeddings_list), labels_tensor, list(paths)


class MILDataModule(pl.LightningDataModule):
    """Lightning Data Module for Multiple Instance Learning"""
    
    def __init__(
        self,
        root: str = "data/",
        datapath: str = '/home/dhkwon/IP/database/DPI_100/all_class/',
        fold: int = 0,
        data_folds: str = 'data_folds.csv',
        num_classes: Optional[int] = None,
        patch_size: int = 224,
        stride: Optional[int] = None,
        min_patches: int = 4,
        max_patches: int = 10000,
        random_sampling: bool = True,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        flip_prob: float = 0.5,
        rand_aug_n: int = 0,
        rand_aug_m: int = 9,
        erase_prob: float = 0.0,
        use_trivial_aug: bool = False,
        mean: Sequence = (0.5, 0.5, 0.5),
        std: Sequence = (0.5, 0.5, 0.5),
        batch_size: int = 8,  # Typically smaller for MIL
        workers: int = 4,
        **kwargs  # Accept additional keyword arguments and ignore them
    ):
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.datapath = datapath
        self.fold = fold
        self.patch_size = patch_size
        self.stride = stride
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.random_sampling = random_sampling
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.flip_prob = flip_prob
        self.rand_aug_n = rand_aug_n
        self.rand_aug_m = rand_aug_m
        self.erase_prob = erase_prob
        self.use_trivial_aug = use_trivial_aug
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.workers = workers
        self.unique_run_id = str(uuid.uuid4())
        self.data_folds = data_folds

        assert(num_classes is not None)
        self.num_classes = num_classes
        
        self.train_dataset_fn = partial(
            MILEmbeddingDataset, 
            root=os.path.join(self.root, self.unique_run_id, "train"),
            max_patches=self.max_patches,
            random_sampling=self.random_sampling
        )
        self.val_dataset_fn = partial(
            MILEmbeddingDataset, 
            root=os.path.join(self.root, self.unique_run_id, "val"),
            max_patches=self.max_patches,
            random_sampling=False  # Use grid sampling for validation for consistency
        )
        self.test_dataset_fn = partial(
            MILEmbeddingDataset, 
            root=os.path.join(self.root, self.unique_run_id, "test"),
            max_patches=self.max_patches,
            random_sampling=False  # Use grid sampling for test for consistency
        )
            
        print(f"Using MIL dataset from {self.datapath}")
        print(f"Patch size: {self.patch_size}, Stride: {self.stride}")
        print(f"Patches per image: {self.min_patches}-{self.max_patches}")

        # Transforms for patches
        self.transforms_train = transforms.Compose([
            transforms.RandomHorizontalFlip(self.flip_prob),
            transforms.TrivialAugmentWide() if self.use_trivial_aug 
            else transforms.RandAugment(self.rand_aug_n, self.rand_aug_m),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        self.transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        # Create directories
        os.makedirs(os.path.join(self.root, self.unique_run_id, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.unique_run_id, "val"), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.unique_run_id, "test"), exist_ok=True)

        # Load fold information
        # self.fold_df = pd.read_csv(os.path.join(self.datapath, self.data_folds))

        # edited to read from absolute path, preventing copy and pasting of fold file to all sub-directories in sampling analysis.
        self.fold_df = pd.read_csv(self.data_folds)
        self._setup_fold_data()
    
    def _setup_fold_data(self):
        """Create symbolic links for the current fold's train and validation data"""
        
        # Clear existing symlinks
        for split in ["train", "val"]:
            split_dir = os.path.join(self.root, self.unique_run_id, f"{split}")
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir)
            # Create class subdirectories
            for class_idx in range(self.num_classes):
                os.makedirs(os.path.join(split_dir, str(class_idx)), exist_ok=True)

        # Create new symlinks
        for _, row in self.fold_df.iterrows():
            # Determine if this sample belongs to train or val for current fold
            is_val = row['fold'] == self.fold
            split = "val" if is_val else "train"
            
            # Find the actual file that starts with the base path
            src_dir = os.path.dirname(os.path.join(self.datapath, row['image_path']))
            src_prefix = os.path.basename(os.path.join(self.datapath, row['image_path']))
            
            # for in-house datasets
            # src_prefix_exact = src_prefix + "_"

            # for Mal-ID datasets
            src_prefix_exact = src_prefix

            if 'house' in self.datapath:
                src_prefix_exact = src_prefix + "_"

            matching_files = [f for f in os.listdir(src_dir) if f.startswith(src_prefix_exact) and f.endswith('.h5')]
            
            # Fallback: data paths indicate sampled_square but actual files use sampled_abs_square
            if not matching_files and 'sampled_square' in src_prefix_exact:
                src_prefix_abs = src_prefix_exact.replace('sampled_square', 'sampled_abs_square')
                matching_files = [f for f in os.listdir(src_dir) if f.startswith(src_prefix_abs) and f.endswith('.h5')]
   
            if matching_files:
                src_path = os.path.join(src_dir, matching_files[0])
            else:
                src_path = os.path.join(self.datapath, row['image_path'])
                
            class_dir = os.path.join(self.root, self.unique_run_id, f"{split}", str(row['class']))
            dst_path = os.path.join(class_dir, os.path.basename(src_path))
    
            if os.path.exists(src_path):
                os.symlink(src_path, dst_path)
            else:
                print(f"Warning: Source file not found: {src_path}")

    def prepare_data(self):
        """Check if the data_folds.csv exists"""
        fold_file = os.path.join(self.datapath, self.data_folds)
        assert os.path.exists(fold_file), f"{self.data_folds} not found in {self.datapath}"

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = self.train_dataset_fn(transform=self.transforms_train)
            self.val_dataset = self.val_dataset_fn(transform=self.transforms_test)
        elif stage == "validate":
            self.val_dataset = self.val_dataset_fn(transform=self.transforms_test)
        elif stage == "test":
            self.test_dataset = self.test_dataset_fn(transform=self.transforms_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=mil_embedding_collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=mil_embedding_collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=mil_embedding_collate_fn,
        )
    
    def __del__(self):
        """Cleanup temporary directories on deletion"""
        # Check if attributes exist before accessing them
        if hasattr(self, 'root') and hasattr(self, 'unique_run_id'):
            for split in ["train", "val"]:
                split_dir = os.path.join(self.root, self.unique_run_id, split)
                if os.path.exists(split_dir):
                    shutil.rmtree(split_dir) 