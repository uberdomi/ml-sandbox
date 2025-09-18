"""
CIFAR-10 dataset implementation.

This module contains the CIFAR-10 dataset class along with its specific
download information and metadata definitions.
"""

import pickle
import tarfile
from pathlib import Path
import numpy as np
from typing import Tuple
from enum import Enum

import torch

from .base import ManagedDataset, DatasetDownloads, DatasetInfo
from .downloaders import download_dataset


# CIFAR-10-specific download information
class Cifar10Downloads(Enum):
    """Download information for CIFAR-10 dataset."""
    
    CIFAR10 = DatasetDownloads(
        name="CIFAR-10 Dataset",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        ],
        filename="cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a",
        description="CIFAR-10 dataset archive (60,000 32x32 color images)"
    )


# CIFAR-10 dataset information
CIFAR10_INFO = DatasetInfo(
    name="CIFAR-10",
    description="CIFAR-10 dataset of natural images (32x32 color images)",
    classes=["airplane", "automobile", "bird", "cat", "deer", 
             "dog", "frog", "horse", "ship", "truck"],
    num_classes=10,
    input_shape=(3, 32, 32),
    license="Unknown",
    citation="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images."
)


class Cifar10Dataset(ManagedDataset):
    """
    CIFAR-10 dataset with automatic download and management.
    
    Loads ALL CIFAR-10 data (train + test) into a unified dataset.
    Use get_dataloaders() to split into train/val/test sets.
    
    Returns:
        sample: torch.Tensor of shape (3, 32, 32) with values in [0, 1]
        target: int class label (0-9)
    """
    
    @property
    def dataset_name(self) -> str:
        return "cifar-10"
    
    @property
    def dataset_info(self) -> DatasetInfo:
        return CIFAR10_INFO
    
    def _download(self, force_download: bool = False) -> None:
        """Download and extract CIFAR-10 dataset."""
        download_dataset(Cifar10Downloads.CIFAR10.value, self.dataset_root, force_download)
        
        # Extract if needed
        extracted_dir = self.dataset_root / "cifar-10-batches-py"
        tar_path = self.dataset_root / "cifar-10-python.tar.gz"
        
        if not extracted_dir.exists() and tar_path.exists():
            print("Extracting CIFAR-10 archive...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.dataset_root)
    
    def _load_data(self) -> None:
        """Load ALL CIFAR-10 data (train + test) into unified dataset."""
        extracted_dir = self.dataset_root / "cifar-10-batches-py"
        
        if not extracted_dir.exists():
            raise FileNotFoundError(
                f"CIFAR-10 data directory not found at {extracted_dir}. "
                "Please ensure the dataset was downloaded and extracted properly."
            )
        
        all_data = []
        all_labels = []
        
        # Load training batches (data_batch_1 through data_batch_5)
        train_samples = 0
        for i in range(1, 6):
            batch_file = extracted_dir / f"data_batch_{i}"
            with open(batch_file, 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
                batch_data = batch_dict[b'data']
                batch_labels = batch_dict[b'labels']
                
                all_data.append(batch_data)
                all_labels.extend(batch_labels)
                train_samples += len(batch_data)
        
        # Load test batch
        test_file = extracted_dir / "test_batch"
        with open(test_file, 'rb') as f:
            test_dict = pickle.load(f, encoding='bytes')
            test_data = test_dict[b'data']
            test_labels = test_dict[b'labels']
            
            all_data.append(test_data)
            all_labels.extend(test_labels)
            test_samples = len(test_data)
        
        # Combine all data
        combined_data = np.concatenate(all_data, axis=0)
        
        # Reshape data from (N, 3072) to (N, 3, 32, 32)
        # CIFAR-10 data comes as flattened arrays where first 1024 entries are red channel,
        # next 1024 are green, and last 1024 are blue
        self.data = combined_data.reshape(-1, 3, 32, 32)
        self.targets = np.array(all_labels, dtype=np.int64)
        
        print(f"Loaded complete CIFAR-10 dataset: {len(self.data):,} samples "
              f"(train: {train_samples:,}, test: {test_samples:,})")
        
        # Load class names for reference
        meta_file = extracted_dir / "batches.meta"
        if meta_file.exists():
            with open(meta_file, 'rb') as f:
                meta_dict = pickle.load(f, encoding='bytes')
                self.class_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]
        else:
            self.class_names = CIFAR10_INFO.classes
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from CIFAR-10 dataset.
        
        Returns:
            sample: torch.Tensor of shape (3, 32, 32) with values in [0, 1]
            target: int class label (0-9)
        """
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to tensor and normalize (values 0-255 -> 0-1)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        
        # Apply transforms if provided
        img_tensor, target = self._apply_transforms(img_tensor, target)
        
        return img_tensor, target