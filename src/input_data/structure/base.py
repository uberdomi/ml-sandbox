"""
Base classes, information and shared functionality for dataset management.

This module contains the abstract base class ManagedDataset and common
utilities used by all dataset implementations.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal, Optional, Union, Tuple, Callable, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ..downloaders import DownloadInfo, download_dataset, check_integrity
from .plots import plot_samples
from .storage import StorageStrategy, MemoryStorage, DiskStorage, HybridStorage

@dataclass
class DatasetInfo:
    """High-level information about a complete dataset including structure and metadata.
    
    Args:
        name: Name of the dataset
        description: Short description of the dataset
        classes: List of class names/labels
        num_classes: Number of classes
        input_shape: Shape of input data (e.g., (3, 32, 32) for CIFAR-10)
        license: License information (optional)
        citation: Citation information (optional)"""
    name: str
    description: str
    classes: list[str]
    num_classes: int
    input_shape: Tuple[int, ...]  # Shape of input data (e.g., (3, 32, 32) for CIFAR-10)
    license: str = ""
    citation: str = ""

    def print(self) -> str:
        """Return a string summary of the dataset information."""
        lines = [
            f"Dataset: {self.name}",
            f"  Description: {self.description}",
            f"  Number of Classes: {self.num_classes}",
            f"  Input Shape: {self.input_shape}",
            f"  Classes: {', '.join(self.classes)}",
        ]
        if self.license:
            lines.append(f"  License: {self.license}")
        if self.citation:
            lines.append(f"  Citation: {self.citation}")
        return "\n".join(lines)

class ManagedDataset(Dataset, ABC):
    """
    Base class for managed datasets that handle their own downloads and storage.
    
    Key features:
    - Downloads and combines ALL available data (train + test) into one unified dataset
    - Supports multiple storage strategies (memory, disk, hybrid)
    - Provides get_dataloaders() method for train/val/test splits
    - Implements visualization methods for data exploration
    """
    
    def __init__(self, 
                 root: Optional[Union[str, Path]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 force_download: bool = False,
                 storage_strategy: Literal["memory", "disk", "hybrid"] = "hybrid",
                 memory_threshold_mb: float = 500.0):
        """
        Initialize managed dataset.

        Args:
            root: Root directory for data (defaults to project_root/data)
            transform: Transform to apply to samples
            target_transform: Transform to apply to targets
            force_download: Whether to force re-download
            storage_strategy: Storage strategy ('memory', 'disk', 'hybrid')
            memory_threshold_mb: Memory threshold for hybrid strategy (in megabytes)
        """
        if root is None:
            # Find project root by looking for pyproject.toml
            current_path = Path(__file__).parent
            while current_path.parent != current_path:
                if (current_path / "pyproject.toml").exists():
                    root = current_path / "data"
                    break
                current_path = current_path.parent
            else:
                root = Path.cwd() / "data"
        
        root = Path(root)
        self.dataset_root = root / self.dataset_name
        self.transform = transform
        self.target_transform = target_transform
        
        # Create dataset directory
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage strategy
        self.storage = self._create_storage_strategy(storage_strategy, memory_threshold_mb)
        
        # Always download (but won't re-download if files exist unless force_download=True)
        self._download(force_download)
        
        # Load or prepare data using storage strategy
        if not self.storage.is_ready() or force_download:
            self._load_and_store_data()
    
    # --- Dataset properties ---

    @property
    @abstractmethod
    def download_infos(self) -> List[DownloadInfo]:
        """List of download links for the dataset."""
        pass
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset (used for folder name)."""
        pass
    
    @property
    @abstractmethod
    def dataset_info(self) -> DatasetInfo:
        """Get the DatasetInfo for this dataset."""
        pass
    
    def print_info(self) -> None:
        """Print information about this dataset."""
        info = self.dataset_info
        print(info.print())
        print(f"Total samples loaded: {len(self):,}")

    # --- Internal methods ---

    def _create_storage_strategy(self, strategy: Literal["hybrid", "memory", "disk"], memory_threshold_mb: float) -> StorageStrategy:
        """Create the appropriate storage strategy."""
        if strategy == "memory":
            return MemoryStorage(self.dataset_root)
        elif strategy == "disk":
            return DiskStorage(self.dataset_root)
        elif strategy == "hybrid":
            return HybridStorage(self.dataset_root, memory_threshold_mb)
        else:
            raise ValueError(f"Unknown storage strategy: {strategy}. Use 'memory', 'disk', or 'hybrid'.")
    
    def _load_and_store_data(self) -> None:
        """Load raw data and store using the chosen strategy."""
        print("Loading raw dataset data...")
        data, targets = self._load_raw_data()
        
        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_info': self.dataset_info,
            'total_samples': len(data)
        }
        
        self.storage.save_data(data, targets, metadata)
        print(f"Storage complete. Memory usage: {self.storage.get_memory_usage_mb():.1f} MB")
    
    @abstractmethod
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw data from downloaded files.
        
        Returns:
            Tuple of (data, targets) as numpy arrays
        """
        pass
    
    def _downloads_valid(self) -> bool:
        """Check if the downloaded files are complete and valid."""
        for info in self.download_infos:
            file_path = self.dataset_root / info.filename

            # Check the validity of the .gz files
            if not check_integrity(file_path, info.md5, info.sha256):
                print(f"File {info.filename} failed integrity check.")
                return False

        return True
    
    def _download(self, force_download: bool = False) -> None:
        """Download all dataset files (both train and test)."""
        if self._downloads_valid() and not force_download:
            print("="*5, f"Dataset '{self.dataset_name}' already exists and is valid.", "="*5)
            return

        print("="*5, f"Downloading dataset '{self.dataset_name}' to {self.dataset_root}...", "="*5)
        for info in self.download_infos:
            download_dataset(info, self.dataset_root, force_download=force_download)

    
    # --- Dataset interface methods ---
    
    def __len__(self) -> int:
        """Return the size of the complete dataset."""
        return self.storage.get_dataset_size()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        # Load sample using storage strategy
        img_data, target = self.storage.load_sample(index)
        
        # Convert to tensor
        if isinstance(img_data, np.ndarray):
            if img_data.dtype != np.float32:
                img_data = img_data.astype(np.float32)
            img_tensor = torch.from_numpy(img_data)
        else:
            img_tensor = img_data
        
        # Apply transforms if provided
        img_tensor, target = self._apply_transforms(img_tensor, target)
        
        return img_tensor, target

    def _apply_transforms(self, sample: torch.Tensor, target: int) -> Tuple[torch.Tensor, int]:
        """
        Apply transforms to sample and target.
        
        Note: Since datasets now return tensors by default, transforms should expect
        tensor inputs rather than PIL Images.
        """
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
    # --- Data loading methods ---

    def get_dataloaders(
        self, 
        train_split: float = 0.6,
        val_split: float = 0.2, 
        test_split: float = 0.2,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        """
        Create train/validation/test dataloaders from the unified dataset.
        
        Args:
            train_split: Fraction of data for training (0.0-1.0)
            val_split: Fraction of data for validation (0.0-1.0) 
            test_split: Fraction of data for testing (0.0-1.0)
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes for data loading
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoader objects (only for splits > 0.0).
        """
        # Validate splits
        total_split = train_split + val_split + test_split
        if not abs(total_split - 1.0) < 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
        
        dataset_size = len(self)
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size  # Remainder goes to test
        
        # Split the dataset
        splits = []
        split_sizes = []
        if train_size > 0:
            splits.append('train')
            split_sizes.append(train_size)
        if val_size > 0:
            splits.append('val') 
            split_sizes.append(val_size)
        if test_size > 0:
            splits.append('test')
            split_sizes.append(test_size)
        
        if not splits:
            return {}
            
        # Perform the split
        split_datasets = random_split(self, split_sizes)
        
        # Create dataloaders and return as dictionary
        dataloaders = {}
        for i, split_name in enumerate(splits):
            dataset = split_datasets[i]
            shuffle_this = shuffle if split_name == 'train' else False
            loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle_this,
                num_workers=num_workers
            )
            dataloaders[split_name] = loader
        
        return dataloaders
    
    # --- Visualization methods ---
    
    def show_random_samples(self, num_samples: int = 8, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Display random samples from the dataset."""
        if len(self) == 0:
            print("Dataset is empty!")
            return
            
        # Select random indices
        indices = random.sample(range(len(self)), min(num_samples, len(self)))
        
        image_list, label_list = [], []
        for idx in indices:
            img, target_idx = self[idx]
            image_list.append(img)
            label_list.append(self.dataset_info.classes[target_idx] if self.dataset_info.classes else str(target_idx))

        plot_samples(image_list, labels=label_list, suptitle=f"Random samples of the {self.dataset_info.name} dataset")
    
    
    def show_illustrative_samples(self, num_per_class: int = 1, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Display representative samples from each class."""
        if len(self) == 0:
            print("Dataset is empty!")
            return
            
        info = self.dataset_info
        num_classes = info.num_classes
        
        # Find samples for each class
        class_samples = {i: [] for i in range(num_classes)}
        
        # Collect samples for each class (stop when we have enough)
        for idx in range(len(self)):
            img, target_idx = self[idx]
            if len(class_samples[target_idx]) < num_per_class:
                class_samples[target_idx].append((idx, img, info.classes[target_idx] if info.classes else str(target_idx)))

            # Stop if we have enough samples for all classes
            if all(len(samples) >= num_per_class for samples in class_samples.values()):
                break

        # Prepare for plotting
        image_list, label_list = [], []
        for class_idx, samples in class_samples.items():
            for idx, img, target in samples:
                image_list.append(img)
                label_list.append(target)

        plot_samples(image_list, labels=label_list, suptitle=f"Illustrative samples of the {self.dataset_info.name} dataset")
