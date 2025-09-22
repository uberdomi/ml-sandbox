"""
Base classes, information and shared functionality for dataset management.

This module contains the abstract base class ManagedDataset and common
utilities used by all dataset implementations.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Callable, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from .downloaders import DownloadInfo, download_and_extract_dataset

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
    - Provides get_dataloaders() method for train/val/test splits
    - Implements visualization methods for data exploration
    - Handles transformations appropriately for tensor data
    - Provides dataset information via print_info()
    """
    
    def __init__(self, 
                 root: Optional[Union[str, Path]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 force_download: bool = False):
        """
        Initialize managed dataset.

        Downloads and loads ALL available data (train + test) into a unified dataset.
        Use get_dataloaders() to split into train/val/test sets.
        
        Args:
            root: Root directory for data (defaults to project_root/data)
            transform: Transform to apply to samples
            target_transform: Transform to apply to targets
            force_download: Whether to force re-download
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
        
        self.root = Path(root)
        self.dataset_root = self.root / self.dataset_name
        self.transform = transform
        self.target_transform = target_transform
        
        # Create dataset directory
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Always download (but won't re-download if files exist unless force_download=True)
        self._download(force_download)
        
        # Load ALL data (train + test) into unified dataset
        self._load_data()
    
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
    
    @abstractmethod
    def _extraction_valid(self) -> bool:
        """Check if the extracted dataset is valid."""
        pass
    
    def _download(self, force_download: bool = False) -> None:
        """Download all dataset files (both train and test)."""
        if self._extraction_valid() and not force_download:
            print(f"Dataset {self.dataset_name} already exists and is valid.")
            return

        print(f"Downloading dataset {self.dataset_name} to {self.dataset_root}...")
        for info in self.download_infos:
            download_and_extract_dataset(info, self.dataset_root)
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load ALL dataset data (train + test) into memory as unified dataset."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the complete dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        pass
    
    def print_info(self) -> None:
        """Print information about this dataset."""
        info = self.dataset_info
        print(info.print())
        print(f"Total samples loaded: {len(self):,}")
    
    def get_dataloaders(self, 
                       train_split: float = 0.6,
                       val_split: float = 0.2, 
                       test_split: float = 0.2,
                       batch_size: int = 32,
                       shuffle: bool = True,
                       num_workers: int = 0) -> Dict[str, DataLoader]:
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
    
    def show_random_samples(self, num_samples: int = 8, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Display random samples from the dataset."""
        if len(self) == 0:
            print("Dataset is empty!")
            return
            
        # Select random indices
        indices = random.sample(range(len(self)), min(num_samples, len(self)))
        
        # Calculate grid size
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            img, target = self[idx]
            
            # Convert tensor to numpy for display
            if isinstance(img, torch.Tensor):
                # Handle different channel orders and shapes
                img_np = img.numpy()
                if len(img_np.shape) == 3:
                    if img_np.shape[0] in [1, 3]:  # CHW format
                        img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HWC
                    if img_np.shape[2] == 1:  # Grayscale
                        img_np = img_np.squeeze(2)
                else:
                    img_np = img_np.squeeze()  # Remove single dimensions
            else:
                img_np = img
            
            axes[i].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
            axes[i].set_title(f'Sample {idx}\nLabel: {target}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{self.dataset_info.name} - Random Samples')
        plt.tight_layout()
        plt.show()
    
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
            img, target = self[idx]
            if len(class_samples[target]) < num_per_class:
                class_samples[target].append((idx, img, target))
            
            # Stop if we have enough samples for all classes
            if all(len(samples) >= num_per_class for samples in class_samples.values()):
                break
        
        # Calculate grid size
        cols = num_classes
        rows = num_per_class
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for class_idx in range(num_classes):
            samples = class_samples[class_idx]
            class_name = info.classes[class_idx] if class_idx < len(info.classes) else str(class_idx)
            
            for sample_idx in range(num_per_class):
                row, col = sample_idx, class_idx
                ax = axes[row, col] if rows > 1 else axes[col]
                
                if sample_idx < len(samples):
                    idx, img, target = samples[sample_idx]
                    
                    # Convert tensor to numpy for display
                    if isinstance(img, torch.Tensor):
                        img_np = img.numpy()
                        if len(img_np.shape) == 3:
                            if img_np.shape[0] in [1, 3]:  # CHW format
                                img_np = np.transpose(img_np, (1, 2, 0))  # Convert to HWC
                            if img_np.shape[2] == 1:  # Grayscale
                                img_np = img_np.squeeze(2)
                        else:
                            img_np = img_np.squeeze()
                    else:
                        img_np = img
                    
                    ax.imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
                    if sample_idx == 0:  # Only show class name on first sample
                        ax.set_title(f'{class_name}\n({target})')
                    else:
                        ax.set_title(f'Sample {idx}')
                else:
                    # No sample available for this class
                    ax.text(0.5, 0.5, 'No sample', ha='center', va='center', transform=ax.transAxes)
                    if sample_idx == 0:
                        ax.set_title(f'{class_name}\n(No data)')
                
                ax.axis('off')
        
        plt.suptitle(f'{info.name} - Representative Samples by Class')
        plt.tight_layout()
        plt.show()
    
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