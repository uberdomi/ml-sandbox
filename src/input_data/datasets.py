"""
Managed dataset classes that derive from torch.utils.data.Dataset.
Each dataset manages its own download, storage, and data access.
"""

import os
import struct
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Callable, Any, Dict, List
from abc import ABC, abstractmethod
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from .enums import DatasetDownloadsEnum, DatasetInfoEnum, CommonDatasets, DatasetInfo
from .downloaders import download_dataset, extract_archive


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
    def dataset_name(self) -> str:
        """Name of the dataset (used for folder name)."""
        pass
    
    @property
    @abstractmethod
    def dataset_info(self) -> 'DatasetInfo':
        """Get the DatasetInfo for this dataset."""
        pass
    
    @abstractmethod
    def _download(self, force_download: bool = False) -> None:
        """Download all dataset files (both train and test).""" 
        pass
    
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
                       num_workers: int = 0) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
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
            Tuple of (train_loader, val_loader, test_loader). Any can be None if split is 0.0.
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
            return None, None, None
            
        # Perform the split
        split_datasets = random_split(self, split_sizes)
        
        # Create dataloaders
        loaders = []
        for i, split_name in enumerate(splits):
            dataset = split_datasets[i]
            shuffle_this = shuffle if split_name == 'train' else False
            loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle_this,
                num_workers=num_workers
            )
            loaders.append(loader)
        
        # Return in standard order: train, val, test
        train_loader = loaders[splits.index('train')] if 'train' in splits else None
        val_loader = loaders[splits.index('val')] if 'val' in splits else None  
        test_loader = loaders[splits.index('test')] if 'test' in splits else None
        
        return train_loader, val_loader, test_loader
    
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


class MnistDataset(ManagedDataset):
    """
    MNIST dataset with automatic download and management.
    
    Loads ALL MNIST data (train + test) into a unified dataset.
    Use get_dataloaders() to split into train/val/test sets.
    
    Returns:
        sample: torch.Tensor of shape (1, 28, 28) with values in [0, 1]
        target: int class label (0-9)
    """
    
    @property
    def dataset_name(self) -> str:
        return "mnist"
    
    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfoEnum.MNIST.value
    
    def _download(self, force_download: bool = False) -> None:
        """Download ALL MNIST dataset files (both train and test)."""
        files_to_download = [
            CommonDatasets.MNIST_TRAIN_IMAGES,
            CommonDatasets.MNIST_TRAIN_LABELS,
            CommonDatasets.MNIST_TEST_IMAGES,
            CommonDatasets.MNIST_TEST_LABELS
        ]
        
        # Download each file
        for dataset_info in files_to_download:
            download_dataset(dataset_info, self.dataset_root, force_download)
    
    def _load_data(self) -> None:
        """Load ALL MNIST data (train + test) into unified dataset."""
        all_images = []
        all_labels = []
        
        # Load training data
        train_images_path = self.dataset_root / "train-images-idx3-ubyte.gz"
        train_labels_path = self.dataset_root / "train-labels-idx1-ubyte.gz"
        
        with gzip.open(train_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8)
            train_images = train_images.reshape(num_images, rows, cols)
        
        with gzip.open(train_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        all_images.append(train_images)
        all_labels.append(train_labels)
        
        # Load test data
        test_images_path = self.dataset_root / "t10k-images-idx3-ubyte.gz"
        test_labels_path = self.dataset_root / "t10k-labels-idx1-ubyte.gz"
        
        with gzip.open(test_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            test_images = np.frombuffer(f.read(), dtype=np.uint8)
            test_images = test_images.reshape(num_images, rows, cols)
        
        with gzip.open(test_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        all_images.append(test_images)
        all_labels.append(test_labels)
        
        # Combine all data
        self.data = np.concatenate(all_images, axis=0)
        self.targets = np.concatenate(all_labels, axis=0)
        
        print(f"Loaded complete MNIST dataset: {len(self.data):,} samples "
              f"(train: {len(train_images):,}, test: {len(test_images):,})")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from MNIST dataset.
        
        Returns:
            sample: torch.Tensor of shape (1, 28, 28) with values in [0, 1]
            target: int class label (0-9)
        """
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to tensor directly (values 0-255 -> 0-1)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if provided
        img_tensor, target = self._apply_transforms(img_tensor, target)
        
        return img_tensor, target


class FashionMnistDataset(ManagedDataset):
    """
    Fashion-MNIST dataset with automatic download and management.
    
    Loads ALL Fashion-MNIST data (train + test) into a unified dataset.
    Use get_dataloaders() to split into train/val/test sets.
    
    Returns:
        sample: torch.Tensor of shape (1, 28, 28) with values in [0, 1]
        target: int class label (0-9)
    """
    
    @property
    def dataset_name(self) -> str:
        return "fashion-mnist"
    
    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfoEnum.FASHION_MNIST.value
    
    def _download(self, force_download: bool = False) -> None:
        """Download ALL Fashion-MNIST dataset files (both train and test)."""
        files_to_download = [
            CommonDatasets.FASHION_MNIST_TRAIN_IMAGES,
            CommonDatasets.FASHION_MNIST_TRAIN_LABELS,
            CommonDatasets.FASHION_MNIST_TEST_IMAGES,
            CommonDatasets.FASHION_MNIST_TEST_LABELS
        ]
        
        for dataset_info in files_to_download:
            download_dataset(dataset_info, self.dataset_root, force_download)
    
    def _load_data(self) -> None:
        """Load ALL Fashion-MNIST data (train + test) into unified dataset."""
        all_images = []
        all_labels = []
        
        # Load training data
        train_images_path = self.dataset_root / "train-images-idx3-ubyte.gz"
        train_labels_path = self.dataset_root / "train-labels-idx1-ubyte.gz"
        
        with gzip.open(train_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8)
            train_images = train_images.reshape(num_images, rows, cols)
        
        with gzip.open(train_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        all_images.append(train_images)
        all_labels.append(train_labels)
        
        # Load test data
        test_images_path = self.dataset_root / "t10k-images-idx3-ubyte.gz"
        test_labels_path = self.dataset_root / "t10k-labels-idx1-ubyte.gz"
        
        with gzip.open(test_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            test_images = np.frombuffer(f.read(), dtype=np.uint8)
            test_images = test_images.reshape(num_images, rows, cols)
        
        with gzip.open(test_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        all_images.append(test_images)
        all_labels.append(test_labels)
        
        # Combine all data
        self.data = np.concatenate(all_images, axis=0)
        self.targets = np.concatenate(all_labels, axis=0)
        
        print(f"Loaded complete Fashion-MNIST dataset: {len(self.data):,} samples "
              f"(train: {len(train_images):,}, test: {len(test_images):,})")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from Fashion-MNIST dataset.
        
        Returns:
            sample: torch.Tensor of shape (1, 28, 28) with values in [0, 1]
            target: int class label (0-9)
        """
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to tensor directly (values 0-255 -> 0-1)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if provided
        img_tensor, target = self._apply_transforms(img_tensor, target)
        
        return img_tensor, target


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
        return DatasetInfoEnum.CIFAR10.value
    
    def _download(self, force_download: bool = False) -> None:
        """Download and extract CIFAR-10 dataset."""
        archive_path = download_dataset(
            CommonDatasets.CIFAR10, self.dataset_root, force_download
        )
        
        # Extract if not already extracted
        extracted_dir = self.dataset_root / "cifar-10-batches-py"
        if not extracted_dir.exists() or force_download:
            extract_archive(archive_path, self.dataset_root, remove_finished=True)
    
    def _load_data(self) -> None:
        """Load ALL CIFAR-10 data (train + test) into unified dataset."""
        extracted_dir = self.dataset_root / "cifar-10-batches-py"
        
        all_data = []
        all_labels = []
        
        # Load training batches
        train_data_list = []
        train_labels_list = []
        
        for i in range(1, 6):  # data_batch_1 to data_batch_5
            batch_file = extracted_dir / f"data_batch_{i}"
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
                train_data_list.append(batch_data[b'data'])
                train_labels_list.extend(batch_data[b'labels'])
        
        train_data = np.concatenate(train_data_list, axis=0)
        train_labels = np.array(train_labels_list)
        
        all_data.append(train_data)
        all_labels.append(train_labels)
        
        # Load test batch
        test_file = extracted_dir / "test_batch"
        with open(test_file, 'rb') as f:
            test_data_dict = pickle.load(f, encoding='bytes')
            test_data = test_data_dict[b'data']
            test_labels = np.array(test_data_dict[b'labels'])
        
        all_data.append(test_data)
        all_labels.append(test_labels)
        
        # Combine all data
        self.data = np.concatenate(all_data, axis=0)
        self.targets = np.concatenate(all_labels, axis=0)
        
        # Reshape data from flat to (height, width, channels)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        print(f"Loaded complete CIFAR-10 dataset: {len(self.data):,} samples "
              f"(train: {len(train_data):,}, test: {len(test_data):,})")
    
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
        
        # Convert to tensor directly (values 0-255 -> 0-1)
        # CIFAR-10 images are (32, 32, 3), we need (3, 32, 32)
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)  # HWC -> CHW
        
        # Apply transforms if provided
        img_tensor, target = self._apply_transforms(img_tensor, target)
        
        return img_tensor, target