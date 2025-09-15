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

import torch
from torch.utils.data import Dataset
from PIL import Image

from .enums import CommonDatasets, DatasetInfo
from .downloaders import download_dataset, extract_archive


class ManagedDataset(Dataset, ABC):
    """
    Base class for managed datasets that handle their own downloads and storage.
    
    Each dataset:
    - Downloads itself to a fixed location under project_root/data/dataset_name
    - Implements __len__ and __getitem__ from PyTorch Dataset
    - Handles transformations appropriately for its structure
    - Provides visualization functionality
    """
    
    def __init__(self, 
                 root: Optional[Union[str, Path]] = None,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True,
                 force_download: bool = False):
        """
        Initialize managed dataset.
        
        Args:
            root: Root directory for data (defaults to project_root/data)
            train: Whether to use training or test set
            transform: Transform to apply to samples
            target_transform: Transform to apply to targets
            download: Whether to download if not present
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
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # Create dataset directory
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        if download:
            self._download(force_download)
        
        # Load data after download
        self._load_data()
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset (used for folder name)."""
        pass
    
    @abstractmethod
    def _download(self, force_download: bool = False) -> None:
        """Download the dataset files."""
        pass
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset into memory after download."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a sample from the dataset."""
        pass
    
    @abstractmethod
    def visualize(self, num_samples: int = 8, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize samples from the dataset."""
        pass
    
    def _apply_transforms(self, sample: Any, target: Any) -> Tuple[Any, Any]:
        """Apply transforms to sample and target."""
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class MnistDataset(ManagedDataset):
    """
    MNIST dataset with automatic download and management.
    
    Returns:
        sample: PIL Image of shape (28, 28)
        target: int class label (0-9)
    """
    
    @property
    def dataset_name(self) -> str:
        return "mnist"
    
    def _download(self, force_download: bool = False) -> None:
        """Download MNIST dataset files."""
        # Determine which files to download based on train/test
        if self.train:
            files_to_download = [
                CommonDatasets.MNIST_TRAIN_IMAGES,
                CommonDatasets.MNIST_TRAIN_LABELS
            ]
        else:
            files_to_download = [
                CommonDatasets.MNIST_TEST_IMAGES,
                CommonDatasets.MNIST_TEST_LABELS
            ]
        
        # Download each file
        for dataset_info in files_to_download:
            download_dataset(dataset_info, self.dataset_root, force_download)
    
    def _load_data(self) -> None:
        """Load MNIST data from downloaded files."""
        if self.train:
            images_file = "train-images-idx3-ubyte.gz"
            labels_file = "train-labels-idx1-ubyte.gz"
        else:
            images_file = "t10k-images-idx3-ubyte.gz"
            labels_file = "t10k-labels-idx1-ubyte.gz"
        
        # Load images
        images_path = self.dataset_root / images_file
        with gzip.open(images_path, 'rb') as f:
            # Read header
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        
        # Load labels
        labels_path = self.dataset_root / labels_file
        with gzip.open(labels_path, 'rb') as f:
            # Read header
            magic, num_labels = struct.unpack('>II', f.read(8))
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        self.data = images
        self.targets = labels
        
        print(f"Loaded MNIST {'training' if self.train else 'test'} set: "
              f"{len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Get a sample from MNIST dataset.
        
        Returns:
            sample: PIL Image of the digit
            target: int class label (0-9)
        """
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to PIL Image (L mode auto-detected for uint8 grayscale)
        img = Image.fromarray(img)
        
        # Apply transforms
        img, target = self._apply_transforms(img, target)
        
        return img, target
    
    def visualize(self, num_samples: int = 8, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize MNIST samples."""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        indices = np.random.choice(len(self), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img, target = self[idx]
            
            # Convert PIL Image back to numpy for visualization
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            
            axes[i].imshow(img_array, cmap='gray')
            axes[i].set_title(f'Label: {target}')
            axes[i].axis('off')
        
        plt.suptitle(f'MNIST {"Training" if self.train else "Test"} Samples')
        plt.tight_layout()
        plt.show()


class FashionMnistDataset(ManagedDataset):
    """
    Fashion-MNIST dataset with automatic download and management.
    
    Returns:
        sample: PIL Image of shape (28, 28)
        target: int class label (0-9)
    """
    
    # Fashion-MNIST class names
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "fashion-mnist"
    
    def _download(self, force_download: bool = False) -> None:
        """Download Fashion-MNIST dataset files."""
        if self.train:
            files_to_download = [
                CommonDatasets.FASHION_MNIST_TRAIN_IMAGES,
                CommonDatasets.FASHION_MNIST_TRAIN_LABELS
            ]
        else:
            files_to_download = [
                CommonDatasets.FASHION_MNIST_TEST_IMAGES,
                CommonDatasets.FASHION_MNIST_TEST_LABELS
            ]
        
        for dataset_info in files_to_download:
            download_dataset(dataset_info, self.dataset_root, force_download)
    
    def _load_data(self) -> None:
        """Load Fashion-MNIST data from downloaded files."""
        if self.train:
            images_file = "train-images-idx3-ubyte.gz"
            labels_file = "train-labels-idx1-ubyte.gz"
        else:
            images_file = "t10k-images-idx3-ubyte.gz"
            labels_file = "t10k-labels-idx1-ubyte.gz"
        
        # Load images
        images_path = self.dataset_root / images_file
        with gzip.open(images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        
        # Load labels
        labels_path = self.dataset_root / labels_file
        with gzip.open(labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        self.data = images
        self.targets = labels
        
        print(f"Loaded Fashion-MNIST {'training' if self.train else 'test'} set: "
              f"{len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Get a sample from Fashion-MNIST dataset.
        
        Returns:
            sample: PIL Image of the fashion item
            target: int class label (0-9)
        """
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to PIL Image (L mode auto-detected for uint8 grayscale)
        img = Image.fromarray(img)
        
        # Apply transforms
        img, target = self._apply_transforms(img, target)
        
        return img, target
    
    def visualize(self, num_samples: int = 8, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize Fashion-MNIST samples."""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        indices = np.random.choice(len(self), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img, target = self[idx]
            
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            
            axes[i].imshow(img_array, cmap='gray')
            axes[i].set_title(f'{self.class_names[target]} ({target})')
            axes[i].axis('off')
        
        plt.suptitle(f'Fashion-MNIST {"Training" if self.train else "Test"} Samples')
        plt.tight_layout()
        plt.show()


class Cifar10Dataset(ManagedDataset):
    """
    CIFAR-10 dataset with automatic download and management.
    
    Returns:
        sample: PIL Image of shape (32, 32, 3)
        target: int class label (0-9)
    """
    
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "cifar-10"
    
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
        """Load CIFAR-10 data from extracted files."""
        extracted_dir = self.dataset_root / "cifar-10-batches-py"
        
        if self.train:
            # Load training batches
            data_list = []
            labels_list = []
            
            for i in range(1, 6):  # data_batch_1 to data_batch_5
                batch_file = extracted_dir / f"data_batch_{i}"
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f, encoding='bytes')
                    data_list.append(batch_data[b'data'])
                    labels_list.extend(batch_data[b'labels'])
            
            self.data = np.concatenate(data_list, axis=0)
            self.targets = np.array(labels_list)
        else:
            # Load test batch
            test_file = extracted_dir / "test_batch"
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f, encoding='bytes')
                self.data = test_data[b'data']
                self.targets = np.array(test_data[b'labels'])
        
        # Reshape data from flat to (height, width, channels)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        print(f"Loaded CIFAR-10 {'training' if self.train else 'test'} set: "
              f"{len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Get a sample from CIFAR-10 dataset.
        
        Returns:
            sample: PIL Image of the object
            target: int class label (0-9)
        """
        img, target = self.data[index], int(self.targets[index])
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms
        img, target = self._apply_transforms(img, target)
        
        return img, target
    
    def visualize(self, num_samples: int = 8, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize CIFAR-10 samples."""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        indices = np.random.choice(len(self), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img, target = self[idx]
            
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img
            
            axes[i].imshow(img_array)
            axes[i].set_title(f'{self.class_names[target]} ({target})')
            axes[i].axis('off')
        
        plt.suptitle(f'CIFAR-10 {"Training" if self.train else "Test"} Samples')
        plt.tight_layout()
        plt.show()