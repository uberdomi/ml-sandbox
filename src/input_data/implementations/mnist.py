"""
MNIST dataset implementation.

This module contains the MNIST dataset class along with its specific
download information and metadata definitions.
"""

import struct
import gzip
import numpy as np
from typing import List, Tuple, override

from ..structure import ManagedDataset, DatasetInfo
from ..downloaders import DownloadInfo


# MNIST-specific download information
MNIST_DOWNLOADS = [
    DownloadInfo(
        name="MNIST Training Images",
        filename="train-images-idx3-ubyte.gz",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
        ],
        md5="f68b3c2dcbeaaa9fbdd348bbdeb94873",
        description="MNIST training set images (60,000 examples)"
    ),
    DownloadInfo(
        name="MNIST Training Labels",
        filename="train-labels-idx1-ubyte.gz",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
        ],
        md5="d53e105ee54ea40749a09fcbcd1e9432",
        description="MNIST training set labels (60,000 examples)"
    ),
    DownloadInfo(
        name="MNIST Test Images",
        filename="t10k-images-idx3-ubyte.gz",
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
        ],
        md5="9fb629c4189551a2d022fa330f9573f3",
        description="MNIST test set images (10,000 examples)"
    ),
    DownloadInfo(
        name="MNIST Test Labels",
        filename="t10k-labels-idx1-ubyte.gz",
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
        ],
        md5="ec29112dd5afa0611ce80d1b7f02629c",
        description="MNIST test set labels (10,000 examples)"
    ),
]

# MNIST dataset information
MNIST_INFO = DatasetInfo(
    name="MNIST",
    description="MNIST database of handwritten digits (28x28 grayscale images)",
    classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    num_classes=10,
    input_shape=(1, 28, 28),
    license="Creative Commons Attribution-Share Alike 3.0",
    citation="LeCun, Y. (1998). The MNIST database of handwritten digits."
)


class MnistDataset(ManagedDataset):
    """
    MNIST dataset with automatic download and flexible storage.
    
    Loads ALL MNIST data (train + test) into a unified dataset.
    Use get_dataloaders() to split into train/val/test sets.
    
    Returns:
        sample: torch.Tensor of shape (1, 28, 28) with values in [0, 1]
        target: int class label (0-9)
    """
    
    @override
    @property
    def download_infos(self) -> List[DownloadInfo]:
        return MNIST_DOWNLOADS

    @override
    @property
    def dataset_name(self) -> str:
        return "mnist"

    @override
    @property
    def dataset_info(self) -> DatasetInfo:
        return MNIST_INFO
    
    @override
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw MNIST data from downloaded files."""
        all_images = []
        all_labels = []
        
        # Load training data
        train_images_path = self.dataset_root / "train-images-idx3-ubyte.gz"
        train_labels_path = self.dataset_root / "train-labels-idx1-ubyte.gz"
        
        with gzip.open(train_images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8)
            train_images = train_images.reshape(num_images, 1, rows, cols)  # Add channel dimension
        
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
            test_images = test_images.reshape(num_images, 1, rows, cols)  # Add channel dimension
        
        with gzip.open(test_labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        all_images.append(test_images)
        all_labels.append(test_labels)
        
        # Combine all data and normalize to 0-1 range
        combined_data = np.concatenate(all_images, axis=0).astype(np.float32) / 255.0
        combined_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Loaded complete MNIST dataset: {len(combined_data):,} samples "
              f"(train: {len(train_images):,}, test: {len(test_images):,})")
        
        return combined_data, combined_labels