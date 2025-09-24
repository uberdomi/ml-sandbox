"""
Fashion-MNIST dataset implementation.

This module contains the Fashion-MNIST dataset class along with its specific
download information and metadata definitions.
"""

import struct
import gzip
import numpy as np
from typing import Tuple, override

from .base import ManagedDataset, DatasetInfo
from .downloaders import DownloadInfo


# Fashion-MNIST-specific download information
FASHION_MNIST_DOWNLOADS = [
    DownloadInfo(
        name="Fashion-MNIST Training Images",
        filename="train-images-idx3-ubyte.gz",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"],
        md5="8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        description="Fashion-MNIST training images (60,000 examples)"
    ),
    DownloadInfo(
        name="Fashion-MNIST Training Labels",
        filename="train-labels-idx1-ubyte.gz",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
        ],
        md5="25c81989df183df01b3e8a0aad5dffbe",
        description="Fashion-MNIST training labels (60,000 examples)"
    ),
    DownloadInfo(
        name="Fashion-MNIST Test Images",
        filename="t10k-images-idx3-ubyte.gz",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
        ],
        md5="bef4ecab320f06d8554ea6380940ec79",
        description="Fashion-MNIST test images (10,000 examples)"
    ),
    DownloadInfo(
        name="Fashion-MNIST Test Labels",
        filename="t10k-labels-idx1-ubyte.gz",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
        ],
        md5="bb300cfdad3c16e7a12a480ee83cd310",
        description="Fashion-MNIST test labels (10,000 examples)"
    )
]

# Fashion-MNIST dataset information
FASHION_MNIST_INFO = DatasetInfo(
    name="Fashion-MNIST",
    description="Fashion-MNIST dataset of clothing images (28x28 grayscale images)",
    classes=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    num_classes=10,
    input_shape=(1, 28, 28),
    license="MIT License",
    citation="Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."
)

class FashionMnistDataset(ManagedDataset):
    """
    Fashion-MNIST dataset with automatic download and management.
    
    Loads ALL Fashion-MNIST data (train + test) into a unified dataset.
    Use get_dataloaders() to split into train/val/test sets.
    
    Returns:
        sample: torch.Tensor of shape (1, 28, 28) with values in [0, 1]
        target: int class label (0-9)
    """

    @override
    @property
    def download_infos(self) -> list[DownloadInfo]:
        return FASHION_MNIST_DOWNLOADS

    @override
    @property
    def dataset_name(self) -> str:
        return "fashion-mnist"

    @override
    @property
    def dataset_info(self) -> DatasetInfo:
        return FASHION_MNIST_INFO

    @override
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw Fashion-MNIST data (train + test) from downloaded files."""
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
        
        # Combine all data and normalize to 0-1 range
        combined_data = np.concatenate(all_images, axis=0).astype(np.float32) / 255.0
        combined_labels = np.concatenate(all_labels, axis=0)

        print(f"Loaded complete Fashion-MNIST dataset: {len(combined_data):,} samples "
              f"(train: {len(train_images):,}, test: {len(test_images):,})")

        return combined_data, combined_labels