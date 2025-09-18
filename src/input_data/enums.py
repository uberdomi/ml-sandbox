"""
Static information about common datasets.
Used by the dataset downloading utilities.
"""

from enum import Enum, auto
from typing import Optional, Tuple
from dataclasses import dataclass

class SupportedDatasets(Enum):
    """Enum of supported datasets."""
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()


@dataclass
class DatasetDownloads:
    """Information for downloading dataset files - URLs, checksums, and metadata."""
    name: str
    urls: list[str]  # Mirror URLs
    filename: str
    md5: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None  # Size in bytes
    description: str = ""

    def print(self) -> str:
        """Return a string summary of the download information."""
        lines = [
            f"Dataset Download: {self.name}",
            f"  Filename: {self.filename}",
            f"  URLs: {len(self.urls)} mirror(s)",
        ]
        for i, url in enumerate(self.urls, 1):
            lines.append(f"    {i}. {url}")
        if self.md5:
            lines.append(f"  MD5: {self.md5}")
        if self.sha256:
            lines.append(f"  SHA256: {self.sha256}")
        if self.file_size:
            lines.append(f"  File Size: {self.file_size:,} bytes")
        if self.description:
            lines.append(f"  Description: {self.description}")
        return "\n".join(lines)


@dataclass  
class DatasetInfo:
    """High-level information about a complete dataset including structure and metadata."""
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

class DatasetDownloadsEnum(Enum):
    """Enum containing download information for dataset files."""
    
    # --- MNIST Dataset Downloads ---
    MNIST_TRAIN_IMAGES = DatasetDownloads(
        name="MNIST Training Images",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
        ],
        filename="train-images-idx3-ubyte.gz",
        md5="f68b3c2dcbeaaa9fbdd348bbdeb94873",
        description="MNIST training set images (60,000 examples)"
    )
    
    MNIST_TRAIN_LABELS = DatasetDownloads(
        name="MNIST Training Labels",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
        ],
        filename="train-labels-idx1-ubyte.gz",
        md5="d53e105ee54ea40749a09fcbcd1e9432",
        description="MNIST training set labels (60,000 examples)"
    )
    
    MNIST_TEST_IMAGES = DatasetDownloads(
        name="MNIST Test Images", 
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
        ],
        filename="t10k-images-idx3-ubyte.gz",
        md5="9fb629c4189551a2d022fa330f9573f3",
        description="MNIST test set images (10,000 examples)"
    )
    
    MNIST_TEST_LABELS = DatasetDownloads(
        name="MNIST Test Labels",
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", 
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
        ],
        filename="t10k-labels-idx1-ubyte.gz",
        md5="ec29112dd5afa0611ce80d1b7f02629c",
        description="MNIST test set labels (10,000 examples)"
    )
    
    # --- Fashion-MNIST Dataset Downloads ---
    FASHION_MNIST_TRAIN_IMAGES = DatasetDownloads(
        name="Fashion-MNIST Training Images",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
        ],
        filename="train-images-idx3-ubyte.gz",
        md5="8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        description="Fashion-MNIST training images (60,000 examples)"
    )
    
    FASHION_MNIST_TRAIN_LABELS = DatasetDownloads(
        name="Fashion-MNIST Training Labels",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
        ],
        filename="train-labels-idx1-ubyte.gz", 
        md5="25c81989df183df01b3e8a0aad5dffbe",
        description="Fashion-MNIST training labels (60,000 examples)"
    )
    
    FASHION_MNIST_TEST_IMAGES = DatasetDownloads(
        name="Fashion-MNIST Test Images",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
        ],
        filename="t10k-images-idx3-ubyte.gz",
        md5="bef4ecab320f06d8554ea6380940ec79",
        description="Fashion-MNIST test images (10,000 examples)"
    )
    
    FASHION_MNIST_TEST_LABELS = DatasetDownloads(
        name="Fashion-MNIST Test Labels",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
        ],
        filename="t10k-labels-idx1-ubyte.gz",
        md5="bb300cfdad3c16e7a12a480ee83cd310",
        description="Fashion-MNIST test labels (10,000 examples)"
    )
    
    # --- CIFAR-10 Dataset Downloads ---
    CIFAR10 = DatasetDownloads(
        name="CIFAR-10",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://s3.amazonaws.com/fast-ai-datasets/cifar10.tar.gz"
        ],
        filename="cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a", 
        description="CIFAR-10 dataset (60,000 32x32 color images in 10 classes)"
    )
    
    # --- CIFAR-100 Dataset Downloads ---
    CIFAR100 = DatasetDownloads(
        name="CIFAR-100",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        ],
        filename="cifar-100-python.tar.gz",
        md5="eb9058c3a382ffc7106e4002c42a8d85",
        description="CIFAR-100 dataset (60,000 32x32 color images in 100 classes)"
    )
    
    # --- SVHN Dataset Downloads ---
    SVHN_TRAIN = DatasetDownloads(
        name="SVHN Training Set",
        urls=[
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        ],
        filename="train_32x32.mat",
        md5="e26dedcc434d2e4c54c9b2d4a06d8373",
        description="SVHN training set (73,257 digits)"
    )
    
    SVHN_TEST = DatasetDownloads(
        name="SVHN Test Set", 
        urls=[
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
        ],
        filename="test_32x32.mat",
        md5="eb5a983be6a315427106f1b164d9cef3",
        description="SVHN test set (26,032 digits)"
    )
    
    SVHN_EXTRA = DatasetDownloads(
        name="SVHN Extra Set",
        urls=[
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
        ],
        filename="extra_32x32.mat", 
        md5="a93ce644f1a588dc4d68dda5feec44a7",
        description="SVHN extra set (531,131 digits)"
    )
    
    # --- STL-10 Dataset Downloads ---
    STL10 = DatasetDownloads(
        name="STL-10",
        urls=[
            "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        ],
        filename="stl10_binary.tar.gz",
        md5="91f7769df0f17e558f3565bffb0c7dfb",
        description="STL-10 dataset (96x96 color images, 10 classes)"
    )


class DatasetInfoEnum(Enum):
    """Enum containing high-level dataset information and metadata."""
    
    MNIST = DatasetInfo(
        name="MNIST",
        description="MNIST database of handwritten digits (28x28 grayscale images)",
        classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        num_classes=10,
        input_shape=(1, 28, 28),
        license="Creative Commons Attribution-Share Alike 3.0",
        citation="LeCun, Y. (1998). The MNIST database of handwritten digits."
    )
    
    FASHION_MNIST = DatasetInfo(
        name="Fashion-MNIST",
        description="Fashion-MNIST dataset of clothing images (28x28 grayscale images)",
        classes=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
        num_classes=10,
        input_shape=(1, 28, 28),
        license="MIT License",
        citation="Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms."
    )
    
    CIFAR10 = DatasetInfo(
        name="CIFAR-10",
        description="CIFAR-10 dataset of natural images (32x32 color images)",
        classes=["airplane", "automobile", "bird", "cat", "deer", 
                "dog", "frog", "horse", "ship", "truck"],
        num_classes=10,
        input_shape=(3, 32, 32),
        license="MIT License",
        citation="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images."
    )


# Keep backward compatibility 
DatasetMetadata = DatasetInfo  # Alias for backward compatibility
Datasets = DatasetInfoEnum  # Alias for backward compatibility

# For downloaders - keep the old CommonDatasets name as alias
CommonDatasets = DatasetDownloadsEnum
