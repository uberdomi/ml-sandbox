"""
Static information about common datasets.
Used by the dataset downloading utilities.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass

@dataclass
class DatasetInfo:
    """Information about a dataset including URLs, checksums, and metadata."""
    name: str
    urls: list[str]  # Mirror URLs
    filename: str
    md5: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None  # Size in bytes
    description: str = ""
    license: str = ""
    citation: str = ""


class CommonDatasets(Enum):
    """Enum containing common ML datasets with their download information."""
    
    # --- MNIST Dataset ---
    MNIST_TRAIN_IMAGES = DatasetInfo(
        name="MNIST Training Images",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
        ],
        filename="train-images-idx3-ubyte.gz",
        md5="f68b3c2dcbeaaa9fbdd348bbdeb94873",
        description="MNIST training set images (60,000 examples)",
        license="Creative Commons Attribution-Share Alike 3.0",
        citation="LeCun, Y. (1998). The MNIST database of handwritten digits."
    )
    
    MNIST_TRAIN_LABELS = DatasetInfo(
        name="MNIST Training Labels",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
        ],
        filename="train-labels-idx1-ubyte.gz",
        md5="d53e105ee54ea40749a09fcbcd1e9432",
        description="MNIST training set labels (60,000 examples)"
    )
    
    MNIST_TEST_IMAGES = DatasetInfo(
        name="MNIST Test Images", 
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
        ],
        filename="t10k-images-idx3-ubyte.gz",
        md5="9fb629c4189551a2d022fa330f9573f3",
        description="MNIST test set images (10,000 examples)"
    )
    
    MNIST_TEST_LABELS = DatasetInfo(
        name="MNIST Test Labels",
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", 
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
        ],
        filename="t10k-labels-idx1-ubyte.gz",
        md5="ec29112dd5afa0611ce80d1b7f02629c",
        description="MNIST test set labels (10,000 examples)"
    )
    
    # --- Fashion-MNIST Dataset ---
    FASHION_MNIST_TRAIN_IMAGES = DatasetInfo(
        name="Fashion-MNIST Training Images",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
        ],
        filename="train-images-idx3-ubyte.gz",
        md5="8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        description="Fashion-MNIST training images (60,000 examples)",
        license="MIT License",
        citation="Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST."
    )
    
    FASHION_MNIST_TRAIN_LABELS = DatasetInfo(
        name="Fashion-MNIST Training Labels",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
        ],
        filename="train-labels-idx1-ubyte.gz", 
        md5="25c81989df183df01b3e8a0aad5dffbe",
        description="Fashion-MNIST training labels (60,000 examples)"
    )
    
    FASHION_MNIST_TEST_IMAGES = DatasetInfo(
        name="Fashion-MNIST Test Images",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
        ],
        filename="t10k-images-idx3-ubyte.gz",
        md5="bef4ecab320f06d8554ea6380940ec79",
        description="Fashion-MNIST test images (10,000 examples)"
    )
    
    FASHION_MNIST_TEST_LABELS = DatasetInfo(
        name="Fashion-MNIST Test Labels",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
        ],
        filename="t10k-labels-idx1-ubyte.gz",
        md5="bb300cfdad3c16e7a12a480ee83cd310",
        description="Fashion-MNIST test labels (10,000 examples)"
    )
    
    # --- CIFAR-10 Dataset ---
    CIFAR10 = DatasetInfo(
        name="CIFAR-10",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://s3.amazonaws.com/fast-ai-datasets/cifar10.tar.gz"
        ],
        filename="cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a", 
        description="CIFAR-10 dataset (60,000 32x32 color images in 10 classes)",
        license="Unknown",
        citation="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images."
    )
    
    # --- CIFAR-100 Dataset ---
    CIFAR100 = DatasetInfo(
        name="CIFAR-100",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        ],
        filename="cifar-100-python.tar.gz",
        md5="eb9058c3a382ffc7106e4002c42a8d85",
        description="CIFAR-100 dataset (60,000 32x32 color images in 100 classes)",
        license="Unknown", 
        citation="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images."
    )
    
    # --- SVHN Dataset ---
    SVHN_TRAIN = DatasetInfo(
        name="SVHN Training Set",
        urls=[
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        ],
        filename="train_32x32.mat",
        md5="e26dedcc434d2e4c54c9b2d4a06d8373",
        description="SVHN training set (73,257 digits)",
        citation="Netzer, Y., et al. (2011). Reading digits in natural images with unsupervised feature learning."
    )
    
    SVHN_TEST = DatasetInfo(
        name="SVHN Test Set", 
        urls=[
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
        ],
        filename="test_32x32.mat",
        md5="eb5a983be6a315427106f1b164d9cef3",
        description="SVHN test set (26,032 digits)"
    )
    
    SVHN_EXTRA = DatasetInfo(
        name="SVHN Extra Set",
        urls=[
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
        ],
        filename="extra_32x32.mat", 
        md5="a93ce644f1a588dc4d68dda5feec44a7",
        description="SVHN extra set (531,131 digits)"
    )
    
    # --- STL-10 Dataset ---
    STL10 = DatasetInfo(
        name="STL-10",
        urls=[
            "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        ],
        filename="stl10_binary.tar.gz",
        md5="91f7769df0f17e558f3565bffb0c7dfb",
        description="STL-10 dataset (96x96 color images, 10 classes)",
        citation="Coates, A., Ng, A., & Lee, H. (2011). STL-10 dataset."
    )
