"""A package containing concrete dataset implementations."""
from .mnist import MnistDataset
from .fashion_mnist import FashionMnistDataset
from .cifar10 import Cifar10Dataset

__all__ = [
    "MnistDataset",
    "FashionMnistDataset",
    "Cifar10Dataset",
]