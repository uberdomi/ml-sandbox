"""
ML Sandbox Utilities Package

This package contains various utilities for machine learning tasks including
dataset management, data loaders, and other common ML utilities.
"""

from .input_data import (
    # Main API function
    create_dataset,

    # Base dataset class
    ManagedDataset,
    # Dataset classes
    MnistDataset,
    FashionMnistDataset, 
    Cifar10Dataset,
    
    # Supported datasets enum
    SupportedDatasets,
    list_supported_datasets,
)

from .trainer import (
    Trainer,
)

from .models import (
    Autoencoder,
)

__version__ = "0.1.0"
__author__ = "ML Sandbox"

__all__ = input_data.__all__