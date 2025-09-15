"""
ML Sandbox Utilities Package

This package contains various utilities for machine learning tasks including
dataset management, data loaders, and other common ML utilities.
"""

from .input_data import (
    # Dataset classes
    ManagedDataset,
    MnistDataset, 
    FashionMnistDataset,
    Cifar10Dataset,
    
    # Enums and configuration
    CommonDatasets,
    DatasetInfo,
    
    # Download utilities
    download_url,
    extract_archive,
    check_integrity,
)

__version__ = "0.1.0"
__author__ = "ML Sandbox"

__all__ = [
    # Dataset classes
    "ManagedDataset",
    "MnistDataset", 
    "FashionMnistDataset",
    "Cifar10Dataset",
    
    # Enums and configuration
    "CommonDatasets",
    "DatasetInfo",
    
    # Download utilities
    "download_url",
    "extract_archive", 
    "check_integrity",
]