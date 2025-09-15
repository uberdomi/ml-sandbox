"""
Input Data Package - Dataset Management and Loading Utilities

This package provides a clean API for managing and loading common ML datasets
with automatic downloading, integrity checking, and PyTorch integration.

Example usage:
    from src.utils.input_data import MnistDataset, CommonDatasets
    
    # Create a dataset instance
    mnist = MnistDataset(train=True, download=True)
    
    # Access dataset metadata
    info = CommonDatasets.MNIST.value
    print(f"Dataset: {info.name}, Classes: {info.classes}")
    
    # Use with PyTorch DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(mnist, batch_size=32, shuffle=True)
"""

# Import dataset classes
from .datasets import (
    ManagedDataset,
    MnistDataset,
    FashionMnistDataset, 
    Cifar10Dataset,
)

# Import enums and configuration
from .enums import (
    DatasetInfo,
    CommonDatasets,
)

# Import download utilities
from .downloaders import (
    download_url,
    extract_archive,
    check_integrity,
    gen_bar_updater,
)

__version__ = "0.1.0"

# Define the public API
__all__ = [
    # Dataset classes - main API
    "ManagedDataset",
    "MnistDataset",
    "FashionMnistDataset", 
    "Cifar10Dataset",
    
    # Configuration and metadata
    "DatasetInfo",
    "CommonDatasets",
    
    # Utility functions
    "download_url",
    "extract_archive",
    "check_integrity",
    "gen_bar_updater",
]

# Convenience shortcuts for common datasets
DATASETS = CommonDatasets

# Quick access to dataset info
def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """
    Get dataset information by name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        
    Returns:
        DatasetInfo object with metadata
        
    Example:
        info = get_dataset_info("mnist")
        print(f"Classes: {info.classes}")
    """
    dataset_name = dataset_name.upper().replace("-", "_")
    try:
        return getattr(CommonDatasets, dataset_name).value
    except AttributeError:
        available = [d.name for d in CommonDatasets]
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

# Quick dataset factory function
def create_dataset(dataset_name: str, train: bool = True, **kwargs):
    """
    Factory function to create dataset instances by name.
    
    Datasets now download by default. Use force_download=True to re-download.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        train: Whether to load training set (default: True)
        **kwargs: Additional arguments passed to dataset constructor (e.g., force_download=True)
        
    Returns:
        Dataset instance
        
    Example:
        train_set = create_dataset("mnist", train=True)
        test_set = create_dataset("fashion-mnist", train=False, force_download=True)
    """
    dataset_name = dataset_name.lower().replace("-", "_")
    
    dataset_map = {
        "mnist": MnistDataset,
        "fashion_mnist": FashionMnistDataset,
        "cifar10": Cifar10Dataset,
        "cifar_10": Cifar10Dataset,  # Alternative name
    }
    
    if dataset_name not in dataset_map:
        available = list(dataset_map.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
    
    dataset_class = dataset_map[dataset_name]
    return dataset_class(train=train, **kwargs)