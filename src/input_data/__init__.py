"""
Input Data Package - Dataset Management and Loading Utilities

This package provides a clean API for managing and loading common ML datasets
with automatic downloading, integrity checking, and PyTorch integration.

Example usage:
    from src.input_data import MnistDataset, CommonDatasets
    
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
    DatasetDownloads,
    DatasetInfo,
    DatasetMetadata,  # Backward compatibility alias
    DatasetDownloadsEnum,
    DatasetInfoEnum,
    CommonDatasets,  # Backward compatibility alias
    Datasets,        # Backward compatibility alias
    SupportedDatasets,
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
    "DatasetDownloads",
    "DatasetInfo",
    "DatasetMetadata",  # Backward compatibility
    "DatasetDownloadsEnum",
    "DatasetInfoEnum", 
    "CommonDatasets",   # Backward compatibility
    "Datasets",         # Backward compatibility
    "SupportedDatasets",
    
    # Utility functions
    "download_url",
    "extract_archive",
    "check_integrity",
    "gen_bar_updater",
]

# Convenience shortcuts for common datasets
DATASETS = CommonDatasets

# Quick access to dataset info
def get_dataset_info(dataset_name: str) -> DatasetMetadata:
    """
    Get dataset information by name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        
    Returns:
        DatasetMetadata object with high-level dataset information
        
    Example:
        info = get_dataset_info("mnist")
        print(f"Classes: {info.classes}")
    """
    # Normalize dataset name for lookup
    dataset_name = dataset_name.upper().replace("-", "_")
    
    # Handle alternative names
    name_mapping = {
        "FASHION_MNIST": "FASHION_MNIST",
        "CIFAR_10": "CIFAR10",
        "CIFAR10": "CIFAR10"
    }
    dataset_name = name_mapping.get(dataset_name, dataset_name)
    
    try:
        return getattr(Datasets, dataset_name).value
    except AttributeError:
        available = [d.name for d in Datasets]
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

# Quick dataset factory function
def create_dataset(dataset_name: str, train: bool = True, **kwargs):
    """
    Factory function to create dataset instances by name.
    
    NOTE: The 'train' parameter is now DEPRECATED. All datasets load complete data
    and use get_dataloaders() for train/val/test splits. Kept for backward compatibility.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        train: DEPRECATED - kept for backward compatibility, ignored
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Complete dataset instance (loads all data - train + test)
        
    Example:
        # New recommended usage
        dataset = create_dataset("mnist")
        train_loader, val_loader, test_loader = dataset.get_dataloaders()
        
        # Old usage still works but loads complete dataset
        dataset = create_dataset("mnist", train=True)  # train parameter ignored
    """
    # Normalize dataset name and create mapping from enum names to classes
    dataset_name_normalized = dataset_name.lower().replace("-", "_")
    
    # Create mapping based on our Datasets enum
    dataset_map = {
        "mnist": MnistDataset,
        "fashion_mnist": FashionMnistDataset,
        "cifar10": Cifar10Dataset,
    }
    
    # Handle alternative names
    alternative_names = {
        "cifar_10": "cifar10",
        "fashion-mnist": "fashion_mnist",
        "cifar-10": "cifar10",
    }
    
    # Use alternative name if it exists
    lookup_name = alternative_names.get(dataset_name_normalized, dataset_name_normalized)
    
    if lookup_name not in dataset_map:
        # Show available names from our Datasets enum
        available = [d.value.name for d in Datasets]
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
    
    dataset_class = dataset_map[lookup_name]
    
    # Remove 'train' parameter if present (no longer used)
    kwargs.pop('train', None)
    
    return dataset_class(**kwargs)