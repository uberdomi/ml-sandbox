"""
Input Data Package - Dataset Management and Loading Utilities

This package provides a clean API for managing and loading common ML datasets
with automatic downloading, integrity checking, and PyTorch integration.

Example usage:
    from input_data import create_dataset, SupportedDatasets
    
    # Create a complete dataset instance (new recommended approach)
    dataset = create_dataset(SupportedDatasets.MNIST)
    
    # Get dataloaders with custom splits
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    
    # Print dataset information and visualize
    dataset.print_info()
    dataset.show_random_samples()
"""

from typing import Union, List

# Import dataset classes from modular structure
from .base import ManagedDataset, DatasetDownloads, DatasetInfo
from .mnist import MnistDataset
from .fashion_mnist import FashionMnistDataset
from .cifar10 import Cifar10Dataset


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
    # Main API function
    "create_dataset",
    
    # Dataset classes
    "ManagedDataset",
    "MnistDataset",
    "FashionMnistDataset", 
    "Cifar10Dataset",
    
    # Configuration and metadata
    "SupportedDatasets",
    "DatasetDownloads",
    "DatasetInfo",
    "DatasetDownloadsEnum",
    "DatasetInfoEnum", 
    
    # Backward compatibility
    "DatasetMetadata",  # Alias for DatasetInfo
    "CommonDatasets",   # Alias for DatasetDownloadsEnum
    "Datasets",         # Alias for DatasetInfoEnum
    "create_dataset_legacy",
    
    # Utility functions
    "download_url",
    "extract_archive",
    "check_integrity",
    "gen_bar_updater",
    
    # Convenience functions
    "get_dataset_info",
    "list_supported_datasets",
]

from enum import Enum, auto

class SupportedDatasets(Enum):
    """Enum of supported datasets."""
    MNIST = auto()
    FASHION_MNIST = auto()
    CIFAR10 = auto()


def list_supported_datasets() -> List[str]:
    """
    Get a list of all supported dataset names.
    
    Returns:
        List of supported dataset names
        
    Example:
        datasets = list_supported_datasets()
        print(f"Supported datasets: {', '.join(datasets)}")
    """
    return [d.name.lower() for d in SupportedDatasets]



# Main API function for creating datasets
def create_dataset(dataset: Union[str, SupportedDatasets], **kwargs) -> ManagedDataset:
    """
    Main API function to create dataset instances.
    
    This is the primary function for creating datasets in the input_data package.
    All datasets load complete data (train + test) and use get_dataloaders() for splits.
    
    Args:
        dataset: Dataset to create. Can be:
                - SupportedDatasets enum value (recommended)
                - String name (case-insensitive, supports aliases)
        **kwargs: Additional arguments passed to dataset constructor
                 (root, transform, target_transform, force_download)
        
    Returns:
        Complete dataset instance (loads all available data)
        
    Example:
        # Using enum (recommended)
        from input_data import create_dataset, SupportedDatasets
        dataset = create_dataset(SupportedDatasets.MNIST)
        
        # Using string name
        dataset = create_dataset("mnist")
        dataset = create_dataset("fashion-mnist")  # Alternative names supported
        
        # Get dataloaders with custom splits
        train_loader, val_loader, test_loader = dataset.get_dataloaders(
            train_split=0.7, val_split=0.15, test_split=0.15
        )
        
        # Print dataset information
        dataset.print_info()
        
        # Visualize samples
        dataset.show_random_samples()
        dataset.show_illustrative_samples()
    """
    # Handle both enum and string inputs
    if isinstance(dataset, SupportedDatasets):
        dataset_enum = dataset
        dataset_name = dataset.name.lower()
    elif isinstance(dataset, str):
        # Normalize string input
        dataset_name_normalized = dataset.lower().replace("-", "_")
        
        # Handle alternative names
        name_mapping = {
            "fashion-mnist": "fashion_mnist",
            "cifar-10": "cifar10",
            "cifar_10": "cifar10",
        }
        dataset_name = name_mapping.get(dataset_name_normalized, dataset_name_normalized)
        
        # Convert to enum
        try:
            dataset_enum = SupportedDatasets[dataset_name.upper()]
        except KeyError:
            available = [d.name.lower() for d in SupportedDatasets]
            raise ValueError(f"Dataset '{dataset}' not supported. Available: {available}")
    else:
        raise TypeError(f"dataset must be SupportedDatasets enum or string, got {type(dataset)}")
    
    # Map enum to dataset classes
    dataset_class_map = {
        SupportedDatasets.MNIST: MnistDataset,
        SupportedDatasets.FASHION_MNIST: FashionMnistDataset,
        SupportedDatasets.CIFAR10: Cifar10Dataset,
    }
    
    if dataset_enum not in dataset_class_map:
        raise NotImplementedError(f"Dataset {dataset_enum.name} is supported but not yet implemented")
    
    dataset_class = dataset_class_map[dataset_enum]
    return dataset_class(**kwargs)


# Backward compatibility function
def create_dataset_legacy(dataset_name: str, train: bool = True, **kwargs):
    """
    Legacy function for backward compatibility.
    
    NOTE: The 'train' parameter is DEPRECATED. Use create_dataset() instead.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        train: DEPRECATED - ignored, kept for backward compatibility
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Complete dataset instance (loads all data - train + test)
    """
    import warnings
    warnings.warn(
        "The 'train' parameter is deprecated. All datasets now load complete data. "
        "Use get_dataloaders() to split into train/val/test sets. "
        "Consider using create_dataset() with SupportedDatasets enum instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Remove 'train' parameter if present (no longer used)
    kwargs.pop('train', None)
    
    return create_dataset(dataset_name, **kwargs)