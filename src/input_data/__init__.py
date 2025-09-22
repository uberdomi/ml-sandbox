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

from typing import List

# Import dataset classes from modular structure
from .base import ManagedDataset
from .mnist import MnistDataset
from .fashion_mnist import FashionMnistDataset
from .cifar10 import Cifar10Dataset

__version__ = "0.1.0"

# Define the public API
__all__ = [
    # Main API function
    "create_dataset",

    # Base dataset class
    "ManagedDataset",
    # Dataset classes
    "MnistDataset",
    "FashionMnistDataset", 
    "Cifar10Dataset",
    
    # Supported datasets enum
    "SupportedDatasets",
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
def create_dataset(dataset: SupportedDatasets, **kwargs) -> ManagedDataset:
    """
    Main API function to create dataset instances.
    
    This is the primary function for creating datasets in the input_data package.
    All datasets load complete data (train + test) and use get_dataloaders() for splits.
    
    Args:
        dataset: Dataset to create out of the SupportedDatasets list.
        **kwargs: Additional arguments passed to dataset constructor
                 (root, transform, target_transform, force_download)
        
    Returns:
        Complete dataset instance (loads all available data)
        
    Example:
        from input_data import create_dataset, SupportedDatasets
        
        dataset = create_dataset(SupportedDatasets.MNIST)

        train_loader, val_loader, test_loader = dataset.get_dataloaders(
            train_split=0.7, val_split=0.15, test_split=0.15
        )
        
        dataset.print_info()
        
        dataset.show_random_samples()
        
        dataset.show_illustrative_samples()
    """
    # Handle both enum and string inputs
    if not isinstance(dataset, SupportedDatasets):
        raise TypeError(f"dataset must be SupportedDatasets enum, got {type(dataset)}")
    
    # Map enum to dataset classes
    dataset_class_map = {
        SupportedDatasets.MNIST: MnistDataset,
        SupportedDatasets.FASHION_MNIST: FashionMnistDataset,
        SupportedDatasets.CIFAR10: Cifar10Dataset,
    }
    
    if dataset not in dataset_class_map:
        raise NotImplementedError(f"Dataset {dataset.name} is supported but not yet implemented")

    dataset_class = dataset_class_map[dataset]
    return dataset_class(**kwargs)