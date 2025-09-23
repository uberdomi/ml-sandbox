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

from typing import List, Optional, Union, Callable, Type
from pathlib import Path

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
def create_dataset(
    dataset: Union[SupportedDatasets, str],
    root: Optional[str] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    force_download: bool = False,
    storage_strategy: str = "hybrid",
    memory_threshold_mb: float = 500.0
) -> ManagedDataset:
    """
    Create a dataset instance.
    
    Args:
        dataset: Dataset to create (enum or string)
        root: Root directory for data storage
        transform: Transform function for samples
        target_transform: Transform function for targets
        force_download: Force re-download of data
        storage_strategy: Storage strategy ('memory', 'disk', 'hybrid')
        memory_threshold_mb: Memory threshold for hybrid strategy (MB)
    
    Returns:
        ManagedDataset instance
        
    Examples:
        # Small dataset - use memory storage
        mnist = create_dataset(SupportedDatasets.MNIST, storage_strategy="memory")
        
        # Large dataset - use disk storage  
        large_dataset = create_dataset(SupportedDatasets.CIFAR10, storage_strategy="disk")
        
        # Automatic choice based on size
        dataset = create_dataset(SupportedDatasets.MNIST, storage_strategy="hybrid", memory_threshold_mb=100)
    """
    # Handle both enum and string inputs
    if not isinstance(dataset, (SupportedDatasets, str)):
        raise TypeError(f"dataset must be SupportedDatasets enum or string, got {type(dataset)}")
    
    # If string is passed, convert to enum
    if isinstance(dataset, str):
        dataset = SupportedDatasets[dataset.upper()]
    
    # Map enum to dataset classes
    dataset_class_map = {
        SupportedDatasets.MNIST: MnistDataset,
        SupportedDatasets.FASHION_MNIST: FashionMnistDataset,
        SupportedDatasets.CIFAR10: Cifar10Dataset,
    }
    
    if dataset not in dataset_class_map:
        raise NotImplementedError(f"Dataset {dataset.name} is supported but not yet implemented")

    dataset_class: Type[ManagedDataset] = dataset_class_map[dataset]
    return dataset_class(
        root=root,
        transform=transform, 
        target_transform=target_transform,
        force_download=force_download,
        storage_strategy=storage_strategy,
        memory_threshold_mb=memory_threshold_mb
    )