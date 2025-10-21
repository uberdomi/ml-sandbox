"""
TODO change description to match the new package name
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

from .trainer import (
    Trainer,
    Regressor,
    Classifier,
)

__all__ = [
    "Trainer",
    "Regressor",
    "Classifier",
]