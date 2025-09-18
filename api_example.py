#!/usr/bin/env python3
"""
Example script demonstrating the input_data package API.

This script shows how to use the new unified dataset API with various
usage patterns and features.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from input_data import (
    create_dataset, 
    get_dataset_info, 
    list_supported_datasets,
    SupportedDatasets
)

def main():
    print("=" * 60)
    print("INPUT_DATA PACKAGE API DEMONSTRATION")
    print("=" * 60)
    
    # 1. List all supported datasets
    print("\n1. SUPPORTED DATASETS")
    print("-" * 30)
    datasets = list_supported_datasets()
    print(f"Available datasets: {', '.join(datasets)}")
    
    # 2. Get dataset information
    print("\n2. DATASET INFORMATION")
    print("-" * 30)
    
    # Using enum (recommended)
    mnist_info = get_dataset_info(SupportedDatasets.MNIST)
    print("MNIST Info (via enum):")
    print(mnist_info.print())
    
    print("\n" + "-" * 30)
    
    # Using string with alternative name
    fashion_info = get_dataset_info("fashion-mnist")
    print("Fashion-MNIST Info (via string alias):")
    print(fashion_info.print())
    
    # 3. Dataset creation examples
    print("\n" + "=" * 60)
    print("3. DATASET CREATION EXAMPLES")
    print("=" * 60)
    
    print("\nNOTE: The following examples show API usage patterns.")
    print("Actual dataset creation would download and load data.")
    print("Remove the 'return' statement below to test with real data.")
    
    return  # Comment this out to test with real data downloads
    
    # Example 1: Using enum (recommended)
    print("\nExample 1: Creating dataset with enum")
    print("-" * 40)
    dataset = create_dataset(SupportedDatasets.MNIST)
    dataset.print_info()
    
    # Get dataloaders with custom splits
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        train_split=0.7, val_split=0.15, test_split=0.15, batch_size=64
    )
    print(f"DataLoaders created: {len(train_loader)} train batches, "
          f"{len(val_loader)} val batches, {len(test_loader)} test batches")
    
    # Example 2: Using string name
    print("\nExample 2: Creating dataset with string")
    print("-" * 40)
    cifar_dataset = create_dataset("cifar-10")  # Alternative name
    cifar_dataset.print_info()
    
    # Example 3: Visualization
    print("\nExample 3: Visualization")
    print("-" * 40)
    print("Showing random samples...")
    dataset.show_random_samples(num_samples=4)
    
    print("Showing illustrative samples (one per class)...")
    dataset.show_illustrative_samples(num_per_class=1)
    
    # Example 4: Custom transforms
    print("\nExample 4: With transforms")
    print("-" * 40)
    import torch
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    transformed_dataset = create_dataset(
        SupportedDatasets.FASHION_MNIST,
        transform=transform
    )
    
    # Get a sample to verify transform was applied
    sample, label = transformed_dataset[0]
    print(f"Transformed sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Example 5: Different split ratios
    print("\nExample 5: Custom split ratios")
    print("-" * 40)
    
    # 80/10/10 split
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        train_split=0.8, val_split=0.1, test_split=0.1
    )
    
    # Train only (no validation/test)
    train_only_loader, _, _ = dataset.get_dataloaders(
        train_split=1.0, val_split=0.0, test_split=0.0
    )
    
    print("✅ All examples completed successfully!")

if __name__ == "__main__":
    main()