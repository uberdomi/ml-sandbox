#!/usr/bin/env python3
"""
Example usage of the datasets utility module.
Demonstrates how to download and use common ML datasets.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.datasets import (
    CommonDatasets,
    download_mnist_dataset,
    download_fashion_mnist_dataset,
    download_cifar_dataset,
    download_by_name,
    list_available_datasets,
    dataset_exists
)


def main():
    """Main function demonstrating dataset utilities usage."""
    
    # Create a data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ML Dataset Utilities Example")
    print("=" * 60)
    
    # List all available datasets
    print("\n1. Listing all available datasets:")
    list_available_datasets()
    
    print("\n" + "=" * 60)
    print("2. Downloading MNIST dataset:")
    print("=" * 60)
    
    # Download MNIST training set
    try:
        mnist_files = download_mnist_dataset(data_dir, train=True, verbose=True)
        print(f"MNIST training files downloaded:")
        print(f"  Images: {mnist_files['images']}")
        print(f"  Labels: {mnist_files['labels']}")
        
        # Check if files exist and are valid
        print(f"\nValidating downloaded files:")
        images_valid = dataset_exists(CommonDatasets.MNIST_TRAIN_IMAGES, data_dir / "mnist")
        labels_valid = dataset_exists(CommonDatasets.MNIST_TRAIN_LABELS, data_dir / "mnist")
        print(f"  Images valid: {images_valid}")
        print(f"  Labels valid: {labels_valid}")
        
    except Exception as e:
        print(f"Error downloading MNIST: {e}")
    
    print("\n" + "=" * 60)
    print("3. Downloading Fashion-MNIST test set:")
    print("=" * 60)
    
    # Download Fashion-MNIST test set
    try:
        fashion_mnist_files = download_fashion_mnist_dataset(data_dir, train=False, verbose=True)
        print(f"Fashion-MNIST test files downloaded:")
        print(f"  Images: {fashion_mnist_files['images']}")
        print(f"  Labels: {fashion_mnist_files['labels']}")
        
    except Exception as e:
        print(f"Error downloading Fashion-MNIST: {e}")
    
    print("\n" + "=" * 60)
    print("4. Downloading CIFAR-10 dataset:")
    print("=" * 60)
    
    # Download and extract CIFAR-10
    try:
        cifar10_dir = download_cifar_dataset(data_dir, cifar_version=10, verbose=True)
        print(f"CIFAR-10 dataset extracted to: {cifar10_dir}")
        
        # List contents of CIFAR-10 directory
        if cifar10_dir.exists():
            print("Contents:")
            for item in cifar10_dir.iterdir():
                print(f"  {item.name}")
                
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
    
    print("\n" + "=" * 60)
    print("5. Download by name example:")
    print("=" * 60)
    
    # Download a specific dataset by name
    try:
        svhn_file = download_by_name("SVHN_TRAIN", data_dir, verbose=True)
        if svhn_file:
            print(f"SVHN training set downloaded to: {svhn_file}")
        else:
            print("Failed to download SVHN dataset")
            
    except Exception as e:
        print(f"Error downloading SVHN: {e}")
    
    print("\n" + "=" * 60)
    print("6. Summary of downloaded datasets:")
    print("=" * 60)
    
    # Show what was downloaded
    if data_dir.exists():
        print("Data directory contents:")
        for item in data_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(data_dir)} ({size_mb:.1f} MB)")
    
    print(f"\nTotal size of data directory: {get_directory_size(data_dir):.1f} MB")
    print("\nExample completed successfully!")


def get_directory_size(path: Path) -> float:
    """Calculate total size of directory in MB."""
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size / (1024 * 1024)


if __name__ == "__main__":
    main()