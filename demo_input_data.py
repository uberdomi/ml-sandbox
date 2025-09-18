#!/usr/bin/env python3
"""
Input Data Package - Comprehensive Usage Example

This script demonstrates all the key features of the refactored input_data package:
- Unified dataset architecture (loads train + test data together)
- Flexible dataloader creation with custom splits
- Dataset information and visualization methods
- Support for multiple datasets (MNIST, Fashion-MNIST, CIFAR-10)
- Transform support and PyTorch integration

Run this script to see the package in action!
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the input_data package
from input_data import create_dataset, SupportedDatasets, get_dataset_info, list_supported_datasets

def main():
    print("🚀 Input Data Package - Comprehensive Demo")
    print("=" * 50)
    
    # Show available datasets
    print("\n📋 Available Datasets:")
    datasets = list_supported_datasets()
    for i, name in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    
    # Demonstrate the main API
    print("\n🎯 Main API Demo - create_dataset()")
    print("-" * 30)
    
    # Create datasets using different methods
    print("\n1️⃣ Creating MNIST dataset using enum:")
    mnist = create_dataset(SupportedDatasets.MNIST, root="demo_data")
    print(f"   ✅ Created {mnist.__class__.__name__}")
    print(f"   📊 Total samples: {len(mnist):,}")
    
    print("\n2️⃣ Creating Fashion-MNIST using string:")
    fashion = create_dataset("fashion-mnist", root="demo_data")
    print(f"   ✅ Created {fashion.__class__.__name__}")
    print(f"   📊 Total samples: {len(fashion):,}")
    
    print("\n3️⃣ Creating CIFAR-10 using alternative name:")
    cifar = create_dataset("cifar-10", root="demo_data")
    print(f"   ✅ Created {cifar.__class__.__name__}")
    print(f"   📊 Total samples: {len(cifar):,}")
    
    # Demonstrate dataset information
    print("\n📝 Dataset Information Demo")
    print("-" * 30)
    
    # Show detailed info for each dataset
    datasets = [("MNIST", mnist), ("Fashion-MNIST", fashion), ("CIFAR-10", cifar)]
    
    for name, dataset in datasets:
        print(f"\n🔍 {name} Dataset Info:")
        info = dataset.dataset_info
        print(f"   Name: {info.name}")
        print(f"   Description: {info.description}")
        print(f"   Classes: {info.num_classes}")
        print(f"   Input Shape: {info.input_shape}")
        print(f"   Sample Classes: {info.classes[:3] if len(info.classes) > 3 else info.classes}")
        if hasattr(info, 'license') and info.license:
            print(f"   License: {info.license}")
    
    # Demonstrate unified dataset architecture
    print("\n🔄 Unified Dataset Architecture Demo")
    print("-" * 40)
    
    print("📦 All datasets load complete data (train + test combined):")
    print(f"   MNIST: 60k train + 10k test = {len(mnist):,} total")
    print(f"   Fashion-MNIST: 60k train + 10k test = {len(fashion):,} total") 
    print(f"   CIFAR-10: 50k train + 10k test = {len(cifar):,} total")
    
    # Demonstrate flexible dataloader creation
    print("\n⚡ Flexible DataLoader Creation Demo")
    print("-" * 40)
    
    print("\n🔧 Creating dataloaders with custom splits (70% train, 20% val, 10% test):")
    dataloaders = mnist.get_dataloaders(
        train_split=0.7, val_split=0.2, test_split=0.1,
        batch_size=32, num_workers=2
    )
    
    print(f"   📊 Train DataLoader: {len(dataloaders['train'].dataset):,} samples")
    print(f"   📊 Validation DataLoader: {len(dataloaders['val'].dataset):,} samples")
    print(f"   📊 Test DataLoader: {len(dataloaders['test'].dataset):,} samples")
    
    # Demonstrate sample access
    print("\n🔍 Sample Access Demo")
    print("-" * 25)
    
    print("\n📷 Sample data from each dataset:")
    for name, dataset in datasets[:2]:  # Show first 2 to keep demo reasonable
        sample, label = dataset[0]
        print(f"   {name}: Sample shape={tuple(sample.shape)}, Label={label}, Type={type(sample).__name__}")
    
    # Demonstrate information methods
    print("\n📊 Information Methods Demo")
    print("-" * 35)
    
    print("\n🖨️ Calling print_info() for MNIST:")
    mnist.print_info()
    
    # Demonstrate API convenience functions
    print("\n🛠️ API Convenience Functions Demo")
    print("-" * 40)
    
    print("\n📋 Getting dataset info without creating dataset:")
    for dataset_name in ["MNIST", "FASHION_MNIST", "CIFAR10"]:
        info = get_dataset_info(dataset_name)
        print(f"   {info.name}: {info.num_classes} classes, shape {info.input_shape}")
    
    # Demonstrate transforms (optional - requires torchvision)
    print("\n🎨 Transform Support Demo")
    print("-" * 30)
    
    try:
        import torchvision.transforms as transforms
        
        # Create dataset with transforms
        transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        mnist_transformed = create_dataset("mnist", root="demo_data", transform=transform)
        sample, label = mnist_transformed[0]
        
        print(f"   ✅ MNIST with transforms: shape={tuple(sample.shape)}")
        print(f"   📊 Value range: [{sample.min():.3f}, {sample.max():.3f}]")
        
    except ImportError:
        print("   ⚠️ torchvision not available - skipping transform demo")
    
    # Demonstrate modular imports
    print("\n🔧 Modular Import Demo")
    print("-" * 25)
    
    print("\n📦 Direct imports from modular structure:")
    try:
        from input_data.mnist import MnistDataset, MNIST_INFO  
        from input_data.fashion_mnist import FASHION_MNIST_INFO
        from input_data.cifar10 import CIFAR10_INFO
        
        print("   ✅ Successfully imported from input_data.mnist")
        print("   ✅ Successfully imported from input_data.fashion_mnist")  
        print("   ✅ Successfully imported from input_data.cifar10")
        
        print(f"\n   📋 Direct access to info constants:")
        print(f"      MNIST_INFO.name = '{MNIST_INFO.name}'")
        print(f"      FASHION_MNIST_INFO.name = '{FASHION_MNIST_INFO.name}'")
        print(f"      CIFAR10_INFO.name = '{CIFAR10_INFO.name}'")
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
    
    # Final summary
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("Key Features Demonstrated:")
    print("✅ Unified dataset architecture (train+test combined)")
    print("✅ Flexible dataloader creation with custom splits")
    print("✅ Multiple ways to create datasets (enum, string, aliases)")
    print("✅ Comprehensive dataset information access")
    print("✅ Sample data access and transforms")
    print("✅ Modular import structure")
    print("✅ API convenience functions")
    
    print(f"\n💾 Data stored in: {Path('demo_data').absolute()}")
    print("🔗 Use these datasets in your ML projects!")


if __name__ == "__main__":
    main()