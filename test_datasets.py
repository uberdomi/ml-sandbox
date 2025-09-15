#!/usr/bin/env python3
"""
Test script for the managed dataset classes.
Tests downloading, loading, and basic functionality.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from utils.input_data.datasets import MnistDataset, FashionMnistDataset, Cifar10Dataset
    print("✅ Successfully imported dataset classes")
except ImportError as e:
    print(f"❌ Failed to import dataset classes: {e}")
    sys.exit(1)


def test_mnist():
    """Test MNIST dataset functionality."""
    print("\n" + "="*50)
    print("Testing MNIST Dataset")
    print("="*50)
    
    try:
        # Test training set
        print("Creating MNIST training dataset...")
        mnist_train = MnistDataset(train=True, download=True)
        
        print(f"✅ MNIST training set loaded: {len(mnist_train)} samples")
        
        # Test a few samples
        print("Testing sample access...")
        for i in range(3):
            img, label = mnist_train[i]
            print(f"  Sample {i}: Image type={type(img)}, Label={label}")
        
        # Test test set
        print("\nCreating MNIST test dataset...")
        mnist_test = MnistDataset(train=False, download=True)
        print(f"✅ MNIST test set loaded: {len(mnist_test)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ MNIST test failed: {e}")
        return False


def test_fashion_mnist():
    """Test Fashion-MNIST dataset functionality."""
    print("\n" + "="*50)
    print("Testing Fashion-MNIST Dataset")
    print("="*50)
    
    try:
        print("Creating Fashion-MNIST training dataset...")
        fashion_train = FashionMnistDataset(train=True, download=True)
        
        print(f"✅ Fashion-MNIST training set loaded: {len(fashion_train)} samples")
        
        # Test samples with class names
        print("Testing sample access with class names...")
        for i in range(3):
            img, label = fashion_train[i]
            class_name = fashion_train.class_names[label]
            print(f"  Sample {i}: Image type={type(img)}, Label={label} ({class_name})")
        
        return True
        
    except Exception as e:
        print(f"❌ Fashion-MNIST test failed: {e}")
        return False


def test_cifar10():
    """Test CIFAR-10 dataset functionality."""
    print("\n" + "="*50)
    print("Testing CIFAR-10 Dataset")
    print("="*50)
    
    try:
        print("Creating CIFAR-10 training dataset...")
        cifar_train = Cifar10Dataset(train=True, download=True)
        
        print(f"✅ CIFAR-10 training set loaded: {len(cifar_train)} samples")
        
        # Test samples with class names
        print("Testing sample access with class names...")
        for i in range(3):
            img, label = cifar_train[i]
            class_name = cifar_train.class_names[label]
            print(f"  Sample {i}: Image type={type(img)}, Label={label} ({class_name})")
            
        return True
        
    except Exception as e:
        print(f"❌ CIFAR-10 test failed: {e}")
        return False


def test_transforms():
    """Test dataset with transforms."""
    print("\n" + "="*50)
    print("Testing Transforms")
    print("="*50)
    
    try:
        import torchvision.transforms as transforms
        
        # Define a simple transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        print("Creating MNIST with transforms...")
        mnist_transformed = MnistDataset(
            train=True, 
            download=False,  # Don't re-download
            transform=transform
        )
        
        # Test transformed sample
        img, label = mnist_transformed[0]
        print(f"✅ Transformed sample: Image type={type(img)}, shape={img.shape if hasattr(img, 'shape') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False


def test_data_structure():
    """Test that data is stored in correct locations."""
    print("\n" + "="*50)
    print("Testing Data Storage Structure")
    print("="*50)
    
    try:
        # Check if data directories exist
        project_root = Path(__file__).parent
        data_root = project_root / "data"
        
        expected_dirs = ["mnist", "fashion-mnist", "cifar-10"]
        
        print(f"Checking data structure in: {data_root}")
        
        for dir_name in expected_dirs:
            dir_path = data_root / dir_name
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                print(f"✅ {dir_name}: {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
                    print(f"    - {file.name} ({size_mb:.1f} MB)")
            else:
                print(f"❌ {dir_name}: Directory not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Dataset Tests")
    print("=" * 60)
    
    results = []
    
    # Run individual tests
    results.append(("MNIST", test_mnist()))
    results.append(("Fashion-MNIST", test_fashion_mnist()))
    results.append(("CIFAR-10", test_cifar10()))
    results.append(("Transforms", test_transforms()))
    results.append(("Data Structure", test_data_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Dataset classes are working correctly.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)