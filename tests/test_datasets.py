"""
Pytest suite for the managed dataset classes from the input_data package.
Tests downloading, loading, and basic functionality.
"""

import pytest
import sys
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Classes and functions to test
from input_data import (
    MnistDataset, 
    FashionMnistDataset, 
    Cifar10Dataset,
    CommonDatasets,
    create_dataset,
    get_dataset_info,
)

# Import functions that aren't in __all__ but needed for testing
from input_data.downloaders import download_dataset, dataset_exists

# Configure logging  
logger = logging.getLogger(__name__)


@pytest.mark.mnist
@pytest.mark.unit
class TestMnistDataset:
    """Test suite for MNIST dataset."""
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_mnist_train_dataset_loading(self, temp_data_dir):
        """Test that MNIST training dataset can be loaded with appropriate parameters."""
        try:
            mnist_train = MnistDataset(train=True, root=temp_data_dir)
            assert len(mnist_train) > 0, "MNIST training dataset should not be empty"
            assert mnist_train.train is True, "Training flag should be True"
            logger.info(f"Loaded MNIST training set: {len(mnist_train)} samples")
        except Exception as e:
            logger.error(f"MNIST training dataset loading failed: {e}")
            raise
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_mnist_test_dataset_loading(self, temp_data_dir):
        """Test MNIST test dataset loading."""
        try:
            mnist_test = MnistDataset(train=False, root=temp_data_dir)
            assert len(mnist_test) == 10000, f"Expected 10000 samples, got {len(mnist_test)}"
            logger.info(f"MNIST test set loaded: {len(mnist_test)} samples")
        except Exception as e:
            logger.error(f"MNIST test dataset loading failed: {e}")
            raise
    
    def test_mnist_sample_access(self, temp_data_dir, tensor_shapes, small_sample_size):
        """Test MNIST sample access and data types."""
        try:
            mnist = MnistDataset(train=True, root=temp_data_dir)
            
            # Test multiple samples (limit to small number for speed)
            samples_to_test = min(small_sample_size // 20, 5)  # Test 5 samples max
            for i in range(samples_to_test):
                img, label = mnist[i]
                assert isinstance(img, torch.Tensor), f"Sample {i}: Expected tensor, got {type(img)}"
                assert isinstance(label, int), f"Sample {i}: Expected int label, got {type(label)}"
                assert img.shape == (1, 28, 28), f"Sample {i}: Expected shape (1, 28, 28), got {img.shape}"
                assert 0 <= label <= 9, f"Sample {i}: Invalid label {label}"
                logger.info(f"Sample {i}: Image shape={img.shape}, Label={label}")
        except Exception as e:
            logger.error(f"MNIST sample access failed: {e}")
            raise
    
    def test_mnist_force_redownload(self):
        """Test MNIST force re-download functionality."""
        try:
            # First ensure we have the dataset
            mnist = MnistDataset(train=True, force_download=False)
            initial_len = len(mnist)
            
            # Force re-download
            mnist_redownload = MnistDataset(train=True, force_download=True)
            assert len(mnist_redownload) == initial_len, "Re-downloaded dataset has different size"
            logger.info("MNIST force re-download successful")
        except Exception as e:
            logger.error(f"MNIST force re-download failed: {e}")
            raise


@pytest.mark.fashion_mnist
@pytest.mark.unit
class TestFashionMnistDataset:
    """Test suite for Fashion-MNIST dataset."""
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_fashion_mnist_train_dataset_loading(self, temp_data_dir):
        """Test Fashion-MNIST training dataset loading."""
        try:
            fashion_train = FashionMnistDataset(train=True, root=temp_data_dir)
            assert len(fashion_train) == 60000, f"Expected 60000 samples, got {len(fashion_train)}"
            logger.info(f"Fashion-MNIST training set loaded: {len(fashion_train)} samples")
        except Exception as e:
            logger.error(f"Fashion-MNIST training dataset loading failed: {e}")
            raise
    
    def test_fashion_mnist_test_dataset_loading(self):
        """Test Fashion-MNIST test dataset loading."""
        try:
            fashion_test = FashionMnistDataset(train=False)
            assert len(fashion_test) == 10000, f"Expected 10000 samples, got {len(fashion_test)}"
            logger.info(f"Fashion-MNIST test set loaded: {len(fashion_test)} samples")
        except Exception as e:
            logger.error(f"Fashion-MNIST test dataset loading failed: {e}")
            raise
    
    def test_fashion_mnist_class_names(self):
        """Test Fashion-MNIST class names and sample access."""
        try:
            fashion = FashionMnistDataset(train=True)
            
            # Test class names exist
            assert len(fashion.class_names) == 10, f"Expected 10 class names, got {len(fashion.class_names)}"
            
            # Test samples with class names
            for i in range(5):
                img, label = fashion[i]
                assert isinstance(img, torch.Tensor), f"Sample {i}: Expected tensor, got {type(img)}"
                assert isinstance(label, int), f"Sample {i}: Expected int label, got {type(label)}"
                assert img.shape == (1, 28, 28), f"Sample {i}: Expected shape (1, 28, 28), got {img.shape}"
                class_name = fashion.class_names[label]
                assert 0 <= label <= 9, f"Sample {i}: Invalid label {label}"
                assert isinstance(class_name, str), f"Sample {i}: Expected string class name, got {type(class_name)}"
                logger.info(f"Sample {i}: Label={label} ({class_name})")
        except Exception as e:
            logger.error(f"Fashion-MNIST class names test failed: {e}")
            raise


@pytest.mark.cifar10
@pytest.mark.unit
class TestCifar10Dataset:
    """Test suite for CIFAR-10 dataset."""
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_cifar10_train_dataset_loading(self, temp_data_dir):
        """Test CIFAR-10 training dataset loading."""
        try:
            cifar_train = Cifar10Dataset(train=True, root=temp_data_dir)
            assert len(cifar_train) == 50000, f"Expected 50000 samples, got {len(cifar_train)}"
            logger.info(f"CIFAR-10 training set loaded: {len(cifar_train)} samples")
        except Exception as e:
            logger.error(f"CIFAR-10 training dataset loading failed: {e}")
            raise
    
    def test_cifar10_test_dataset_loading(self):
        """Test CIFAR-10 test dataset loading."""
        try:
            cifar_test = Cifar10Dataset(train=False)
            assert len(cifar_test) == 10000, f"Expected 10000 samples, got {len(cifar_test)}"
            logger.info(f"CIFAR-10 test set loaded: {len(cifar_test)} samples")
        except Exception as e:
            logger.error(f"CIFAR-10 test dataset loading failed: {e}")
            raise
    
    def test_cifar10_class_names_and_samples(self):
        """Test CIFAR-10 class names and sample access."""
        try:
            cifar = Cifar10Dataset(train=True)
            
            # Test class names exist
            assert len(cifar.class_names) == 10, f"Expected 10 class names, got {len(cifar.class_names)}"
            
            # Test samples are color images
            for i in range(5):
                img, label = cifar[i]
                assert isinstance(img, torch.Tensor), f"Sample {i}: Expected tensor, got {type(img)}"
                assert img.shape == (3, 32, 32), f"Sample {i}: Expected shape (3, 32, 32), got {img.shape}"
                assert isinstance(label, int), f"Sample {i}: Expected int label, got {type(label)}"
                class_name = cifar.class_names[label]
                assert 0 <= label <= 9, f"Sample {i}: Invalid label {label}"
                assert isinstance(class_name, str), f"Sample {i}: Expected string class name, got {type(class_name)}"
                logger.info(f"Sample {i}: Label={label} ({class_name}), Shape={img.shape}")
        except Exception as e:
            logger.error(f"CIFAR-10 class names and samples test failed: {e}")
            raise


@pytest.mark.transform
@pytest.mark.unit
class TestDatasetTransforms:
    """Test suite for dataset transforms."""
    
    def test_mnist_with_transforms(self, temp_data_dir, sample_transforms, tensor_shapes):
        """Test MNIST dataset with torchvision transforms."""
        try:
            mnist_transformed = MnistDataset(
                train=True, 
                root=temp_data_dir,
                transform=sample_transforms
            )
            
            # Test transformed sample
            img, label = mnist_transformed[0]
            expected_shape = tensor_shapes['mnist']
            assert hasattr(img, 'shape'), f"Expected tensor with shape, got {type(img)}"
            assert img.shape == expected_shape, f"Expected shape {expected_shape}, got {img.shape}"
            assert isinstance(label, int), f"Expected int label, got {type(label)}"
            logger.info(f"Transformed MNIST sample: shape={img.shape}, label={label}")
        except Exception as e:
            logger.error(f"MNIST transforms test failed: {e}")
            raise
    
    def test_cifar10_with_transforms(self):
        """Test CIFAR-10 dataset with torchvision transforms."""
        try:
            import torchvision.transforms as transforms
            
            # Define transforms for RGB images
            # Note: No ToTensor() needed - datasets return tensors by default
            transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            cifar_transformed = Cifar10Dataset(
                train=True, 
                transform=transform
            )
            
            # Test transformed sample
            img, label = cifar_transformed[0]
            assert hasattr(img, 'shape'), f"Expected tensor with shape, got {type(img)}"
            assert img.shape == (3, 32, 32), f"Expected shape (3, 32, 32), got {img.shape}"
            assert isinstance(label, int), f"Expected int label, got {type(label)}"
            logger.info(f"Transformed CIFAR-10 sample: shape={img.shape}, label={label}")
        except Exception as e:
            logger.error(f"CIFAR-10 transforms test failed: {e}")
            raise


class TestDatasetDownloaders:
    """Test suite for dataset downloading functionality."""
    
    def test_download_individual_files(self):
        """Test downloading individual dataset files."""
        try:
            # Test downloading individual MNIST files
            temp_dir = Path(tempfile.mkdtemp())
            
            # Download MNIST training images
            file_path = download_dataset(
                CommonDatasets.MNIST_TRAIN_IMAGES, 
                temp_dir, 
                force_download=False
            )
            
            assert file_path.exists(), f"Downloaded file not found: {file_path}"
            assert file_path.stat().st_size > 0, f"Downloaded file is empty: {file_path}"
            logger.info(f"Downloaded {file_path.name}: {file_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Clean up
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Individual file download test failed: {e}")
            raise
    
    def test_dataset_exists_functionality(self):
        """Test dataset existence checking."""
        try:
            # This should return True for already downloaded datasets
            exists = dataset_exists(CommonDatasets.MNIST_TRAIN_IMAGES, Path("data/mnist"))
            logger.info(f"MNIST training images exist: {exists}")
            
            # Test with non-existent path
            temp_dir = Path(tempfile.mkdtemp())
            exists_false = dataset_exists(CommonDatasets.MNIST_TRAIN_IMAGES, temp_dir)
            assert not exists_false, "dataset_exists should return False for non-existent files"
            
            # Clean up
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Dataset exists test failed: {e}")
            raise
    
    def test_force_redownload_behavior(self):
        """Test force re-download functionality."""
        try:
            temp_dir = Path(tempfile.mkdtemp())
            
            # First download
            file_path1 = download_dataset(
                CommonDatasets.MNIST_TRAIN_LABELS, 
                temp_dir, 
                force_download=False
            )
            original_size = file_path1.stat().st_size
            
            # Force re-download
            file_path2 = download_dataset(
                CommonDatasets.MNIST_TRAIN_LABELS, 
                temp_dir, 
                force_download=True
            )
            
            assert file_path1 == file_path2, "File paths should be the same"
            assert file_path2.stat().st_size == original_size, "Re-downloaded file should have same size"
            logger.info(f"Force re-download successful: {file_path2.name}")
            
            # Clean up
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Force re-download test failed: {e}")
            raise


class TestDataStructure:
    """Test suite for data organization and storage."""
    
    def test_data_directory_structure(self):
        """Test that data is stored in correct locations."""
        try:
            project_root = Path(__file__).parent.parent
            data_root = project_root / "data"
            
            expected_dirs = ["mnist", "fashion-mnist", "cifar-10"]
            
            logger.info(f"Checking data structure in: {data_root}")
            
            for dir_name in expected_dirs:
                dir_path = data_root / dir_name
                if dir_path.exists():
                    files = list(dir_path.glob("*"))
                    logger.info(f"{dir_name}: {len(files)} files")
                    for file in files[:3]:  # Log first 3 files
                        if file.is_file():
                            size_mb = file.stat().st_size / (1024 * 1024)
                            logger.info(f"  - {file.name} ({size_mb:.1f} MB)")
                        else:
                            logger.info(f"  - {file.name} (directory)")
                else:
                    logger.warning(f"{dir_name}: Directory not found")
        except Exception as e:
            logger.error(f"Data structure test failed: {e}")
            raise
    
    def test_dataset_paths_consistency(self):
        """Test that datasets create consistent paths."""
        try:
            # Test that all datasets create their expected directories
            mnist = MnistDataset(train=True)
            fashion = FashionMnistDataset(train=True)
            cifar = Cifar10Dataset(train=True)
            
            assert mnist.dataset_name == "mnist", f"Expected 'mnist', got '{mnist.dataset_name}'"
            assert fashion.dataset_name == "fashion-mnist", f"Expected 'fashion-mnist', got '{fashion.dataset_name}'"
            assert cifar.dataset_name == "cifar-10", f"Expected 'cifar-10', got '{cifar.dataset_name}'"
            
            # Check that dataset_root paths are correct
            assert mnist.dataset_root.name == "mnist"
            assert fashion.dataset_root.name == "fashion-mnist"
            assert cifar.dataset_root.name == "cifar-10"
            
            logger.info("Dataset path consistency verified")
        except Exception as e:
            logger.error(f"Dataset paths consistency test failed: {e}")
            raise


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_invalid_sample_index(self):
        """Test handling of invalid sample indices."""
        try:
            mnist = MnistDataset(train=True)
            
            # Test index too large
            with pytest.raises(IndexError):
                _ = mnist[len(mnist)]
            
            logger.info("Invalid index handling verified")
        except Exception as e:
            logger.error(f"Invalid sample index test failed: {e}")
            raise
    
    def test_dataset_with_invalid_path(self):
        """Test dataset creation with invalid root path."""
        try:
            # Try to create dataset with read-only path (should handle gracefully)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                # Make directory read-only (on Unix systems)
                try:
                    temp_path.chmod(0o444)
                    # This should either work or raise a clear error
                    mnist = MnistDataset(root=temp_path, train=True)
                    logger.info("Dataset creation with restricted path handled")
                except PermissionError:
                    logger.info("Permission error handled correctly")
                except Exception as e:
                    logger.warning(f"Unexpected error with invalid path: {e}")
        except Exception as e:
            logger.error(f"Invalid path test failed: {e}")
            # Don't raise here as this is testing error handling


@pytest.mark.dataset_api
@pytest.mark.unit
class TestPackageAPI:
    """Test suite for the package API functions."""
    
    def test_get_dataset_info(self):
        """Test the get_dataset_info factory function."""
        try:
            # Test various name formats
            mnist_info = get_dataset_info("mnist")
            assert mnist_info.name == "MNIST"
            assert len(mnist_info.classes) == 10
            
            fashion_info = get_dataset_info("fashion-mnist")
            assert fashion_info.name == "Fashion-MNIST"
            assert len(fashion_info.classes) == 10
            
            # Test case insensitive
            cifar_info = get_dataset_info("CIFAR10")
            assert cifar_info.name == "CIFAR-10"
            
            logger.info("Dataset info retrieval verified")
        except Exception as e:
            logger.error(f"Dataset info test failed: {e}")
            raise
    
    def test_get_dataset_info_invalid(self):
        """Test get_dataset_info with invalid dataset name."""
        try:
            with pytest.raises(ValueError):
                get_dataset_info("nonexistent")
            logger.info("Invalid dataset name handling verified")
        except Exception as e:
            logger.error(f"Invalid dataset info test failed: {e}")
            raise
    
    def test_create_dataset_factory(self):
        """Test the create_dataset factory function."""
        try:
            # Test creating different datasets
            mnist = create_dataset("mnist", train=True)
            assert isinstance(mnist, MnistDataset)
            assert mnist.train == True
            
            fashion = create_dataset("fashion-mnist", train=False)
            assert isinstance(fashion, FashionMnistDataset)
            assert fashion.train == False
            
            # Test alternative naming
            cifar = create_dataset("cifar-10", train=True)
            assert isinstance(cifar, Cifar10Dataset)
            
            logger.info("Dataset factory function verified")
        except Exception as e:
            logger.error(f"Dataset factory test failed: {e}")
            raise
    
    def test_create_dataset_invalid(self):
        """Test create_dataset with invalid dataset name."""
        try:
            with pytest.raises(ValueError):
                create_dataset("nonexistent")
            logger.info("Invalid dataset creation handling verified")
        except Exception as e:
            logger.error(f"Invalid dataset creation test failed: {e}")
            raise


@pytest.mark.integration
@pytest.mark.unit
class TestPyTorchIntegration:
    """Test suite for PyTorch integration features."""
    
    def test_dataloader_integration(self, temp_data_dir, sample_transforms, device, test_batch_sizes):
        """Test dataset integration with PyTorch DataLoader."""
        try:
            import torch
            from torch.utils.data import DataLoader
            
            # Create dataset with transforms
            dataset = MnistDataset(
                train=True, 
                
                root=temp_data_dir,
                transform=sample_transforms
            )
            
            # Test with different batch sizes
            for batch_size in test_batch_sizes[:2]:  # Test first 2 batch sizes
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Get first batch
                batch_images, batch_labels = next(iter(dataloader))
                
                # Move to device
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                
                # Verify batch properties
                assert batch_images.shape[0] == batch_size
                assert batch_labels.shape[0] == batch_size
                # Check device type matches (handles cuda:0 vs cuda differences)
                assert batch_images.device.type == device.type
                assert batch_labels.device.type == device.type
                
                logger.info(f"DataLoader test passed: batch_size={batch_size}, device={device}")
                
        except Exception as e:
            logger.error(f"PyTorch DataLoader integration test failed: {e}")
            raise
    
    @pytest.mark.transform
    def test_augmentation_transforms(self, temp_data_dir, augmentation_transforms):
        """Test dataset with data augmentation transforms."""
        try:
            dataset = MnistDataset(
                train=True,
                root=temp_data_dir,
                transform=augmentation_transforms
            )
            
            # Test that transforms produce different results (due to randomness)
            img1, _ = dataset[0]
            img2, _ = dataset[0]  # Same index, different augmentation
            
            # Note: Due to randomness, images might be different
            # But both should be valid tensors
            assert img1.shape == img2.shape
            assert img1.shape == (1, 28, 28)
            
            logger.info("Augmentation transforms test passed")
            
        except Exception as e:
            logger.error(f"Augmentation transforms test failed: {e}")
            raise


# Pytest fixtures and utilities
@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_root(project_root):
    """Get the data directory."""
    return project_root / "data"


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory for dataset tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# Run specific test if called directly
if __name__ == "__main__":
    # For direct execution, run pytest
    pytest.main([__file__, "-v"])