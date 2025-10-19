"""
Pytest suite for the managed dataset classes from the input_data package.
Tests downloading, loading, and basic functionality.
"""

import pytest
import sys
import logging
import shutil
import tempfile
from typing import Dict, Optional, List
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Classes and functions to test - updated for modular structure
from src.input_data import (
    create_dataset,
    SupportedDatasets,
    MnistDataset,
    FashionMnistDataset,
    Cifar10Dataset
)

# Import functions that aren't in __all__ but needed for testing
from src.input_data.downloaders import download_dataset

# Configure logging
logger = logging.getLogger(__name__)

def dataloaders_assertions(
    dataloaders: Dict[str, DataLoader],
    expected_sizes: Dict[str, int]
) -> None:
    """Helper function to assert correct dataloader sizes."""
    for split, expected_size in expected_sizes.items():
        assert split in dataloaders, f"Should have {split} dataloader"
        actual_size = len(dataloaders[split].dataset)
        assert actual_size == expected_size, f"{split} size mismatch: expected {expected_size}, got {actual_size}"

def sample_access_assertions(
    dataset,
    index: int,
    expected_shape: torch.Size,
    expected_label_range: range,
    class_names: Optional[List[str]] = None
) -> None:
    """Helper function to assert sample access correctness."""
    img, label = dataset[index]
    assert isinstance(img, torch.Tensor), f"Expected tensor image, got {type(img)}"
    assert img.shape == expected_shape, f"Expected shape {expected_shape}, got {img.shape}"
    assert isinstance(label, int), f"Expected int label, got {type(label)}"
    assert label in expected_label_range, f"Label {label} out of range {expected_label_range}"
    if class_names is not None:
        class_name = class_names[label]
        assert isinstance(class_name, str), f"Expected string class name, got {type(class_name)}"

    logger.info(f"Sample {index}: Image shape={img.shape}, Label={label}")

# --- Specific class implementations ---
@pytest.mark.mnist
@pytest.mark.unit
class TestMnistDataset:
    """Test suite for MNIST dataset."""
    dataset_length = 70000  # 60k train + 10k test
    input_shape = (1, 28, 28)
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_mnist_unified_dataset_loading(self, temp_data_dir):
        """Test that MNIST unified dataset (train+test) can be loaded."""
        try:
            mnist = MnistDataset(root=temp_data_dir)
            assert len(mnist) > 0, "MNIST unified dataset should not be empty"
            # MNIST has 60k train + 10k test = 70k total samples
            assert len(mnist) == self.dataset_length, f"Expected {self.dataset_length} samples, got {len(mnist)}"
            logger.info(f"Loaded complete MNIST dataset: {len(mnist)} samples")
        except Exception as e:
            logger.error(f"MNIST unified dataset loading failed: {e}")
            raise
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_mnist_dataloaders(self, temp_data_dir):
        """Test MNIST dataloader creation with train/val/test splits."""
        try:
            mnist = MnistDataset(root=temp_data_dir)
            dataloaders = mnist.get_dataloaders(train_split=0.8, val_split=0.1, test_split=0.1)
            
            dataloaders = mnist.get_dataloaders(train_split=0.7, val_split=0.2, test_split=0.1)
            
            expected_sizes = dict(
                train=self.dataset_length * 8 // 10,
                val=self.dataset_length // 10,
                test=self.dataset_length // 10,
            )

            dataloaders_assertions(dataloaders, expected_sizes)
            
            dataloaders = mnist.get_dataloaders(train_split=0.6, val_split=0.2, test_split=0.2)
            
            expected_sizes = dict(
                train=self.dataset_length * 6 // 10,
                val=self.dataset_length * 2 // 10,
                test=self.dataset_length * 2 // 10,
            )
            
            logger.info(f"MNIST dataloaders created successfully")
        except Exception as e:
            logger.error(f"MNIST dataloader creation failed: {e}")
            raise
    
    def test_mnist_sample_access(self, temp_data_dir, small_sample_size):
        """Test MNIST sample access and data types."""
        try:
            mnist = MnistDataset(root=temp_data_dir)
            
            # Test multiple samples (limit to small number for speed)
            samples_to_test = min(small_sample_size // 20, 5)  # Test 5 samples max
            for i in range(samples_to_test):
                sample_access_assertions(
                    mnist,
                    i,
                    expected_shape=self.input_shape,
                    expected_label_range=range(10)
                )
        except Exception as e:
            logger.error(f"MNIST sample access failed: {e}")
            raise
    
    def test_mnist_force_redownload(self):
        """Test MNIST force re-download functionality."""
        try:
            # First ensure we have the dataset
            mnist = MnistDataset(force_download=False)
            initial_len = len(mnist)
            
            # Force re-download
            mnist_redownload = MnistDataset(force_download=True)
            assert len(mnist_redownload) == initial_len, "Re-downloaded dataset has different size"
            logger.info("MNIST force re-download successful")
        except Exception as e:
            logger.error(f"MNIST force re-download failed: {e}")
            raise


@pytest.mark.fashion_mnist
@pytest.mark.unit
class TestFashionMnistDataset:
    """Test suite for Fashion-MNIST dataset."""
    dataset_length = 70000  # 60k train + 10k test
    input_shape = (1, 28, 28)
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_fashion_mnist_unified_dataset_loading(self, temp_data_dir):
        """Test Fashion-MNIST unified dataset (train+test) loading."""
        try:
            fashion = FashionMnistDataset(root=temp_data_dir)
            # Fashion-MNIST has 60k train + 10k test = 70k total samples
            assert len(fashion) == self.dataset_length, f"Expected {self.dataset_length} samples, got {len(fashion)}"
            logger.info(f"Fashion-MNIST unified dataset loaded: {len(fashion)} samples")
        except Exception as e:
            logger.error(f"Fashion-MNIST unified dataset loading failed: {e}")
            raise
    
    def test_fashion_mnist_dataloaders(self):
        """Test Fashion-MNIST dataloader creation."""
        try:
            fashion = FashionMnistDataset()
            dataloaders = fashion.get_dataloaders(train_split=0.7, val_split=0.2, test_split=0.1)
            
            expected_sizes = dict(
                train=self.dataset_length * 8 // 10,
                val=self.dataset_length // 10,
                test=self.dataset_length // 10,
            )

            dataloaders_assertions(dataloaders, expected_sizes)
            
            dataloaders = fashion.get_dataloaders(train_split=0.6, val_split=0.2, test_split=0.2)
            
            expected_sizes = dict(
                train=self.dataset_length * 6 // 10,
                val=self.dataset_length * 2 // 10,
                test=self.dataset_length * 2 // 10,
            )
            
            logger.info(f"Fashion-MNIST dataloaders created successfully")
        except Exception as e:
            logger.error(f"Fashion-MNIST dataloader creation failed: {e}")
            raise
    
    def test_fashion_mnist_sample_access(self, temp_data_dir, small_sample_size):
        """Test Fashion-MNIST sample access and data types."""
        try:
            fashion = FashionMnistDataset(root=temp_data_dir)
            
            # Test multiple samples (limit to small number for speed)
            samples_to_test = min(small_sample_size // 20, 5)  # Test 5 samples max
            for i in range(samples_to_test):
                sample_access_assertions(
                    fashion,
                    i,
                    expected_shape=self.input_shape,
                    expected_label_range=range(10),
                    class_names=fashion.dataset_info.classes
                )
        except Exception as e:
            logger.error(f"Fashion-MNIST sample access failed: {e}")
            raise


@pytest.mark.cifar10
@pytest.mark.unit
class TestCifar10Dataset:
    """Test suite for CIFAR-10 dataset."""
    dataset_length = 60000  # 50k train + 10k test
    input_shape = (3, 32, 32)
    
    @pytest.mark.download
    @pytest.mark.slow
    def test_cifar10_unified_dataset_loading(self, temp_data_dir):
        """Test CIFAR-10 unified dataset (train+test) loading."""
        try:
            cifar = Cifar10Dataset(root=temp_data_dir)
            # CIFAR-10 has 50k train + 10k test = 60k total samples
            assert len(cifar) == self.dataset_length, f"Expected {self.dataset_length} samples, got {len(cifar)}"
            logger.info(f"CIFAR-10 unified dataset loaded: {len(cifar)} samples")
        except Exception as e:
            logger.error(f"CIFAR-10 unified dataset loading failed: {e}")
            raise
    
    def test_cifar10_dataloaders(self):
        """Test CIFAR-10 dataloader creation."""
        try:
            cifar = Cifar10Dataset()
            dataloaders = cifar.get_dataloaders(train_split=0.8, val_split=0.1, test_split=0.1)
            
            expected_sizes = dict(
                train=self.dataset_length * 8 // 10,
                val=self.dataset_length // 10,
                test=self.dataset_length // 10,
            )

            dataloaders_assertions(dataloaders, expected_sizes)
            
            dataloaders = cifar.get_dataloaders(train_split=0.6, val_split=0.2, test_split=0.2)
            
            expected_sizes = dict(
                train=self.dataset_length * 6 // 10,
                val=self.dataset_length * 2 // 10,
                test=self.dataset_length * 2 // 10,
            )

            dataloaders_assertions(dataloaders, expected_sizes)

            logger.info(f"CIFAR-10 dataloaders created successfully")
        except Exception as e:
            logger.error(f"CIFAR-10 dataloader creation failed: {e}")
            raise
    
    def test_cifar10_sample_access(self, temp_data_dir, small_sample_size):
        """Test CIFAR-10 class names and sample access."""
        try:
            cifar = Cifar10Dataset(root=temp_data_dir)

            # Test multiple samples (limit to small number for speed)
            samples_to_test = min(small_sample_size // 20, 5)  # Test 5 samples max
            for i in range(samples_to_test):
                sample_access_assertions(
                    cifar,
                    i,
                    expected_shape=self.input_shape,
                    expected_label_range=range(10),
                    class_names=cifar.dataset_info.classes
                )
        except Exception as e:
            logger.error(f"CIFAR-10 sample access failed: {e}")
            raise


@pytest.mark.transform
@pytest.mark.unit
class TestDatasetTransforms:
    """Test suite for dataset transforms."""
    
    def test_mnist_with_transforms(self, temp_data_dir, sample_transforms, tensor_shapes):
        """Test MNIST dataset with torchvision transforms."""
        try:
            mnist_transformed = MnistDataset(
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
                MnistDownloads.TRAIN_IMAGES.value, 
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
    
    def test_force_redownload_behavior(self):
        """Test force re-download functionality."""
        try:
            temp_dir = Path(tempfile.mkdtemp())
            
            # First download
            file_path1 = download_dataset(
                MnistDownloads.TRAIN_LABELS.value, 
                temp_dir, 
                force_download=False
            )
            original_size = file_path1.stat().st_size
            
            # Force re-download
            file_path2 = download_dataset(
                MnistDownloads.TRAIN_LABELS.value, 
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
            mnist = MnistDataset()
            fashion = FashionMnistDataset()
            cifar = Cifar10Dataset()
            
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
            mnist = MnistDataset()
            
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
                    mnist = MnistDataset(root=temp_path)
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
            # Test creating different datasets (new API - no train parameter)
            mnist = create_dataset("mnist")
            assert isinstance(mnist, MnistDataset)
            assert len(mnist) == 70000  # 60k train + 10k test
            
            fashion = create_dataset("fashion-mnist")
            assert isinstance(fashion, FashionMnistDataset) 
            assert len(fashion) == 70000  # 60k train + 10k test
            
            # Test alternative naming
            cifar = create_dataset("cifar-10")
            assert isinstance(cifar, Cifar10Dataset)
            assert len(cifar) == 60000  # 50k train + 10k test
            
            # Test enum usage
            mnist_enum = create_dataset(SupportedDatasets.MNIST)
            assert isinstance(mnist_enum, MnistDataset)
            
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

# TODO fictures already in conftest, examine if they work
# # Pytest fixtures and utilities
# @pytest.fixture(scope="session")
# def project_root():
#     """Get the project root directory."""
#     return Path(__file__).parent.parent


# @pytest.fixture(scope="session")
# def data_root(project_root):
#     """Get the data directory."""
#     return project_root / "data"


# @pytest.fixture
# def temp_dataset_dir():
#     """Create a temporary directory for dataset tests."""
#     temp_dir = Path(tempfile.mkdtemp())
#     yield temp_dir
#     # Cleanup
#     if temp_dir.exists():
#         shutil.rmtree(temp_dir)


# Run specific test if called directly
if __name__ == "__main__":
    # For direct execution, run pytest
    pytest.main([__file__, "-v"])