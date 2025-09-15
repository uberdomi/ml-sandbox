# 🧪 Test Suite for ML Sandbox Dataset Utilities

This directory contains comprehensive tests for the ML Sandbox dataset utilities package. The test suite covers dataset loading, data transformations, PyTorch integration, error handling, and API functionality.

## 📋 **Test Structure**

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py          # Pytest configuration and shared fixtures
├── test_datasets.py     # Main test suite for dataset functionality
└── README.md           # This documentation
```

## 🚀 **Quick Start**

### Run All Tests
```bash
# Run all tests with basic output
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run all tests quietly (just dots and summary)
pytest tests/ -q
```

### Run Specific Test Categories

```bash
# Run only unit tests (fast)
pytest tests/ -m "unit"

# Run only MNIST-related tests
pytest tests/ -m "mnist"

# Run transformation tests
pytest tests/ -m "transform"

# Run API tests
pytest tests/ -m "dataset_api"
```

### Skip Slow or Download Tests

```bash
# Skip slow tests (recommended for development)
pytest tests/ -m "not slow"

# Skip download tests (when offline)
pytest tests/ -m "not download"

# Skip both slow and download tests
pytest tests/ -m "not slow and not download"
```

## 🏷️ **Test Markers**

Our test suite uses markers to categorize and filter tests:

### **Speed Markers**
- `slow` - Tests that take longer to run (usually involving downloads)
- `download` - Tests that require internet connection

### **Category Markers**
- `unit` - Unit tests (isolated functionality)
- `integration` - Integration tests (multiple components)

### **Dataset Markers**
- `mnist` - Tests specific to MNIST dataset
- `fashion_mnist` - Tests specific to Fashion-MNIST dataset
- `cifar10` - Tests specific to CIFAR-10 dataset

### **Feature Markers**
- `transform` - Tests for data transformations
- `dataset_api` - Tests for API functions (create_dataset, get_dataset_info)
- `error_handling` - Tests for error conditions and edge cases

## 🔍 **Advanced Usage**

### Performance Analysis
```bash
# Show timing for 10 slowest tests
pytest tests/ --durations=10

# Show timing for all tests
pytest tests/ --durations=0
```

### Verbose Output Options
```bash
# Show detailed output for all tests
pytest tests/ -v

# Show extra summary info
pytest tests/ -ra

# Show detailed output + extra summary
pytest tests/ -v -ra

# Show detailed traceback on failures
pytest tests/ --tb=long
```

### Running Specific Tests
```bash
# Run specific test class
pytest tests/test_datasets.py::TestMnistDataset -v

# Run specific test method
pytest tests/test_datasets.py::TestMnistDataset::test_mnist_sample_access -v

# Run multiple specific test classes
pytest tests/test_datasets.py::TestMnistDataset tests/test_datasets.py::TestPackageAPI -v
```

### Filtering by Test Names
```bash
# Run tests with "mnist" in the name
pytest tests/ -k "mnist"

# Run tests with "transform" or "api" in the name
pytest tests/ -k "transform or api"

# Run tests excluding error handling
pytest tests/ -k "not error"
```

### Custom Options
```bash
# Skip download tests (custom option)
pytest tests/ --no-download

# Enable warnings
pytest tests/ -W default

# Disable output capture (see print statements)
pytest tests/ -s
```

## 🛠️ **Development Workflow**

### During Development (Fast Tests)
```bash
# Quick feedback loop - skip slow tests
pytest tests/ -m "not slow and not download" -q
```

### Before Commit (Medium Tests)
```bash
# Include unit tests but skip downloads
pytest tests/ -m "not download" -v
```

### CI/Full Testing (All Tests)
```bash
# Run everything including downloads
pytest tests/ -v --durations=5
```

## 📊 **Test Coverage by Component**

### **Dataset Classes** (`TestMnistDataset`, `TestFashionMnistDataset`, `TestCifar10Dataset`)
- ✅ Dataset loading (train/test splits)
- ✅ Sample access and data types
- ✅ Force re-download functionality
- ✅ Class name validation

### **Data Transformations** (`TestDatasetTransforms`)
- ✅ PyTorch transforms integration
- ✅ Custom transforms
- ✅ Transform chaining
- ✅ Augmentation transforms

### **Download Utilities** (`TestDatasetDownloaders`)
- ✅ URL downloading with progress bars
- ✅ Archive extraction
- ✅ Integrity checking
- ✅ Mirror fallback

### **API Functions** (`TestPackageAPI`)
- ✅ `get_dataset_info()` factory function
- ✅ `create_dataset()` factory function
- ✅ Error handling for invalid dataset names

### **PyTorch Integration** (`TestPyTorchIntegration`)
- ✅ DataLoader integration
- ✅ Device handling (CPU/GPU)
- ✅ Batch processing
- ✅ Transform pipelines

### **Error Handling** (`TestErrorHandling`)
- ✅ Invalid sample indices
- ✅ Invalid file paths
- ✅ Network failures
- ✅ Corrupted data handling

## 🧩 **Fixtures Available**

### **Paths and Directories**
- `project_root` - Project root directory
- `data_root` - Data directory path
- `temp_data_dir` - Temporary directory for test data

### **PyTorch Utilities**
- `device` - Best available device (CUDA/CPU)
- `sample_transforms` - Standard transforms for grayscale datasets
- `cifar_transforms` - Transforms for RGB datasets
- `augmentation_transforms` - Data augmentation transforms

### **Test Configuration**
- `tensor_shapes` - Expected tensor shapes for datasets
- `test_batch_sizes` - Common batch sizes for testing
- `small_sample_size` - Small sample count for quick tests
- `dataset_config` - Default dataset configuration

## 🐛 **Debugging Tests**

### Show Output from Tests
```bash
# See print statements and logging
pytest tests/ -s

# Capture=no (see all output)
pytest tests/ --capture=no
```

### Debug Specific Test
```bash
# Run single test with full output
pytest tests/test_datasets.py::TestMnistDataset::test_mnist_sample_access -v -s

# Add pdb debugging
pytest tests/ --pdb
```

### Check Test Discovery
```bash
# See what tests would be run
pytest tests/ --collect-only

# See what tests match a marker
pytest tests/ -m "mnist" --collect-only
```

## 📈 **Performance Guidelines**

### **Fast Tests** (< 1 second)
- Unit tests for API functions
- Tensor shape validation
- Error condition testing

### **Medium Tests** (1-10 seconds)
- Dataset sample access
- Transform application
- Small dataset operations

### **Slow Tests** (> 10 seconds)
- Full dataset downloads
- Large dataset processing
- Integration with external services

## 🔧 **Configuration**

Test configuration is managed through:

1. **`pyproject.toml`** - Main pytest configuration
2. **`conftest.py`** - Fixtures and test setup
3. **Environment variables** - Custom test behavior

### Key Configuration Options

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "download: marks tests requiring internet", 
    # ... more markers
]
addopts = [
    "--durations=5",        # Show 5 slowest tests
    "--strict-markers",     # Enforce marker registration
    "--tb=short",          # Shorter tracebacks
]
```

## 🤝 **Contributing**

When adding new tests:

1. **Use appropriate markers** - Mark tests with relevant categories
2. **Use fixtures** - Leverage shared fixtures instead of duplicating setup
3. **Handle errors gracefully** - Log errors before re-raising
4. **Keep tests focused** - One concept per test method
5. **Add docstrings** - Describe what each test verifies

### Example New Test
```python
@pytest.mark.unit
@pytest.mark.your_feature
class TestYourFeature:
    """Test suite for your new feature."""
    
    def test_your_functionality(self, temp_data_dir, sample_transforms):
        """Test your specific functionality."""
        try:
            # Your test code here
            result = your_function(temp_data_dir, sample_transforms)
            assert result is not None
            logger.info("Your test passed")
        except Exception as e:
            logger.error(f"Your test failed: {e}")
            raise
```

## 📚 **Additional Resources**

- [Pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/notes/unittest.html)
- [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)

---

**Happy Testing! 🎯**