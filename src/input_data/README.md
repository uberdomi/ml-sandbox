# Input Data Package 📦

A comprehensive, modern Python package for managing and loading popular ML datasets with automatic downloading, integrity verification, and PyTorch integration.

## 🌟 Key Features

### 🔄 Unified Dataset Architecture

- **All-in-One Loading**: Loads complete datasets (train + test data combined) into a single unified dataset
- **Flexible Splits**: Use `get_dataloaders()` to create custom train/validation/test splits from the unified data
- **No Train/Test Separation**: Eliminates the old `train=True/False` pattern in favor of flexible runtime splits

### 🚀 Modern API Design

- **Primary API**: `create_dataset()` function with enum or string-based dataset selection
- **Type Safety**: Full type hints and dataclass-based configuration
- **Multiple Access Methods**: Support for enums, strings, and alternative dataset names

### 🧩 Modular Architecture

- **Clean Separation**: Each dataset in its own module (`mnist.py`, `fashion_mnist.py`, `cifar10.py`)
- **Base Classes**: Common functionality in `base.py` with abstract base class `ManagedDataset`
- **Flexible Imports**: Use the main API or import specific datasets directly

### 📊 Rich Dataset Information

- **Metadata Access**: Comprehensive dataset information including classes, shapes, licenses
- **Print Methods**: Built-in `print_info()` for detailed dataset summaries  
- **Class Information**: Easy access to class names and dataset statistics

### 🎨 Visualization & Analysis

- **Sample Visualization**: `show_random_samples()` and `show_illustrative_samples()` methods
- **PyTorch Integration**: Native PyTorch tensor support with transform pipelines
- **Flexible DataLoaders**: Built-in dataloader creation with customizable parameters

## 📋 Supported Datasets

| Dataset | Size | Classes | Shape | Description |
|---------|------|---------|-------|-------------|
| **MNIST** | 70,000 | 10 | (1, 28, 28) | Handwritten digits |
| **Fashion-MNIST** | 70,000 | 10 | (1, 28, 28) | Fashion items |
| **CIFAR-10** | 60,000 | 10 | (3, 32, 32) | Natural images |

## 🚀 Quick Start

### Basic Usage

```python
from input_data import create_dataset, SupportedDatasets

# Create a dataset (loads all data: train + test)
dataset = create_dataset(SupportedDatasets.MNIST)
print(f"Total samples: {len(dataset):,}")  # 70,000

# Get flexible dataloaders
dataloaders = dataset.get_dataloaders(
    train_split=0.7,    # 70% for training
    val_split=0.2,      # 20% for validation  
    test_split=0.1,     # 10% for testing
    batch_size=32
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
test_loader = dataloaders['test']
```

### Alternative Creation Methods

```python
# Using string names
mnist = create_dataset("mnist")
fashion = create_dataset("fashion-mnist")  # Alternative names supported
cifar = create_dataset("cifar-10")

# Using enums (recommended for type safety)
mnist = create_dataset(SupportedDatasets.MNIST)
```

### Dataset Information

```python
# Print comprehensive info
dataset.print_info()

# Access metadata programmatically
info = dataset.dataset_info
print(f"Classes: {info.classes}")
print(f"Shape: {info.input_shape}")
print(f"License: {info.license}")
```

### Sample Visualization

```python
# Show random samples
dataset.show_random_samples(num_samples=16)

# Show representative samples from each class
dataset.show_illustrative_samples(samples_per_class=3)
```

## 🔧 Advanced Usage

### Custom Data Transforms

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

dataset = create_dataset("mnist", transform=transform)
```

### Custom Storage Location

```python
dataset = create_dataset("mnist", root="/path/to/data")
```

### Force Re-download

```python
dataset = create_dataset("mnist", force_download=True)
```

### Direct Module Imports

```python
# Import specific datasets directly
from input_data.mnist import MnistDataset, MNIST_INFO
from input_data.fashion_mnist import FashionMnistDataset
from input_data.cifar10 import Cifar10Dataset

# Create dataset directly
mnist = MnistDataset(root="data")

# Access info constants
print(MNIST_INFO.name)  # "MNIST"
print(MNIST_INFO.classes)  # ["0", "1", "2", ...]
```

## 🏗️ Architecture Overview

### Package Structure

```text
input_data/
├── __init__.py          # Main API and exports
├── base.py              # ManagedDataset abstract base class  
├── enums.py             # DatasetDownloads and DatasetInfo dataclasses
├── downloaders.py       # Download utilities
├── mnist.py             # MNIST dataset implementation
├── fashion_mnist.py     # Fashion-MNIST dataset implementation
├── cifar10.py           # CIFAR-10 dataset implementation
└── datasets.py          # Legacy compatibility layer
```

### Class Hierarchy

```python
ManagedDataset (ABC)
├── MnistDataset
├── FashionMnistDataset  
└── Cifar10Dataset
```

### Key Classes

- **`ManagedDataset`**: Abstract base class with common functionality
- **`DatasetDownloads`**: Dataclass for download URLs, checksums, and metadata
- **`DatasetInfo`**: Dataclass for dataset metadata (classes, shapes, licenses)
- **`SupportedDatasets`**: Enum for type-safe dataset selection

## 🔍 API Reference

### Main Functions

#### `create_dataset(dataset, **kwargs) -> ManagedDataset`

Create a dataset instance.

**Parameters:**

- `dataset`: `SupportedDatasets` enum or string name
- `root`: Data storage directory (default: "data")
- `transform`: Optional transform function
- `target_transform`: Optional target transform function  
- `force_download`: Force re-download even if data exists

**Returns:** Dataset instance with unified train+test data

#### `get_dataset_info(dataset) -> DatasetInfo`

Get dataset metadata without creating dataset instance.

#### `list_supported_datasets() -> List[str]`

List all supported dataset names.

### Dataset Methods

#### `get_dataloaders(train_split=0.8, val_split=0.1, test_split=0.1, **kwargs) -> Dict[str, DataLoader]`

Create train/validation/test dataloaders from unified dataset.

**Parameters:**

- `train_split`, `val_split`, `test_split`: Split ratios (must sum to 1.0)
- `batch_size`: Batch size for all loaders (default: 32)
- `shuffle`: Shuffle training data (default: True)
- `random_seed`: Seed for reproducible splits (default: 42)
- `num_workers`: Number of worker processes (default: 0)

**Returns:** Dictionary with 'train', 'val', 'test' DataLoader objects

#### `print_info() -> None`

Print comprehensive dataset information.

#### `show_random_samples(num_samples=8, figsize=(12, 8)) -> None`

Display random samples from the dataset.

#### `show_illustrative_samples(samples_per_class=3, figsize=(15, 10)) -> None`

Display representative samples from each class.

### Properties

- `dataset_name`: String identifier for the dataset
- `dataset_info`: DatasetInfo object with metadata
- `dataset_root`: Path where dataset files are stored

## 🧪 Testing

The package includes comprehensive tests covering all functionality:

```bash
# Run all tests
python -m pytest tests/test_datasets.py -v

# Run specific test categories
python -m pytest tests/test_datasets.py -k "mnist" -v
python -m pytest tests/test_datasets.py -k "create_dataset" -v
```

## 📊 Usage Examples

### Complete ML Workflow

```python
from input_data import create_dataset, SupportedDatasets
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create dataset
dataset = create_dataset(SupportedDatasets.MNIST)

# 2. Get dataloaders  
loaders = dataset.get_dataloaders(
    train_split=0.8, val_split=0.1, test_split=0.1,
    batch_size=64, num_workers=4
)

# 3. Create a simple model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 4. Train (example)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for batch_idx, (data, target) in enumerate(loaders['train']):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Data Exploration

```python
from input_data import create_dataset

# Load and explore dataset
dataset = create_dataset("fashion-mnist")

# Print detailed information
dataset.print_info()

# Show random samples
dataset.show_random_samples(num_samples=16)

# Show samples from each class
dataset.show_illustrative_samples(samples_per_class=2)

# Access individual samples
image, label = dataset[0]
print(f"First sample: shape={image.shape}, label={label}")

# Get class information
info = dataset.dataset_info  
print(f"Classes: {info.classes}")
print(f"Class {label}: {info.classes[label]}")
```

## 🔄 Migration from Old API

If you're upgrading from the old train/test split API:

### Old API ❌

```python
# OLD: Separate train/test datasets
train_dataset = MnistDataset(train=True)
test_dataset = MnistDataset(train=False)

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
```

### New API ✅

```python
# NEW: Unified dataset with flexible splits
dataset = create_dataset("mnist")

loaders = dataset.get_dataloaders(
    train_split=0.8, val_split=0.1, test_split=0.1,
    batch_size=32
)

train_loader = loaders['train']
val_loader = loaders['val'] 
test_loader = loaders['test']
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib (for visualization)
- Pillow (for image processing)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original dataset creators and maintainers
- PyTorch team for the excellent framework
- Contributors to the machine learning community

---

Happy Dataset Management! 🚀
