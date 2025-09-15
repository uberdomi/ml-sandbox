# Dataset Utilities

The `src/utils/datasets.py` module provides comprehensive utilities for downloading and managing common ML datasets. It's inspired by PyTorch Vision's dataset utilities and includes support for popular datasets like MNIST, Fashion-MNIST, CIFAR-10/100, SVHN, and STL-10.

## Features

- **Enum-based dataset definitions**: All dataset information (URLs, checksums, metadata) stored in a convenient enum
- **Automatic integrity checking**: MD5/SHA256 verification for downloaded files
- **Progress bars**: Visual download progress using tqdm
- **Mirror support**: Multiple download URLs for reliability
- **Smart caching**: Avoids re-downloading existing valid files
- **Archive extraction**: Automatic extraction of compressed datasets
- **Comprehensive metadata**: Includes descriptions, citations, and license information

## Available Datasets

### MNIST
- Training and test images/labels
- 60,000 training + 10,000 test examples
- 28x28 grayscale handwritten digits (0-9)

### Fashion-MNIST
- Training and test images/labels  
- 60,000 training + 10,000 test examples
- 28x28 grayscale fashion items (10 categories)

### CIFAR-10/100
- 32x32 color images
- CIFAR-10: 60,000 images in 10 classes
- CIFAR-100: 60,000 images in 100 classes

### SVHN (Street View House Numbers)
- Training, test, and extra sets
- 32x32 color images of house numbers
- Real-world digit recognition dataset

### STL-10
- 96x96 color images in 10 classes
- Designed for unsupervised learning research

## Quick Start

```python
from src.utils.datasets import (
    download_mnist_dataset,
    download_cifar_dataset,
    list_available_datasets,
    CommonDatasets
)

# List all available datasets
list_available_datasets()

# Download MNIST training set
mnist_files = download_mnist_dataset("./data", train=True)
print(f"Downloaded: {mnist_files['images']} and {mnist_files['labels']}")

# Download and extract CIFAR-10
cifar_dir = download_cifar_dataset("./data", cifar_version=10)
print(f"CIFAR-10 extracted to: {cifar_dir}")

# Download by name
from src.utils.datasets import download_by_name
svhn_file = download_by_name("SVHN_TRAIN", "./data")
```

## API Reference

### Core Functions

#### `download_dataset(dataset_info, root, force_download=False, verbose=True)`
Download a single dataset file using DatasetInfo from the enum.

**Parameters:**
- `dataset_info`: CommonDatasets enum value
- `root`: Directory to download to
- `force_download`: Re-download even if file exists
- `verbose`: Print progress information

**Returns:** Path to downloaded file

#### `download_mnist_dataset(root, train=True, force_download=False, verbose=True)`
Download complete MNIST dataset (images + labels).

**Returns:** Dictionary with 'images' and 'labels' keys

#### `download_fashion_mnist_dataset(root, train=True, force_download=False, verbose=True)`
Download complete Fashion-MNIST dataset (images + labels).

**Returns:** Dictionary with 'images' and 'labels' keys

#### `download_cifar_dataset(root, cifar_version=10, force_download=False, verbose=True)`
Download and extract CIFAR dataset.

**Parameters:**
- `cifar_version`: 10 or 100

**Returns:** Path to extracted dataset directory

#### `download_by_name(dataset_name, root, force_download=False, verbose=True)`
Download dataset by name string.

**Parameters:**
- `dataset_name`: Name from CommonDatasets enum (case-insensitive)

### Utility Functions

#### `dataset_exists(dataset_info, root)`
Check if dataset exists and passes integrity verification.

#### `check_integrity(file_path, md5=None, sha256=None)`
Verify file integrity using checksums.

#### `extract_archive(from_path, to_path=None, remove_finished=False)`
Extract various archive formats (.tar, .tar.gz, .zip, .gz, etc.).

#### `list_available_datasets()`
Print information about all available datasets.

## Dataset Information Structure

Each dataset in the `CommonDatasets` enum contains:

```python
@dataclass
class DatasetInfo:
    name: str              # Human-readable name
    urls: list[str]        # Mirror URLs for download
    filename: str          # File name to save as
    md5: Optional[str]     # MD5 checksum for verification
    sha256: Optional[str]  # SHA256 checksum for verification
    file_size: Optional[int]  # File size in bytes
    description: str       # Dataset description
    license: str          # License information
    citation: str         # Citation for academic use
```

## Command Line Usage

The module can be used from command line:

```bash
# List all available datasets
python src/utils/datasets.py list

# Download a specific dataset
python src/utils/datasets.py download MNIST_TRAIN_IMAGES ./data
```

## Example Usage

See `example_datasets.py` for a comprehensive example that demonstrates:

1. Listing available datasets
2. Downloading MNIST training set
3. Downloading Fashion-MNIST test set  
4. Downloading and extracting CIFAR-10
5. Using download-by-name functionality
6. Validating downloaded files

Run the example:

```bash
python example_datasets.py
```

## Error Handling

The utilities include comprehensive error handling:

- **Network errors**: Retry with mirror URLs
- **Integrity failures**: Automatic re-download
- **Disk space**: Clear error messages
- **Permissions**: Helpful troubleshooting info

## Dependencies

- `tqdm`: Progress bars
- `pathlib`: Modern path handling
- Standard library: `urllib`, `hashlib`, `tarfile`, `zipfile`, `gzip`

## Best Practices

1. **Always verify integrity**: The utilities automatically check file integrity
2. **Use mirrors**: Multiple URLs provided for reliability
3. **Cache intelligently**: Files are only re-downloaded if corrupted
4. **Organize data**: Datasets are organized in subdirectories by type
5. **Handle errors**: Robust error handling with informative messages

## Extending the Dataset Collection

To add new datasets, simply extend the `CommonDatasets` enum:

```python
NEW_DATASET = DatasetInfo(
    name="My Dataset",
    urls=["https://example.com/dataset.tar.gz"],
    filename="dataset.tar.gz", 
    md5="checksum_here",
    description="Description of the dataset",
    license="License information",
    citation="Academic citation"
)
```

## Integration with ML Frameworks

The downloaded datasets can be easily integrated with:

- **PyTorch**: Use with custom Dataset classes
- **TensorFlow**: Load with tf.data APIs
- **scikit-learn**: For traditional ML algorithms
- **NumPy/Pandas**: For data analysis and preprocessing

This utility module provides a solid foundation for dataset management in machine learning projects, ensuring reproducible and reliable data access.