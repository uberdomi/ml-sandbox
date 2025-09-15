"""
Dataset utilities for downloading and managing common ML datasets.
Based on PyTorch Vision utils and adapted for general ML usage.
"""

import os
import shutil
import urllib.request
import urllib.error
import tarfile
import zipfile
import gzip
import hashlib
from pathlib import Path
from enum import Enum
from typing import Optional, Union, Dict, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm


USER_AGENT = "ml-sandbox/1.0"


@dataclass
class DatasetInfo:
    """Information about a dataset including URLs, checksums, and metadata."""
    name: str
    urls: list[str]  # Mirror URLs
    filename: str
    md5: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None  # Size in bytes
    description: str = ""
    license: str = ""
    citation: str = ""


class CommonDatasets(Enum):
    """Enum containing common ML datasets with their download information."""
    
    # --- MNIST Dataset ---
    MNIST_TRAIN_IMAGES = DatasetInfo(
        name="MNIST Training Images",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
        ],
        filename="train-images-idx3-ubyte.gz",
        md5="f68b3c2dcbeaaa9fbdd348bbdeb94873",
        description="MNIST training set images (60,000 examples)",
        license="Creative Commons Attribution-Share Alike 3.0",
        citation="LeCun, Y. (1998). The MNIST database of handwritten digits."
    )
    
    MNIST_TRAIN_LABELS = DatasetInfo(
        name="MNIST Training Labels",
        urls=[
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"
        ],
        filename="train-labels-idx1-ubyte.gz",
        md5="d53e105ee54ea40749a09fcbcd1e9432",
        description="MNIST training set labels (60,000 examples)"
    )
    
    MNIST_TEST_IMAGES = DatasetInfo(
        name="MNIST Test Images", 
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
        ],
        filename="t10k-images-idx3-ubyte.gz",
        md5="9fb629c4189551a2d022fa330f9573f3",
        description="MNIST test set images (10,000 examples)"
    )
    
    MNIST_TEST_LABELS = DatasetInfo(
        name="MNIST Test Labels",
        urls=[
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", 
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
        ],
        filename="t10k-labels-idx1-ubyte.gz",
        md5="ec29112dd5afa0611ce80d1b7f02629c",
        description="MNIST test set labels (10,000 examples)"
    )
    
    # --- Fashion-MNIST Dataset ---
    FASHION_MNIST_TRAIN_IMAGES = DatasetInfo(
        name="Fashion-MNIST Training Images",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
        ],
        filename="train-images-idx3-ubyte.gz",
        md5="8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        description="Fashion-MNIST training images (60,000 examples)",
        license="MIT License",
        citation="Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST."
    )
    
    FASHION_MNIST_TRAIN_LABELS = DatasetInfo(
        name="Fashion-MNIST Training Labels",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
        ],
        filename="train-labels-idx1-ubyte.gz", 
        md5="25c81989df183df01b3e8a0aad5dffbe",
        description="Fashion-MNIST training labels (60,000 examples)"
    )
    
    FASHION_MNIST_TEST_IMAGES = DatasetInfo(
        name="Fashion-MNIST Test Images",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
        ],
        filename="t10k-images-idx3-ubyte.gz",
        md5="bef4ecab320f06d8554ea6380940ec79",
        description="Fashion-MNIST test images (10,000 examples)"
    )
    
    FASHION_MNIST_TEST_LABELS = DatasetInfo(
        name="Fashion-MNIST Test Labels",
        urls=[
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
        ],
        filename="t10k-labels-idx1-ubyte.gz",
        md5="bb300cfdad3c16e7a12a480ee83cd310",
        description="Fashion-MNIST test labels (10,000 examples)"
    )
    
    # --- CIFAR-10 Dataset ---
    CIFAR10 = DatasetInfo(
        name="CIFAR-10",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "https://s3.amazonaws.com/fast-ai-datasets/cifar10.tar.gz"
        ],
        filename="cifar-10-python.tar.gz",
        md5="c58f30108f718f92721af3b95e74349a", 
        description="CIFAR-10 dataset (60,000 32x32 color images in 10 classes)",
        license="Unknown",
        citation="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images."
    )
    
    # --- CIFAR-100 Dataset ---
    CIFAR100 = DatasetInfo(
        name="CIFAR-100",
        urls=[
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        ],
        filename="cifar-100-python.tar.gz",
        md5="eb9058c3a382ffc7106e4002c42a8d85",
        description="CIFAR-100 dataset (60,000 32x32 color images in 100 classes)",
        license="Unknown", 
        citation="Krizhevsky, A. (2009). Learning multiple layers of features from tiny images."
    )
    
    # --- SVHN Dataset ---
    SVHN_TRAIN = DatasetInfo(
        name="SVHN Training Set",
        urls=[
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        ],
        filename="train_32x32.mat",
        md5="e26dedcc434d2e4c54c9b2d4a06d8373",
        description="SVHN training set (73,257 digits)",
        citation="Netzer, Y., et al. (2011). Reading digits in natural images with unsupervised feature learning."
    )
    
    SVHN_TEST = DatasetInfo(
        name="SVHN Test Set", 
        urls=[
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
        ],
        filename="test_32x32.mat",
        md5="eb5a983be6a315427106f1b164d9cef3",
        description="SVHN test set (26,032 digits)"
    )
    
    SVHN_EXTRA = DatasetInfo(
        name="SVHN Extra Set",
        urls=[
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
        ],
        filename="extra_32x32.mat", 
        md5="a93ce644f1a588dc4d68dda5feec44a7",
        description="SVHN extra set (531,131 digits)"
    )
    
    # --- STL-10 Dataset ---
    STL10 = DatasetInfo(
        name="STL-10",
        urls=[
            "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        ],
        filename="stl10_binary.tar.gz",
        md5="91f7769df0f17e558f3565bffb0c7dfb",
        description="STL-10 dataset (96x96 color images, 10 classes)",
        citation="Coates, A., Ng, A., & Lee, H. (2011). STL-10 dataset."
    )


def gen_bar_updater():
    """Create a tqdm progress bar updater for urllib.request.urlretrieve."""
    pbar = tqdm(total=None, unit='B', unit_scale=True, desc="Downloading")

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def calculate_md5(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def calculate_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def check_integrity(file_path: Union[str, Path], md5: Optional[str] = None, 
                   sha256: Optional[str] = None) -> bool:
    """Check the integrity of a file using MD5 or SHA256 checksum."""
    if not os.path.exists(file_path):
        return False
    
    if md5:
        return calculate_md5(file_path) == md5
    elif sha256:
        return calculate_sha256(file_path) == sha256
    else:
        # If no checksum provided, just check if file exists and is not empty
        return os.path.getsize(file_path) > 0


def download_url(url: str, root: Union[str, Path], filename: Optional[str] = None,
                md5: Optional[str] = None, sha256: Optional[str] = None) -> Path:
    """
    Download a file from a URL with progress bar and integrity checking.
    
    Args:
        url: URL to download from
        root: Directory to save the file
        filename: Name to save the file as (defaults to basename of URL)
        md5: Expected MD5 checksum for verification
        sha256: Expected SHA256 checksum for verification
        
    Returns:
        Path to the downloaded file
        
    Raises:
        RuntimeError: If download fails or integrity check fails
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = os.path.basename(url)
    
    file_path = root / filename
    
    # Check if file already exists and is valid
    if check_integrity(file_path, md5, sha256):
        print(f"File {filename} already exists and is valid.")
        return file_path
    
    print(f"Downloading {url} to {file_path}")
    
    try:
        # Add User-Agent header to avoid blocking
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            with open(file_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=f"Downloading {filename}"
            ) as pbar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    
    # Verify integrity
    if not check_integrity(file_path, md5, sha256):
        os.remove(file_path)
        raise RuntimeError(f"Downloaded file {filename} is corrupted or incomplete.")
    
    return file_path


def _is_tarxz(filename: str) -> bool:
    """Check if filename is a .tar.xz archive."""
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    """Check if filename is a .tar archive."""
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    """Check if filename is a .tar.gz archive.""" 
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    """Check if filename is a .tgz archive."""
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    """Check if filename is a .gz file (but not .tar.gz)."""
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    """Check if filename is a .zip archive."""
    return filename.endswith(".zip")


def extract_archive(from_path: Union[str, Path], 
                   to_path: Optional[Union[str, Path]] = None,
                   remove_finished: bool = False) -> None:
    """
    Extract an archive file.
    
    Args:
        from_path: Path to the archive file
        to_path: Directory to extract to (defaults to parent directory of archive)
        remove_finished: Whether to delete the archive after extraction
        
    Raises:
        ValueError: If archive format is not supported
        FileNotFoundError: If archive file doesn't exist
    """
    from_path = Path(from_path)
    
    if not from_path.exists():
        raise FileNotFoundError(f"Archive file not found: {from_path}")
    
    if to_path is None:
        to_path = from_path.parent
    else:
        to_path = Path(to_path)
        to_path.mkdir(parents=True, exist_ok=True)
    
    filename = from_path.name
    
    print(f"Extracting {filename}...")
    
    try:
        if _is_tar(filename):
            with tarfile.open(from_path, 'r') as tar:
                tar.extractall(path=to_path)
        elif _is_targz(filename) or _is_tgz(filename):
            with tarfile.open(from_path, 'r:gz') as tar:
                tar.extractall(path=to_path)
        elif _is_tarxz(filename):
            with tarfile.open(from_path, 'r:xz') as tar:
                tar.extractall(path=to_path)
        elif _is_gzip(filename):
            # For single .gz files, decompress to same directory
            output_path = to_path / from_path.stem
            with open(output_path, "wb") as out_f, gzip.GzipFile(from_path) as gz_f:
                shutil.copyfileobj(gz_f, out_f)
        elif _is_zip(filename):
            with zipfile.ZipFile(from_path, 'r') as zip_f:
                zip_f.extractall(to_path)
        else:
            raise ValueError(f"Extraction of {filename} not supported. "
                           "Supported formats: .tar, .tar.gz, .tgz, .tar.xz, .gz, .zip")
                           
    except Exception as e:
        raise RuntimeError(f"Failed to extract {filename}: {e}")
    
    if remove_finished:
        from_path.unlink()
        print(f"Removed archive: {filename}")


def download_and_extract_archive(url: str, download_root: Union[str, Path],
                                extract_root: Optional[Union[str, Path]] = None,
                                filename: Optional[str] = None, 
                                md5: Optional[str] = None,
                                sha256: Optional[str] = None,
                                remove_finished: bool = False) -> None:
    """
    Download and extract an archive in one step.
    
    Args:
        url: URL to download from
        download_root: Directory to download the archive to
        extract_root: Directory to extract to (defaults to download_root)
        filename: Name to save the archive as
        md5: Expected MD5 checksum
        sha256: Expected SHA256 checksum  
        remove_finished: Whether to delete the archive after extraction
    """
    download_root = Path(download_root)
    if extract_root is None:
        extract_root = download_root
    
    # Download the archive
    archive_path = download_url(url, download_root, filename, md5, sha256)
    
    # Extract the archive
    extract_archive(archive_path, extract_root, remove_finished)


def dataset_exists(dataset_info: DatasetInfo, root: Union[str, Path]) -> bool:
    """
    Check if a dataset file already exists and is valid.
    
    Args:
        dataset_info: DatasetInfo enum value containing file information
        root: Directory where the dataset should be located
        
    Returns:
        True if dataset exists and passes integrity check
    """
    root = Path(root)
    file_path = root / dataset_info.value.filename
    
    return check_integrity(file_path, dataset_info.value.md5, dataset_info.value.sha256)


def download_dataset(dataset_info: DatasetInfo, root: Union[str, Path],
                    force_download: bool = False, verbose: bool = True) -> Path:
    """
    Download a dataset using the information from CommonDatasets enum.
    
    Args:
        dataset_info: DatasetInfo enum value
        root: Directory to download to
        force_download: Whether to re-download even if file exists
        verbose: Whether to print progress information
        
    Returns:
        Path to the downloaded file
        
    Raises:
        RuntimeError: If all download attempts fail
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    file_path = root / dataset_info.value.filename
    
    # Check if already exists and valid
    if not force_download and dataset_exists(dataset_info, root):
        if verbose:
            print(f"Dataset {dataset_info.value.name} already exists and is valid: {file_path}")
        return file_path
    
    if verbose:
        print(f"Downloading {dataset_info.value.name}...")
        if dataset_info.value.description:
            print(f"Description: {dataset_info.value.description}")
    
    # Try each mirror URL until one succeeds
    last_error = None
    for url in dataset_info.value.urls:
        try:
            return download_url(
                url, root, dataset_info.value.filename,
                dataset_info.value.md5, dataset_info.value.sha256
            )
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Failed to download from {url}: {e}")
            continue
    
    # If we get here, all downloads failed
    raise RuntimeError(f"Failed to download {dataset_info.value.name} from all mirrors. "
                      f"Last error: {last_error}")


def download_mnist_dataset(root: Union[str, Path], train: bool = True, 
                          force_download: bool = False, verbose: bool = True) -> Dict[str, Path]:
    """
    Download complete MNIST dataset (images and labels).
    
    Args:
        root: Directory to download to
        train: Whether to download training set (True) or test set (False)
        force_download: Whether to re-download even if files exist
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with 'images' and 'labels' keys pointing to downloaded files
    """
    root = Path(root) / "mnist"
    
    if train:
        images_info = CommonDatasets.MNIST_TRAIN_IMAGES
        labels_info = CommonDatasets.MNIST_TRAIN_LABELS
    else:
        images_info = CommonDatasets.MNIST_TEST_IMAGES  
        labels_info = CommonDatasets.MNIST_TEST_LABELS
    
    # Download both files
    images_path = download_dataset(images_info, root, force_download, verbose)
    labels_path = download_dataset(labels_info, root, force_download, verbose)
    
    return {
        'images': images_path,
        'labels': labels_path
    }


def download_fashion_mnist_dataset(root: Union[str, Path], train: bool = True,
                                  force_download: bool = False, verbose: bool = True) -> Dict[str, Path]:
    """
    Download complete Fashion-MNIST dataset (images and labels).
    
    Args:
        root: Directory to download to
        train: Whether to download training set (True) or test set (False) 
        force_download: Whether to re-download even if files exist
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with 'images' and 'labels' keys pointing to downloaded files
    """
    root = Path(root) / "fashion-mnist"
    
    if train:
        images_info = CommonDatasets.FASHION_MNIST_TRAIN_IMAGES
        labels_info = CommonDatasets.FASHION_MNIST_TRAIN_LABELS
    else:
        images_info = CommonDatasets.FASHION_MNIST_TEST_IMAGES
        labels_info = CommonDatasets.FASHION_MNIST_TEST_LABELS
    
    # Download both files
    images_path = download_dataset(images_info, root, force_download, verbose)
    labels_path = download_dataset(labels_info, root, force_download, verbose)
    
    return {
        'images': images_path, 
        'labels': labels_path
    }


def download_cifar_dataset(root: Union[str, Path], cifar_version: int = 10,
                          force_download: bool = False, verbose: bool = True) -> Path:
    """
    Download and extract CIFAR dataset.
    
    Args:
        root: Directory to download and extract to
        cifar_version: CIFAR version (10 or 100)
        force_download: Whether to re-download even if files exist
        verbose: Whether to print progress information
        
    Returns:
        Path to the extracted dataset directory
        
    Raises:
        ValueError: If cifar_version is not 10 or 100
    """
    if cifar_version == 10:
        dataset_info = CommonDatasets.CIFAR10
        extract_dir = "cifar-10-batches-py"
    elif cifar_version == 100:
        dataset_info = CommonDatasets.CIFAR100
        extract_dir = "cifar-100-python"
    else:
        raise ValueError("cifar_version must be 10 or 100")
    
    root = Path(root) / f"cifar-{cifar_version}"
    extract_path = root / extract_dir
    
    # Check if already extracted
    if not force_download and extract_path.exists() and any(extract_path.iterdir()):
        if verbose:
            print(f"CIFAR-{cifar_version} dataset already exists: {extract_path}")
        return extract_path
    
    # Download and extract
    download_and_extract_archive(
        dataset_info.value.urls[0], root, root,
        dataset_info.value.filename, dataset_info.value.md5,
        remove_finished=True
    )
    
    if verbose:
        print(f"CIFAR-{cifar_version} dataset extracted to: {extract_path}")
    
    return extract_path


def list_available_datasets() -> None:
    """Print information about all available datasets."""
    print("Available datasets in CommonDatasets enum:")
    print("=" * 50)
    
    # Group datasets by type
    dataset_groups = {}
    for dataset in CommonDatasets:
        name_parts = dataset.name.split('_')
        dataset_type = name_parts[0]
        if dataset_type not in dataset_groups:
            dataset_groups[dataset_type] = []
        dataset_groups[dataset_type].append(dataset)
    
    for group_name, datasets in dataset_groups.items():
        print(f"\n{group_name}:")
        for dataset in datasets:
            info = dataset.value
            print(f"  • {info.name}")
            if info.description:
                print(f"    Description: {info.description}")
            print(f"    Filename: {info.filename}")
            if info.md5:
                print(f"    MD5: {info.md5}")
            if info.citation:
                print(f"    Citation: {info.citation}")
            print()


# Convenience functions for common operations
def get_dataset_info(dataset_name: str) -> Optional[DatasetInfo]:
    """
    Get dataset information by name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        
    Returns:
        DatasetInfo object if found, None otherwise
    """
    dataset_name = dataset_name.upper().replace('-', '_')
    
    try:
        return getattr(CommonDatasets, dataset_name).value
    except AttributeError:
        return None


def download_by_name(dataset_name: str, root: Union[str, Path],
                    force_download: bool = False, verbose: bool = True) -> Optional[Path]:
    """
    Download a dataset by name.
    
    Args:
        dataset_name: Name of the dataset from CommonDatasets enum
        root: Directory to download to
        force_download: Whether to re-download even if file exists
        verbose: Whether to print progress information
        
    Returns:
        Path to downloaded file if successful, None if dataset not found
    """
    dataset_name = dataset_name.upper().replace('-', '_')
    
    try:
        dataset_enum = getattr(CommonDatasets, dataset_name)
        return download_dataset(dataset_enum, root, force_download, verbose)
    except AttributeError:
        if verbose:
            print(f"Dataset '{dataset_name}' not found in CommonDatasets enum.")
            print("Use list_available_datasets() to see available datasets.")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_available_datasets()
        elif sys.argv[1] == "download" and len(sys.argv) > 3:
            dataset_name = sys.argv[2]
            root_dir = sys.argv[3]
            result = download_by_name(dataset_name, root_dir, verbose=True)
            if result:
                print(f"Successfully downloaded to: {result}")
        else:
            print("Usage:")
            print("  python datasets.py list")
            print("  python datasets.py download <dataset_name> <root_directory>")
    else:
        print("ML Dataset Utilities")
        print("Available commands:")
        print("  list - Show all available datasets")
        print("  download <name> <dir> - Download a specific dataset")