"""
Dataset utilities for downloading and extracting common ML datasets.
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
from typing import Optional, Union, Dict, Tuple, Any
from tqdm import tqdm

from src.utils.input_data.enums import DatasetInfo

USER_AGENT = "ml-sandbox/1.0"

def gen_bar_updater():
    """Create a tqdm progress bar updater for urllib.request.urlretrieve."""
    pbar = tqdm(total=None, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading")

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
    
    print(f"Downloading {filename} from {url}")
    
    try:
        # Create progress bar updater
        progress_updater = gen_bar_updater()
        
        # Add User-Agent header to avoid blocking by creating an opener
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', USER_AGENT)]
        urllib.request.install_opener(opener)
        
        # Download with progress bar
        urllib.request.urlretrieve(url, str(file_path), reporthook=progress_updater)
        
        # Close the progress bar
        progress_updater.__self__.close()
                    
    except urllib.error.URLError as e:
        if file_path.exists():
            file_path.unlink()  # Remove partial download
        raise RuntimeError(f"Failed to download {url}: {e}")
    except Exception as e:
        if file_path.exists():
            file_path.unlink()  # Remove partial download
        raise RuntimeError(f"Unexpected error downloading {url}: {e}")
    
    # Verify integrity
    if not check_integrity(file_path, md5, sha256):
        file_path.unlink()  # Remove corrupted file
        raise RuntimeError(f"Downloaded file {filename} is corrupted or incomplete.")
    
    print(f"✅ Successfully downloaded and verified {filename}")
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
    file_path = root / dataset_info.filename
    
    return check_integrity(file_path, dataset_info.md5, dataset_info.sha256)


def download_dataset(dataset_info: DatasetInfo, root: Union[str, Path],
                    force_download: bool = False, verbose: bool = True) -> Path:
    """
    Download a dataset using the information from DatasetInfo enum.
    
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
    
    file_path = root / dataset_info.filename
    
    # Check if already exists and valid
    if not force_download and dataset_exists(dataset_info, root):
        if verbose:
            print(f"Dataset {dataset_info.name} already exists and is valid: {file_path}")
        return file_path
    
    if verbose:
        print(f"Downloading {dataset_info.name}...")
        if dataset_info.description:
            print(f"Description: {dataset_info.description}")
    
    # Try each mirror URL until one succeeds
    last_error = None
    for url in dataset_info.urls:
        try:
            return download_url(
                url, root, dataset_info.filename,
                dataset_info.md5, dataset_info.sha256
            )
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Failed to download from {url}: {e}")
            continue
    
    # If we get here, all downloads failed
    raise RuntimeError(f"Failed to download {dataset_info.name} from all mirrors. "
                      f"Last error: {last_error}")