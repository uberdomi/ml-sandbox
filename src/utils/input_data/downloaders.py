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

USER_AGENT = "ml-sandbox/1.0"

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