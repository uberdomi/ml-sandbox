"""
Utilities for downloading and verifying files.
"""

import os
import urllib.request
import urllib.error
import hashlib
from pathlib import Path
from typing import Optional, Union
from tqdm import tqdm

USER_AGENT = "ml-sandbox/1.0"

def gen_bar_updater():
    """Create a tqdm progress bar updater for urllib.request.urlretrieve."""
    pbar = tqdm(total=None, unit='B', unit_scale=True, unit_divisor=1024, desc="Downloading")

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    # Attach the progress bar to the function so we can close it later
    bar_update.pbar = pbar
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


def download_url(
    url: str,
    root: Union[str, Path],
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    sha256: Optional[str] = None
) -> Path:
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
        progress_hook = gen_bar_updater()
        
        # Add User-Agent header to avoid blocking by creating an opener
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', USER_AGENT)]
        urllib.request.install_opener(opener)
        
        # Download with progress bar
        urllib.request.urlretrieve(url, str(file_path), reporthook=progress_hook)
        
        # Close the progress bar
        progress_hook.pbar.close()

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