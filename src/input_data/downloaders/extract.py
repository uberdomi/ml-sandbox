"""
Utilities for extracting common ML datasets.
"""

import shutil
import tarfile
import zipfile
import gzip
from pathlib import Path
from typing import Optional, Union

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


def extract_archive(
    from_path: Union[str, Path], 
    to_path: Optional[Union[str, Path]] = None,
    remove_finished: bool = False
) -> None:
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