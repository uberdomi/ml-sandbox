
"""
Dataset utilities for downloading and extracting common ML datasets.
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from .fetch import download_url, check_integrity
from .extract import extract_archive

@dataclass
class DownloadInfo:
    """Information for downloading dataset files - URLs, checksums, and metadata."""
    name: str
    urls: list[str]  # Mirror URLs
    filename: str
    md5: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None  # Size in bytes
    description: str = ""

    def print(self) -> str:
        """Return a string summary of the download information."""
        lines = [
            f"Dataset Download: {self.name}",
            f"  Filename: {self.filename}",
            f"  URLs: {len(self.urls)} mirror(s)",
        ]
        for i, url in enumerate(self.urls, 1):
            lines.append(f"    {i}. {url}")
        if self.md5:
            lines.append(f"  MD5: {self.md5}")
        if self.sha256:
            lines.append(f"  SHA256: {self.sha256}")
        if self.file_size:
            lines.append(f"  File Size: {self.file_size:,} bytes")
        if self.description:
            lines.append(f"  Description: {self.description}")
        return "\n".join(lines)

def download_and_extract_archive(
    url: str,
    download_root: Union[str, Path],
    extract_root: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None, 
    md5: Optional[str] = None,
    sha256: Optional[str] = None,
    remove_finished: bool = False
) -> None:
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


def dataset_exists(download_info: DownloadInfo, root: Union[str, Path]) -> bool:
    """
    Check if a dataset file already exists and is valid.
    
    Args:
        download_info: DownloadInfo object containing necessary information
        root: Directory where the dataset should be located
        
    Returns:
        True if dataset exists and passes integrity check
    """
    root = Path(root)
    
    file_path = root / download_info.filename
    
    return check_integrity(file_path, download_info.md5, download_info.sha256)

def download_and_extract_dataset(
    download_info: DownloadInfo,
    download_root: Union[str, Path],
    extract_root: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    remove_finished: bool = False,
    verbose: bool = True
) -> None:
    """
    Download and extract a dataset based on the information provided in DownloadInfo.
    
    Args:
        download_info: DownloadInfo object containing necessary information
        download_root: Directory to download to
        extract_root: Directory to extract to (defaults to download_root)
        force_download: Whether to re-download even if file exists
        remove_finished: Whether to delete the archive after extraction
        verbose: Whether to print progress information
    """
    download_root = Path(download_root)
    if extract_root is None:
        extract_root = download_root
    else:
        extract_root = Path(extract_root)
    
    
    if not force_download and dataset_exists(download_info, download_root):
        if verbose:
            print(f"Dataset {download_info.name} already exists and is valid.")
        return
    
    if verbose:
        print(f"Downloading and extracting {download_info.name}...")
        if download_info.description:
            print(f"Description: {download_info.description}")
    
    # Try each mirror URL until one succeeds
    last_error = None
    for url in download_info.urls:
        try:
            download_and_extract_archive(
                url, download_root, extract_root, download_info.filename,
                download_info.md5, download_info.sha256, remove_finished
            )
            return  # Success
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Failed to download from {url}: {e}")
            continue
    
    # If we get here, all downloads failed
    raise RuntimeError(f"Failed to download {download_info.name} from all mirrors. "
                      f"Last error: {last_error}")



def download_dataset(
    download_info: DownloadInfo,
    root: Union[str, Path],
    force_download: bool = False,
    verbose: bool = True
) -> Path:
    """
    Download a dataset using the information from DatasetDownloads or DatasetDownloadsEnum enum.
    
    Args:
        download_info: DatasetDownloads object or DatasetDownloadsEnum enum value
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
    
    # Handle both DatasetDownloads objects and DatasetDownloadsEnum enum values
    if hasattr(download_info, 'value'):
        # It's a DatasetDownloadsEnum enum, get the DatasetDownloads
        info = download_info.value
    else:
        # It's already a DatasetDownloads object
        info = download_info
    
    file_path = root / info.filename
    
    # Check if already exists and valid
    if not force_download and dataset_exists(download_info, root):
        if verbose:
            print(f"Dataset {info.name} already exists and is valid: {file_path}")
        return file_path
    
    if verbose:
        print(f"Downloading {info.name}...")
        if info.description:
            print(f"Description: {info.description}")
    
    # Try each mirror URL until one succeeds
    last_error = None
    for url in info.urls:
        try:
            return download_url(
                url, root, info.filename,
                info.md5, info.sha256
            )
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Failed to download from {url}: {e}")
            continue
    
    # If we get here, all downloads failed
    raise RuntimeError(f"Failed to download {info.name} from all mirrors. "
                      f"Last error: {last_error}")