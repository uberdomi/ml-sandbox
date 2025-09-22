
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
    """Information for downloading dataset files - URLs, checksums, and metadata.
    
    Args:
        name: Descriptive name of the dataset/file
        filename: Name to save the downloaded file as
        extract_folder: Subfolder to extract into within root (default: "data")
        urls: List of mirror URLs to download from
        md5: Expected MD5 checksum (optional)
        sha256: Expected SHA256 checksum (optional)
        file_size: Expected file size in bytes (optional)
        description: Additional description or notes (optional)
    """
    # File information
    name: str
    filename: str
    extract_folder: str
    # Mirror URLs
    urls: list[str]
    # Integrity checks
    md5: Optional[str] = None
    sha256: Optional[str] = None
    file_size: Optional[int] = None  # Size in bytes
    # Additional notes for verbosity
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
    root: Union[str, Path],
    filename: Optional[str] = None,
    extract_folder: str = "data",
    md5: Optional[str] = None,
    sha256: Optional[str] = None,
    remove_finished: bool = False
) -> None:
    """
    Download and extract an archive in one step.
    
    Args:
        url: URL to download from
        root: Directory to download and extract the archive to
        filename: Name to save the archive as
        extract_folder: Subfolder to extract into within root
        md5: Expected MD5 checksum
        sha256: Expected SHA256 checksum  
        remove_finished: Whether to delete the archive after extraction
    """
    root = Path(root)
    
    # Download the archive
    archive_path = download_url(url, root, filename, md5, sha256)
    
    extract_root = root / extract_folder
    extract_root.mkdir(parents=True, exist_ok=True)
    
    # Extract the archive
    extract_archive(archive_path, extract_root, remove_finished)


def download_and_extract_dataset(
    download_info: DownloadInfo,
    root: Union[str, Path],
    remove_finished: bool = False,
    verbose: bool = True
) -> None:
    """
    Download and extract a dataset based on the information provided in DownloadInfo.
    
    Args:
        download_info: DownloadInfo object containing necessary information
        root: Root directory to download and extract the dataset
        remove_finished: Whether to delete the archive after extraction
        verbose: Whether to print progress information
    """
    root = Path(root)
    
    if verbose:
        print(f"Downloading and extracting {download_info.name}...")
        if download_info.description:
            print(f"Description: {download_info.description}")
    
    # Try each mirror URL until one succeeds
    last_error = None
    for url in download_info.urls:
        try:
            download_and_extract_archive(
                url, root, download_info.filename, extract_folder=download_info.extract_folder,
                md5=download_info.md5, sha256=download_info.sha256, remove_finished=remove_finished
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