
"""
Dataset utilities for downloading common ML datasets.
"""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from .fetch import download_url

@dataclass
class DownloadInfo:
    """Information for downloading dataset files - URLs, checksums, and metadata.
    
    Args:
        name: Descriptive name of the dataset/file
        filename: Name to save the downloaded file as
        urls: List of mirror URLs to download from
        md5: Expected MD5 checksum (optional)
        sha256: Expected SHA256 checksum (optional)
        file_size: Expected file size in bytes (optional)
        description: Additional description or notes (optional)
    """
    # File information
    name: str
    filename: str
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

def download_dataset(
    download_info: DownloadInfo,
    root: Union[str, Path],
    force_download: bool = False,
    verbose: bool = True
) -> None:
    """
    Download and extract a dataset based on the information provided in DownloadInfo.
    
    Args:
        download_info: DownloadInfo object containing necessary information
        root: Root directory to download and extract the dataset
        force_download: Whether to re-download even if files are already present
        verbose: Whether to print progress information
    """
    root = Path(root)
    info = download_info  # For brevity
    
    if verbose:
        print(f"Downloading and extracting {info.name}...")
        if info.description:
            print(f"Description: {info.description}")
    
    # Try each mirror URL until one succeeds
    last_error = None
    for url in info.urls:
        try:
    
            # Download the archive
            archive_path = download_url(
                url,
                root,
                filename=info.filename,
                force_download=force_download,
                md5=info.md5,
                sha256=info.sha256
            )
            
            if verbose:
                print(f"Downloaded {info.filename} from {url} to {archive_path}")
            
            return  # Success
        except Exception as e:
            last_error = e
            if verbose:
                print(f"Failed to download from {url}: {e}")
            continue
    
    # If we get here, all downloads failed
    raise RuntimeError(f"Failed to download {info.name} from all mirrors. "
                      f"Last error: {last_error}")