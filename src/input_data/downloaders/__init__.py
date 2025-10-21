"""
Dataset utilities for downloading and extracting common ML datasets.
"""

from .datasets import DownloadInfo, download_dataset
from .fetch import download_url, check_integrity
from .extract import extract_archive