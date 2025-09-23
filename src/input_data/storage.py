"""
Storage strategies for dataset management.

This module provides different storage strategies for datasets:
- MemoryStorage: Load all data into memory (current approach)
- DiskStorage: Store data as files on disk and load on-demand
- HybridStorage: Intelligent switching based on dataset size
"""

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Any, Optional, Union
import numpy as np
from PIL import Image
import torch

class StorageStrategy(ABC):
    """Abstract base class for dataset storage strategies."""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.storage_root = dataset_root / "storage"
        self.storage_root.mkdir(exist_ok=True)
    
    @abstractmethod
    def save_data(self, data: np.ndarray, targets: np.ndarray, metadata: dict = None) -> None:
        """Save dataset to storage."""
        pass
    
    @abstractmethod
    def load_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Load a single sample from storage."""
        pass
    
    @abstractmethod
    def get_dataset_size(self) -> int:
        """Get the total number of samples."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the storage is ready for use."""
        pass
    
    @abstractmethod
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        pass


class MemoryStorage(StorageStrategy):
    """Store entire dataset in memory (current approach)."""
    
    def __init__(self, dataset_root: Path):
        super().__init__(dataset_root)
        self.data = None
        self.targets = None
        self.metadata = {}
    
    def save_data(self, data: np.ndarray, targets: np.ndarray, metadata: dict = None) -> None:
        """Save data to memory."""
        self.data = data
        self.targets = targets
        self.metadata = metadata or {}
        
        # Also save to disk as backup
        cache_file = self.storage_root / "memory_cache.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'data': data,
                'targets': targets,
                'metadata': metadata
            }, f)
    
    def load_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Load sample from memory."""
        if self.data is None:
            self._load_from_cache()
        return self.data[index], self.targets[index]
    
    def get_dataset_size(self) -> int:
        """Get dataset size."""
        if self.data is None:
            self._load_from_cache()
        return len(self.data) if self.data is not None else 0
    
    def is_ready(self) -> bool:
        """Check if memory storage is ready."""
        return self.data is not None or (self.storage_root / "memory_cache.pkl").exists()
    
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage."""
        if self.data is None:
            return 0.0
        return (self.data.nbytes + self.targets.nbytes) / (1024 * 1024)
    
    def _load_from_cache(self):
        """Load data from disk cache."""
        cache_file = self.storage_root / "memory_cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.data = cache_data['data']
                self.targets = cache_data['targets']
                self.metadata = cache_data.get('metadata', {})


class DiskStorage(StorageStrategy):
    """Store dataset as individual files on disk."""
    
    def __init__(self, dataset_root: Path, image_format: str = 'png'):
        super().__init__(dataset_root)
        self.image_format = image_format.lower()
        self.images_dir = self.storage_root / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_root / "metadata.pkl"
        self.index_file = self.storage_root / "index.pkl"
        
        # Load metadata if exists
        self.metadata = {}
        self.file_index = {}
        self._load_metadata()
    
    def save_data(self, data: np.ndarray, targets: np.ndarray, metadata: dict = None) -> None:
        """Save data as individual image files."""
        print(f"Saving {len(data)} samples to disk storage...")
        
        self.metadata = metadata or {}
        self.file_index = {}
        
        for i, (img_data, target) in enumerate(zip(data, targets)):
            # Create filename
            filename = f"{i:06d}_class{target:02d}.{self.image_format}"
            filepath = self.images_dir / filename
            
            # Save image
            self._save_image(img_data, filepath)
            
            # Update index
            self.file_index[i] = {
                'filepath': str(filepath),
                'target': int(target),
                'shape': img_data.shape
            }
            
            if (i + 1) % 10000 == 0:
                print(f"Saved {i + 1}/{len(data)} images...")
        
        # Save metadata and index
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.file_index, f)
        
        print(f"Disk storage complete: {len(data)} images saved.")
    
    def load_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Load sample from disk."""
        if index not in self.file_index:
            raise IndexError(f"Sample {index} not found in storage")
        
        sample_info = self.file_index[index]
        img_data = self._load_image(sample_info['filepath'], sample_info['shape'])
        return img_data, sample_info['target']
    
    def get_dataset_size(self) -> int:
        """Get dataset size."""
        return len(self.file_index)
    
    def is_ready(self) -> bool:
        """Check if disk storage is ready."""
        return self.index_file.exists() and len(self.file_index) > 0
    
    def get_memory_usage_mb(self) -> float:
        """Disk storage uses minimal memory."""
        return 1.0  # Just metadata
    
    def _save_image(self, img_data: np.ndarray, filepath: Path) -> None:
        """Save image data to file."""
        if len(img_data.shape) == 2:  # Grayscale
            # Normalize to 0-255 if needed
            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                img_data = (img_data * 255).astype(np.uint8)
            Image.fromarray(img_data, mode='L').save(filepath)
        
        elif len(img_data.shape) == 3:  # Color
            if img_data.shape[0] == 3:  # CHW format -> HWC
                img_data = np.transpose(img_data, (1, 2, 0))
            
            # Normalize to 0-255 if needed
            if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                img_data = (img_data * 255).astype(np.uint8)
            Image.fromarray(img_data, mode='RGB').save(filepath)
        
        else:
            raise ValueError(f"Unsupported image shape: {img_data.shape}")
    
    def _load_image(self, filepath: str, original_shape: tuple) -> np.ndarray:
        """Load image from file."""
        img = Image.open(filepath)
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1
        
        # Restore original format
        if len(original_shape) == 3 and original_shape[0] in [1, 3]:  # CHW format
            if len(img_array.shape) == 2:  # Grayscale
                img_array = img_array.reshape(1, *img_array.shape)
            elif len(img_array.shape) == 3:  # RGB -> CHW
                img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        
        if self.index_file.exists():
            with open(self.index_file, 'rb') as f:
                self.file_index = pickle.load(f)


class HybridStorage(StorageStrategy):
    """Intelligently choose between memory and disk storage based on dataset size."""
    
    def __init__(self, dataset_root: Path, memory_threshold_mb: float = 500.0):
        super().__init__(dataset_root)
        self.memory_threshold_mb = memory_threshold_mb
        self.strategy = None
    
    def save_data(self, data: np.ndarray, targets: np.ndarray, metadata: dict = None) -> None:
        """Choose storage strategy and save data."""
        # Estimate memory usage
        estimated_mb = (data.nbytes + targets.nbytes) / (1024 * 1024)
        
        if estimated_mb <= self.memory_threshold_mb:
            print(f"Using memory storage ({estimated_mb:.1f} MB <= {self.memory_threshold_mb} MB)")
            self.strategy = MemoryStorage(self.dataset_root)
        else:
            print(f"Using disk storage ({estimated_mb:.1f} MB > {self.memory_threshold_mb} MB)")
            self.strategy = DiskStorage(self.dataset_root)
        
        self.strategy.save_data(data, targets, metadata)
        
        # Save strategy choice
        strategy_file = self.storage_root / "strategy.txt"
        with open(strategy_file, 'w') as f:
            f.write(type(self.strategy).__name__)
    
    def load_sample(self, index: int) -> Tuple[np.ndarray, int]:
        """Load sample using the chosen strategy."""
        if self.strategy is None:
            self._load_strategy()
        return self.strategy.load_sample(index)
    
    def get_dataset_size(self) -> int:
        """Get dataset size."""
        if self.strategy is None:
            self._load_strategy()
        return self.strategy.get_dataset_size()
    
    def is_ready(self) -> bool:
        """Check if storage is ready."""
        strategy_file = self.storage_root / "strategy.txt"
        if not strategy_file.exists():
            return False
        
        if self.strategy is None:
            self._load_strategy()
        return self.strategy.is_ready()
    
    def get_memory_usage_mb(self) -> float:
        """Get memory usage."""
        if self.strategy is None:
            return 0.0
        return self.strategy.get_memory_usage_mb()
    
    def _load_strategy(self) -> None:
        """Load the previously chosen strategy."""
        strategy_file = self.storage_root / "strategy.txt"
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                strategy_name = f.read().strip()
            
            if strategy_name == "MemoryStorage":
                self.strategy = MemoryStorage(self.dataset_root)
            elif strategy_name == "DiskStorage":
                self.strategy = DiskStorage(self.dataset_root)
            else:
                raise ValueError(f"Unknown storage strategy: {strategy_name}")