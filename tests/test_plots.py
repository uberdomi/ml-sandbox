#!/usr/bin/env python3
"""
Test script for the new plotting utilities in plots.py
"""

import pytest
import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.input_data.structure.plots import (
    plot_sample_grayscale,
    plot_sample_rgb,
    plot_sample,
    plot_samples_grayscale,
    plot_samples_rgb,
    plot_samples,
    plot_tensor_grid,
    quick_plot,
)

@pytest.mark.plots
@pytest.mark.unit
class TestPlots:
    """Unit tests for plotting functions"""

    def test_single_grayscale(self, temp_data_dir: Path):
        """Test single grayscale image plotting"""
        print("Testing single grayscale image...")
        
        # Create a simple test pattern
        tensor = torch.zeros(28, 28)
        tensor[10:18, 10:18] = 1.0  # White square
        tensor[5:23, 5] = 0.5       # Gray vertical line
        tensor[5, 5:23] = 0.5       # Gray horizontal line
        
        fig, ax = plot_sample_grayscale(tensor, label="Test Pattern", figsize=(6, 6))
        plt.savefig(temp_data_dir / "test_single_grayscale.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_single_grayscale.png")


    def test_single_rgb(self, temp_data_dir: Path):
        """Test single RGB image plotting"""
        print("Testing single RGB image...")
        
        # Create a colorful test pattern
        tensor = torch.zeros(3, 32, 32)
        tensor[0, 10:22, 10:22] = 1.0  # Red square
        tensor[1, 5:27, 5:15] = 1.0    # Green rectangle
        tensor[2, 15:25, 20:30] = 1.0  # Blue rectangle
        
        fig, ax = plot_sample_rgb(tensor, label="RGB Test", figsize=(6, 6))
        plt.savefig(temp_data_dir / "test_single_rgb.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_single_rgb.png")


    def test_multiple_grayscale(self, temp_data_dir: Path):
        """Test multiple grayscale image plotting"""
        print("Testing multiple grayscale images...")
        
        # Create test patterns
        tensors = []
        labels = []
        for i in range(6):
            tensor = torch.zeros(28, 28)
            # Create different patterns
            if i == 0:
                tensor[10:18, 10:18] = 1.0  # Square
            elif i == 1:
                tensor[torch.arange(28), torch.arange(28)] = 1.0  # Diagonal
            elif i == 2:
                tensor[14, :] = 1.0  # Horizontal line
                tensor[:, 14] = 1.0  # Vertical line
            elif i == 3:
                # Circle-like pattern
                y, x = torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij')
                center = 14
                radius = 8
                mask = (x - center)**2 + (y - center)**2 <= radius**2
                tensor[mask] = 1.0
            elif i == 4:
                # Checkerboard pattern
                tensor[::2, ::2] = 1.0
                tensor[1::2, 1::2] = 1.0
            else:
                # Random noise
                tensor = torch.rand(28, 28)
            
            tensors.append(tensor)
            labels.append(f"Pattern {i+1}")
        
        fig, axes = plot_samples_grayscale(tensors, labels=labels, cols=3, figsize=(12, 8))
        plt.savefig(temp_data_dir / "test_multiple_grayscale.png")
        plt.close()

        assert os.path.exists(temp_data_dir / "test_multiple_grayscale.png")


    def test_multiple_rgb(self, temp_data_dir: Path):
        """Test multiple RGB image plotting"""
        print("Testing multiple RGB images...")
        
        # Create colorful test patterns
        tensors = []
        labels = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"]
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            tensor = torch.zeros(3, 32, 32)
            # Create different patterns with the specified color
            if i % 2 == 0:
                # Square pattern
                tensor[:, 10:22, 10:22] = torch.tensor(color).unsqueeze(-1).unsqueeze(-1)
            else:
                # Circle pattern
                y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
                center = 16
                radius = 10
                mask = (x - center)**2 + (y - center)**2 <= radius**2
                for c in range(3):
                    tensor[c][mask] = color[c]
            
            tensors.append(tensor)
        
        fig, axes = plot_samples_rgb(tensors, labels=labels, cols=3, figsize=(12, 8))
        plt.savefig(temp_data_dir / "test_multiple_rgb.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_multiple_rgb.png")


    def test_tensor_grid(self, temp_data_dir: Path):
        """Test batch tensor plotting"""
        print("Testing tensor grid plotting...")
        
        # Create a batch of mixed patterns
        batch_gray = torch.randn(8, 1, 28, 28)
        batch_rgb = torch.randn(6, 3, 32, 32)
        
        # Test grayscale batch
        fig, axes = plot_tensor_grid(batch_gray, 
                                    labels=[f"Gray {i}" for i in range(8)],
                                    cols=4, 
                                    suptitle="Grayscale Batch")
        plt.savefig(temp_data_dir / "test_tensor_grid_gray.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_tensor_grid_gray.png")
        
        # Test RGB batch
        fig, axes = plot_tensor_grid(batch_rgb, 
                                    labels=[f"RGB {i}" for i in range(6)],
                                    is_rgb=True,
                                    cols=3,
                                    suptitle="RGB Batch")
        plt.savefig(temp_data_dir / "test_tensor_grid_rgb.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_tensor_grid_rgb.png")


    def test_quick_plot(self, temp_data_dir: Path):
        """Test the quick_plot convenience function"""
        print("Testing quick_plot function...")
        
        # Test single tensor
        tensor = torch.randn(28, 28)
        fig, ax = quick_plot(tensor, labels="Single", title="Quick Plot Test")
        plt.savefig(temp_data_dir / "test_quick_plot_single.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_quick_plot_single.png")
        
        # Test list of tensors
        tensors = [torch.randn(1, 28, 28) for _ in range(4)]
        fig, axes = quick_plot(tensors, labels=[f"Item {i}" for i in range(4)], title="Quick Plot List")
        plt.savefig(temp_data_dir / "test_quick_plot_list.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_quick_plot_list.png")


    def test_plot_sample_auto_detect(self, temp_data_dir: Path):
        """Test plot_sample function with automatic RGB/grayscale detection"""
        print("Testing plot_sample auto-detection...")
        
        # Test grayscale detection
        grayscale_tensor = torch.randn(28, 28)
        fig, ax = plot_sample(grayscale_tensor, label="Auto Gray", title="Auto-detected Grayscale")
        plt.savefig(temp_data_dir / "test_plot_sample_gray.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_sample_gray.png")
        
        # Test RGB detection with (3, H, W) format
        rgb_tensor_chw = torch.randn(3, 32, 32)
        fig, ax = plot_sample(rgb_tensor_chw, label="Auto RGB", title="Auto-detected RGB (CHW)")
        plt.savefig(temp_data_dir / "test_plot_sample_rgb_chw.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_sample_rgb_chw.png")
        
        # Test RGB detection with (H, W, 3) format
        rgb_tensor_hwc = torch.randn(32, 32, 3)
        fig, ax = plot_sample(rgb_tensor_hwc, label="Auto RGB HWC", title="Auto-detected RGB (HWC)")
        plt.savefig(temp_data_dir / "test_plot_sample_rgb_hwc.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_sample_rgb_hwc.png")
        
        # Test grayscale with channel dim (1, H, W)
        grayscale_1hw = torch.randn(1, 28, 28)
        fig, ax = plot_sample(grayscale_1hw, label="Gray 1HW", title="Grayscale (1, H, W)")
        plt.savefig(temp_data_dir / "test_plot_sample_1hw.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_sample_1hw.png")


    def test_plot_samples_auto_detect(self, temp_data_dir: Path):
        """Test plot_samples function with automatic RGB/grayscale detection"""
        print("Testing plot_samples auto-detection...")
        
        # Test mixed grayscale tensors
        grayscale_tensors = [
            torch.randn(28, 28),      # 2D
            torch.randn(1, 28, 28),   # 3D with channel first
            torch.randn(28, 28, 1),   # 3D with channel last
        ]
        labels_gray = ["2D Gray", "1HW Gray", "HW1 Gray"]
        
        fig, axes = plot_samples(
            grayscale_tensors, 
            labels=labels_gray, 
            suptitle="Auto-detected Grayscale Samples",
            cols=3
        )
        plt.savefig(temp_data_dir / "test_plot_samples_gray_auto.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_samples_gray_auto.png")
        
        # Test RGB tensors
        rgb_tensors = [
            torch.randn(3, 32, 32),   # CHW format
            torch.randn(32, 32, 3),   # HWC format
        ]
        labels_rgb = ["RGB CHW", "RGB HWC"]
        
        fig, axes = plot_samples(
            rgb_tensors,
            labels=labels_rgb,
            suptitle="Auto-detected RGB Samples",
            cols=2
        )
        plt.savefig(temp_data_dir / "test_plot_samples_rgb_auto.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_samples_rgb_auto.png")


    def test_plot_sample_edge_cases(self, temp_data_dir: Path):
        """Test plot_sample with edge cases and error conditions"""
        print("Testing plot_sample edge cases...")
        
        # Test with numpy array input
        numpy_tensor = np.random.randn(28, 28)
        fig, ax = plot_sample(numpy_tensor, label="NumPy", title="NumPy Array Input")
        plt.savefig(temp_data_dir / "test_plot_sample_numpy.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_sample_numpy.png")
        
        # Test invalid tensor shape
        try:
            invalid_tensor = torch.randn(2, 3, 4, 5)  # 4D tensor
            plot_sample(invalid_tensor)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid tensor shape" in str(e)


    def test_plot_samples_edge_cases(self, temp_data_dir: Path):
        """Test plot_samples with edge cases and error conditions"""
        print("Testing plot_samples edge cases...")
        
        # Test with single tensor in list
        single_tensor = [torch.randn(28, 28)]
        fig, axes = plot_samples(single_tensor, labels=["Single"], suptitle="Single Tensor List")
        plt.savefig(temp_data_dir / "test_plot_samples_single.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_samples_single.png")
        
        # Test with mixed numpy/torch tensors
        mixed_tensors = [
            torch.randn(28, 28),
            np.random.randn(28, 28)
        ]
        fig, axes = plot_samples(mixed_tensors, labels=["Torch", "NumPy"], suptitle="Mixed Types")
        plt.savefig(temp_data_dir / "test_plot_samples_mixed.png")
        plt.close()
        
        assert os.path.exists(temp_data_dir / "test_plot_samples_mixed.png")
        
        # Test empty tensor list
        try:
            plot_samples([])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "cannot be empty" in str(e)


    def test_error_handling(self, temp_data_dir: Path):
        """Test error handling in plotting functions"""
        print("Testing error handling...")
        
        try:
            # Invalid tensor shape for grayscale
            invalid_tensor = torch.randn(2, 3, 4, 5)  # 4D tensor
            plot_sample_grayscale(invalid_tensor)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        try:
            # Invalid tensor shape for RGB
            invalid_tensor = torch.randn(28, 28)  # 2D tensor for RGB
            plot_sample_rgb(invalid_tensor)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        try:
            # Mismatched labels length
            tensors = [torch.randn(28, 28) for _ in range(3)]
            labels = ["A", "B"]  # Only 2 labels for 3 tensors
            plot_samples_grayscale(tensors, labels=labels)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        

# Run specific test if called directly
if __name__ == "__main__":
    # For direct execution, run pytest
    pytest.main([__file__, "-v"])