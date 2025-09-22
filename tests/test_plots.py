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

from src.input_data.plots import (
    plot_sample_grayscale,
    plot_sample_rgb,
    plot_samples_grayscale,
    plot_samples_rgb,
    plot_tensor_grid,
    quick_plot,
    plot_class_samples
)

@pytest.mark.plots
def test_single_grayscale(temp_data_dir: Path):
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

@pytest.mark.plots
def test_single_rgb(temp_data_dir: Path):
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

@pytest.mark.plots
def test_multiple_grayscale(temp_data_dir: Path):
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

@pytest.mark.plots
def test_multiple_rgb(temp_data_dir: Path):
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

@pytest.mark.plots
def test_tensor_grid(temp_data_dir: Path):
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

@pytest.mark.plots
def test_quick_plot(temp_data_dir: Path):
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

@pytest.mark.plots
def test_error_handling(temp_data_dir: Path):
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