"""
Visualization utilities for tensor data.

This module provides helper functions for visualizing image tensors from dataset classes.
Functions support both single sample and multi-sample plotting for grayscale and RGB images.
"""

import math
from typing import Union, List, Optional, Tuple, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_sample_grayscale(
    tensor: torch.Tensor, 
    label: Optional[Union[int, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
    cmap: str = 'gray',
    show_axes: bool = False,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a single grayscale image tensor.
    
    Args:
        tensor: Input tensor of shape (H, W) or (1, H, W)
        label: Optional label to display in title
        title: Custom title for the plot
        figsize: Figure size as (width, height)
        cmap: Colormap for grayscale visualization
        show_axes: Whether to show axis ticks and labels
        save_path: Optional path to save the figure
        
    Returns:
        Tuple of (figure, axes) objects
        
    Raises:
        ValueError: If tensor dimensions are invalid for grayscale visualization
        
    Example:
        >>> tensor = torch.randn(28, 28)
        >>> fig, ax = plot_sample_grayscale(tensor, label=5)
        >>> plt.show()
    """
    # Handle tensor conversion and validation
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = np.array(tensor)
    
    # Handle different tensor shapes
    if tensor_np.ndim == 3:
        if tensor_np.shape[0] == 1:
            # Shape (1, H, W) -> (H, W)
            tensor_np = tensor_np.squeeze(0)
        elif tensor_np.shape[2] == 1:
            # Shape (H, W, 1) -> (H, W)
            tensor_np = tensor_np.squeeze(2)
        else:
            raise ValueError(f"Invalid tensor shape for grayscale: {tensor_np.shape}. "
                           "Expected (H, W), (1, H, W), or (H, W, 1)")
    elif tensor_np.ndim != 2:
        raise ValueError(f"Invalid tensor shape for grayscale: {tensor_np.shape}. "
                       "Expected 2D or 3D tensor")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Display the image
    im = ax.imshow(tensor_np, cmap=cmap, interpolation='nearest')
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')
    elif label is not None:
        ax.set_title(f'Label: {label}', fontsize=12, fontweight='bold')
    
    # Handle axes display
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar for reference
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig, ax


def plot_sample_rgb(
    tensor: torch.Tensor,
    label: Optional[Union[int, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
    show_axes: bool = False,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> Tuple[Figure, Axes]:
    """
    Plot a single RGB image tensor.
    
    Args:
        tensor: Input tensor of shape (3, H, W) or (H, W, 3)
        label: Optional label to display in title
        title: Custom title for the plot
        figsize: Figure size as (width, height)
        show_axes: Whether to show axis ticks and labels
        save_path: Optional path to save the figure
        normalize: Whether to normalize tensor values to [0, 1] range
        
    Returns:
        Tuple of (figure, axes) objects
        
    Raises:
        ValueError: If tensor dimensions are invalid for RGB visualization
        
    Example:
        >>> tensor = torch.randn(3, 32, 32)
        >>> fig, ax = plot_sample_rgb(tensor, label="airplane")
        >>> plt.show()
    """
    # Handle tensor conversion and validation
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = np.array(tensor)
    
    # Handle different tensor shapes
    if tensor_np.ndim != 3:
        raise ValueError(f"Invalid tensor shape for RGB: {tensor_np.shape}. "
                       "Expected 3D tensor")
    
    if tensor_np.shape[0] == 3:
        # Shape (3, H, W) -> (H, W, 3) for matplotlib
        tensor_np = tensor_np.transpose(1, 2, 0)
    elif tensor_np.shape[2] != 3:
        raise ValueError(f"Invalid tensor shape for RGB: {tensor_np.shape}. "
                       "Expected (3, H, W) or (H, W, 3)")
    
    # Normalize if requested
    if normalize:
        # Normalize to [0, 1] range
        tensor_min, tensor_max = tensor_np.min(), tensor_np.max()
        if tensor_max > tensor_min:
            tensor_np = (tensor_np - tensor_min) / (tensor_max - tensor_min)
        # Clip to ensure valid range
        tensor_np = np.clip(tensor_np, 0, 1)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Display the image
    ax.imshow(tensor_np, interpolation='nearest')
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold')
    elif label is not None:
        ax.set_title(f'Label: {label}', fontsize=12, fontweight='bold')
    
    # Handle axes display
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig, ax


def plot_samples_grayscale(
    tensors: List[torch.Tensor],
    labels: Optional[List[Union[int, str]]] = None,
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cols: Optional[int] = None,
    cmap: str = 'gray',
    show_axes: bool = False,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None
) -> Tuple[Figure, List[Axes]]:
    """
    Plot multiple grayscale image tensors in subplots.
    
    Args:
        tensors: List of tensors, each of shape (H, W) or (1, H, W)
        labels: Optional list of labels for each tensor
        titles: Optional list of custom titles for each subplot
        figsize: Figure size as (width, height). Auto-calculated if None
        cols: Number of columns in subplot grid. Auto-calculated if None
        cmap: Colormap for grayscale visualization
        show_axes: Whether to show axis ticks and labels
        save_path: Optional path to save the figure
        suptitle: Main title for the entire figure
        
    Returns:
        Tuple of (figure, list of axes) objects
        
    Raises:
        ValueError: If input lists have mismatched lengths
        
    Example:
        >>> tensors = [torch.randn(28, 28) for _ in range(6)]
        >>> labels = list(range(6))
        >>> fig, axes = plot_samples_grayscale(tensors, labels=labels, cols=3)
        >>> plt.show()
    """
    if not tensors:
        raise ValueError("tensors list cannot be empty")
    
    n_samples = len(tensors)
    
    # Validate input lengths
    if labels is not None and len(labels) != n_samples:
        raise ValueError(f"Length mismatch: {len(labels)} labels for {n_samples} tensors")
    if titles is not None and len(titles) != n_samples:
        raise ValueError(f"Length mismatch: {len(titles)} titles for {n_samples} tensors")
    
    # Calculate grid dimensions
    if cols is None:
        cols = min(4, n_samples)  # Default to max 4 columns
    rows = math.ceil(n_samples / cols)
    
    # Calculate figure size
    if figsize is None:
        figsize = (cols * 3, rows * 3)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each tensor
    for i, tensor in enumerate(tensors):
        ax = axes[i]
        
        # Convert tensor to numpy
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = np.array(tensor)
        
        # Handle tensor shape
        if tensor_np.ndim == 3 and tensor_np.shape[0] == 1:
            tensor_np = tensor_np.squeeze(0)
        elif tensor_np.ndim == 3 and tensor_np.shape[2] == 1:
            tensor_np = tensor_np.squeeze(2)
        elif tensor_np.ndim != 2:
            raise ValueError(f"Invalid tensor shape for grayscale: {tensor_np.shape}")
        
        # Display image
        im = ax.imshow(tensor_np, cmap=cmap, interpolation='nearest')
        
        # Set title
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        elif labels is not None and i < len(labels):
            ax.set_title(f'Label: {labels[i]}', fontsize=10)
        else:
            ax.set_title(f'Sample {i+1}', fontsize=10)
        
        # Handle axes display
        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    # Set main title
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig, axes[:n_samples]


def plot_samples_rgb(
    tensors: List[torch.Tensor],
    labels: Optional[List[Union[int, str]]] = None,
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cols: Optional[int] = None,
    show_axes: bool = False,
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None,
    normalize: bool = True
) -> Tuple[Figure, List[Axes]]:
    """
    Plot multiple RGB image tensors in subplots.
    
    Args:
        tensors: List of tensors, each of shape (3, H, W) or (H, W, 3)
        labels: Optional list of labels for each tensor
        titles: Optional list of custom titles for each subplot
        figsize: Figure size as (width, height). Auto-calculated if None
        cols: Number of columns in subplot grid. Auto-calculated if None
        show_axes: Whether to show axis ticks and labels
        save_path: Optional path to save the figure
        suptitle: Main title for the entire figure
        normalize: Whether to normalize tensor values to [0, 1] range
        
    Returns:
        Tuple of (figure, list of axes) objects
        
    Raises:
        ValueError: If input lists have mismatched lengths
        
    Example:
        >>> tensors = [torch.randn(3, 32, 32) for _ in range(8)]
        >>> labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse"]
        >>> fig, axes = plot_samples_rgb(tensors, labels=labels, cols=4)
        >>> plt.show()
    """
    if not tensors:
        raise ValueError("tensors list cannot be empty")
    
    n_samples = len(tensors)
    
    # Validate input lengths
    if labels is not None and len(labels) != n_samples:
        raise ValueError(f"Length mismatch: {len(labels)} labels for {n_samples} tensors")
    if titles is not None and len(titles) != n_samples:
        raise ValueError(f"Length mismatch: {len(titles)} titles for {n_samples} tensors")
    
    # Calculate grid dimensions
    if cols is None:
        cols = min(4, n_samples)  # Default to max 4 columns
    rows = math.ceil(n_samples / cols)
    
    # Calculate figure size
    if figsize is None:
        figsize = (cols * 3, rows * 3)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each tensor
    for i, tensor in enumerate(tensors):
        ax = axes[i]
        
        # Convert tensor to numpy
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = np.array(tensor)
        
        # Handle tensor shape
        if tensor_np.ndim != 3:
            raise ValueError(f"Invalid tensor shape for RGB: {tensor_np.shape}")
        
        if tensor_np.shape[0] == 3:
            # Shape (3, H, W) -> (H, W, 3)
            tensor_np = tensor_np.transpose(1, 2, 0)
        elif tensor_np.shape[2] != 3:
            raise ValueError(f"Invalid tensor shape for RGB: {tensor_np.shape}")
        
        # Normalize if requested
        if normalize:
            tensor_min, tensor_max = tensor_np.min(), tensor_np.max()
            if tensor_max > tensor_min:
                tensor_np = (tensor_np - tensor_min) / (tensor_max - tensor_min)
            tensor_np = np.clip(tensor_np, 0, 1)
        
        # Display image
        ax.imshow(tensor_np, interpolation='nearest')
        
        # Set title
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        elif labels is not None and i < len(labels):
            ax.set_title(f'Label: {labels[i]}', fontsize=10)
        else:
            ax.set_title(f'Sample {i+1}', fontsize=10)
        
        # Handle axes display
        if not show_axes:
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    # Set main title
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig, axes[:n_samples]


def plot_tensor_grid(
    tensor: torch.Tensor,
    labels: Optional[List[Union[int, str]]] = None,
    is_rgb: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    cols: Optional[int] = None,
    titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    cmap: str = 'gray',
    show_axes: bool = False,
    normalize: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Figure, List[Axes]]:
    """
    Convenience function to plot a batch of tensors.
    
    This function automatically determines whether to use grayscale or RGB plotting
    based on tensor shape or the is_rgb parameter.
    
    Args:
        tensor: Batch tensor of shape (N, C, H, W) or (N, H, W, C)
        labels: Optional list of labels for each sample
        is_rgb: Whether to treat as RGB images (auto-detected if None)
        figsize: Figure size as (width, height)
        cols: Number of columns in subplot grid
        titles: Optional list of custom titles
        suptitle: Main title for the entire figure
        cmap: Colormap for grayscale visualization
        show_axes: Whether to show axis ticks and labels
        normalize: Whether to normalize RGB values to [0, 1] range
        save_path: Optional path to save the figure
        
    Returns:
        Tuple of (figure, list of axes) objects
        
    Example:
        >>> # Batch of grayscale images
        >>> batch = torch.randn(8, 1, 28, 28)
        >>> fig, axes = plot_tensor_grid(batch, cols=4)
        >>> plt.show()
        
        >>> # Batch of RGB images
        >>> batch = torch.randn(6, 3, 32, 32)
        >>> fig, axes = plot_tensor_grid(batch, is_rgb=True, cols=3)
        >>> plt.show()
    """
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.detach().cpu().numpy()
    else:
        tensor_np = np.array(tensor)
    
    if tensor_np.ndim != 4:
        raise ValueError(f"Expected 4D tensor (batch), got shape: {tensor_np.shape}")
    
    # Extract individual tensors
    tensor_list = [tensor_np[i] for i in range(tensor_np.shape[0])]
    
    # Auto-detect RGB vs grayscale if not specified
    if not is_rgb:
        # Check if channels dimension suggests RGB or grayscale
        if tensor_np.shape[1] == 3:  # (N, 3, H, W)
            is_rgb = True
        elif tensor_np.shape[3] == 3:  # (N, H, W, 3)
            is_rgb = True
        elif tensor_np.shape[1] == 1 or tensor_np.shape[3] == 1:  # (N, 1, H, W) or (N, H, W, 1)
            is_rgb = False
        else:
            # Default to grayscale for ambiguous cases
            is_rgb = False
    
    # Convert to torch tensors for compatibility with plot functions
    tensor_list = [torch.from_numpy(t) for t in tensor_list]
    
    # Use appropriate plotting function
    if is_rgb:
        return plot_samples_rgb(
            tensors=tensor_list,
            labels=labels,
            titles=titles,
            figsize=figsize,
            cols=cols,
            show_axes=show_axes,
            save_path=save_path,
            suptitle=suptitle,
            normalize=normalize
        )
    else:
        return plot_samples_grayscale(
            tensors=tensor_list,
            labels=labels,
            titles=titles,
            figsize=figsize,
            cols=cols,
            cmap=cmap,
            show_axes=show_axes,
            save_path=save_path,
            suptitle=suptitle
        )


# Utility functions for common use cases
def quick_plot(
    data: Union[torch.Tensor, List[torch.Tensor], np.ndarray],
    labels: Optional[Union[List[Union[int, str]], Union[int, str]]] = None,
    title: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Union[Axes, List[Axes]]]:
    """
    Quick plotting function that automatically determines the best visualization approach.
    
    Args:
        data: Single tensor, list of tensors, or numpy array
        labels: Labels for the data (single label or list)
        title: Title for the plot
        **kwargs: Additional arguments passed to specific plot functions
        
    Returns:
        Tuple of (figure, axes) - axes can be single Axes or list depending on input
        
    Example:
        >>> # Single grayscale image
        >>> tensor = torch.randn(28, 28)
        >>> fig, ax = quick_plot(tensor, labels=5)
        
        >>> # Multiple RGB images
        >>> tensors = [torch.randn(3, 32, 32) for _ in range(4)]
        >>> fig, axes = quick_plot(tensors, labels=["cat", "dog", "bird", "car"])
    """
    # Handle single tensor
    if isinstance(data, (torch.Tensor, np.ndarray)):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        # Check if it's a batch or single sample
        if data.ndim == 4:
            # Batch tensor
            return plot_tensor_grid(data, labels=labels, suptitle=title, **kwargs)
        elif data.ndim in [2, 3]:
            # Single sample
            # Determine if RGB or grayscale
            if data.ndim == 3 and (data.shape[0] == 3 or data.shape[2] == 3):
                return plot_sample_rgb(data, label=labels, title=title, **kwargs)
            else:
                return plot_sample_grayscale(data, label=labels, title=title, **kwargs)
        else:
            raise ValueError(f"Unsupported tensor shape: {data.shape}")
    
    # Handle list of tensors
    elif isinstance(data, list):
        if not data:
            raise ValueError("Empty list provided")
        
        # Check first tensor to determine type
        first_tensor = data[0]
        if isinstance(first_tensor, np.ndarray):
            first_tensor = torch.from_numpy(first_tensor)
        
        # Determine if RGB or grayscale
        if first_tensor.ndim == 3 and (first_tensor.shape[0] == 3 or first_tensor.shape[2] == 3):
            return plot_samples_rgb(data, labels=labels, suptitle=title, **kwargs)
        else:
            return plot_samples_grayscale(data, labels=labels, suptitle=title, **kwargs)
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def plot_class_samples(
    dataset: Any,
    class_names: List[str],
    samples_per_class: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, List[Axes]]:
    """
    Plot representative samples from each class in a dataset.
    
    This function is designed to work with dataset objects that have __getitem__ method.
    
    Args:
        dataset: Dataset object with __getitem__ method returning (tensor, label)
        class_names: List of class names
        samples_per_class: Number of samples to show per class
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Tuple of (figure, list of axes) objects
        
    Example:
        >>> fig, axes = plot_class_samples(
        ...     dataset=mnist_dataset,
        ...     class_names=[str(i) for i in range(10)],
        ...     samples_per_class=3
        ... )
    """
    n_classes = len(class_names)
    total_samples = n_classes * samples_per_class
    
    # Calculate grid dimensions
    cols = samples_per_class
    rows = n_classes
    
    if figsize is None:
        figsize = (cols * 3, rows * 2)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Collect samples for each class
    class_samples = {i: [] for i in range(n_classes)}
    
    # Sample from dataset
    for i in range(len(dataset)):
        tensor, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        if label < n_classes and len(class_samples[label]) < samples_per_class:
            class_samples[label].append(tensor)
        
        # Check if we have enough samples for all classes
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    # Plot samples
    for class_idx in range(n_classes):
        for sample_idx in range(samples_per_class):
            ax = axes[class_idx, sample_idx]
            
            if sample_idx < len(class_samples[class_idx]):
                tensor = class_samples[class_idx][sample_idx]
                
                # Convert to numpy
                if isinstance(tensor, torch.Tensor):
                    tensor_np = tensor.detach().cpu().numpy()
                else:
                    tensor_np = np.array(tensor)
                
                # Determine if RGB or grayscale and plot
                if tensor_np.ndim == 3:
                    if tensor_np.shape[0] == 3:
                        # RGB (3, H, W) -> (H, W, 3)
                        tensor_np = tensor_np.transpose(1, 2, 0)
                        # Normalize for display
                        tensor_min, tensor_max = tensor_np.min(), tensor_np.max()
                        if tensor_max > tensor_min:
                            tensor_np = (tensor_np - tensor_min) / (tensor_max - tensor_min)
                        tensor_np = np.clip(tensor_np, 0, 1)
                        ax.imshow(tensor_np)
                    elif tensor_np.shape[0] == 1:
                        # Grayscale (1, H, W) -> (H, W)
                        tensor_np = tensor_np.squeeze(0)
                        ax.imshow(tensor_np, cmap='gray')
                    else:
                        # Assume (H, W, C) format
                        if tensor_np.shape[2] == 1:
                            tensor_np = tensor_np.squeeze(2)
                            ax.imshow(tensor_np, cmap='gray')
                        else:
                            ax.imshow(tensor_np)
                else:
                    # 2D grayscale
                    ax.imshow(tensor_np, cmap='gray')
            else:
                # No sample available
                ax.text(0.5, 0.5, 'No sample', ha='center', va='center', transform=ax.transAxes)
            
            # Set title for first column
            if sample_idx == 0:
                ax.set_ylabel(f'{class_names[class_idx]}', fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle(f'Representative Samples ({samples_per_class} per class)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig, axes.flatten().tolist()
