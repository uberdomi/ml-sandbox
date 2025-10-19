"""
Test script to verify all key dependencies are properly installed.
Run with: uv run python test.py
"""

def test_imports():
    """Test that all key packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    # Core numerical and data processing
    print("\n[1/10] Testing numpy...")
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
    
    print("[2/10] Testing pandas...")
    import pandas as pd
    print(f"  ✓ pandas {pd.__version__}")
    
    # Image processing
    print("[3/10] Testing pillow...")
    import PIL
    print(f"  ✓ Pillow {PIL.__version__}")
    
    # Data visualization
    print("[4/10] Testing matplotlib...")
    import matplotlib
    print(f"  ✓ matplotlib {matplotlib.__version__}")
    
    print("[5/10] Testing seaborn...")
    import seaborn as sns
    print(f"  ✓ seaborn {sns.__version__}")
    
    # Progress bars
    print("[6/10] Testing tqdm...")
    import tqdm
    print(f"  ✓ tqdm {tqdm.__version__}")
    
    # Jupyter support
    print("[7/10] Testing ipykernel...")
    import ipykernel
    print(f"  ✓ ipykernel {ipykernel.__version__}")
    
    # Testing
    print("[8/10] Testing pytest...")
    import pytest
    print(f"  ✓ pytest {pytest.__version__}")
    
    # PyTorch
    print("[9/10] Testing torch...")
    import torch
    print(f"  ✓ torch {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"  ✓ CUDA available: {device_name}")
    else:
        print(f"  ℹ CUDA not available (CPU-only mode)")
    
    print("[10/10] Testing torchvision...")
    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")
    
    # Test project imports
    print("\n[Bonus] Testing project modules...")
    from src.input_data import create_dataset, SupportedDatasets
    print(f"  ✓ src.input_data imports successfully")
    
    print("\n" + "=" * 60)
    print("✓ All imports successful!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        test_imports()
    except Exception as e:
        print(f"\n✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)