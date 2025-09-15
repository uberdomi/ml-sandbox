"""
Pytest configuration and shared fixtures for dataset tests.

This file provides common fixtures and configuration for all tests in the test suite.
It automatically sets up logging, provides PyTorch utilities, and creates reusable
test fixtures for datasets, transforms, and temporary directories.
"""

import pytest
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any

# Third-party imports
import torch
import torchvision.transforms as transforms

# Add the src directory to Python path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# ============================================================================
# Session-scoped fixtures (created once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session") 
def data_root(project_root):
    """Get the data directory for datasets."""
    return project_root / "data"

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data that persists for the entire test session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_datasets_"))
    yield temp_dir
    # Cleanup after all tests complete
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def device():
    """Provide the best available device for testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")
    return device

# ============================================================================
# Function-scoped fixtures (created for each test function)
# ============================================================================

@pytest.fixture
def sample_transforms():
    """Provide common transforms for grayscale datasets (MNIST, Fashion-MNIST)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

@pytest.fixture
def cifar_transforms():
    """Provide CIFAR-specific transforms for RGB datasets."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

@pytest.fixture
def augmentation_transforms():
    """Provide data augmentation transforms for testing."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

@pytest.fixture
def test_batch_sizes():
    """Provide common batch sizes for testing."""
    return [1, 4, 16, 32]

@pytest.fixture
def small_sample_size():
    """Provide a small sample size for quick testing."""
    return 100

@pytest.fixture
def dataset_config():
    """Provide common dataset configuration for tests."""
    return {
        'train': True,
        'download': False,  # Default to False to avoid unnecessary downloads
        'transform': None,
        'target_transform': None,
    }

@pytest.fixture
def tensor_shapes():
    """Provide expected tensor shapes for different datasets."""
    return {
        'mnist': (1, 28, 28),
        'fashion_mnist': (1, 28, 28),
        'cifar10': (3, 32, 32),
    }

# ============================================================================
# Utility fixtures
# ============================================================================

@pytest.fixture
def mock_download_config():
    """Provide configuration for mocking downloads during testing."""
    return {
        'chunk_size': 1024,
        'timeout': 30,
        'max_retries': 3,
    }

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    yield tmp_path
    # Cleanup
    if tmp_path.exists():
        tmp_path.unlink()

# ============================================================================
# Auto-use fixtures (run automatically for every test)
# ============================================================================

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for each test with consistent format."""
    logger = logging.getLogger(__name__)
    logger.info("Starting test")
    yield
    logger.info("Test completed")

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    # Set deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Pytest hooks and configuration
# ============================================================================
# Note: Markers are defined in pyproject.toml to avoid duplication

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically based on test names."""
    for item in items:
        # Auto-mark tests based on naming patterns
        if "mnist" in item.name.lower() and "fashion" not in item.name.lower():
            item.add_marker(pytest.mark.mnist)
        if "fashion" in item.name.lower():
            item.add_marker(pytest.mark.fashion_mnist)
        if "cifar" in item.name.lower():
            item.add_marker(pytest.mark.cifar10)
        if "transform" in item.name.lower():
            item.add_marker(pytest.mark.transform)
        if "download" in item.name.lower():
            item.add_marker(pytest.mark.download)
            item.add_marker(pytest.mark.slow)  # Downloads are typically slow
        if "error" in item.name.lower() or "invalid" in item.name.lower():
            item.add_marker(pytest.mark.error_handling)
        if "api" in item.name.lower() or "factory" in item.name.lower():
            item.add_marker(pytest.mark.dataset_api)

def pytest_runtest_setup(item):
    """Setup run before each test item."""
    # Skip download tests if --no-download option is used
    if "download" in [mark.name for mark in item.iter_markers()]:
        if item.config.getoption("--no-download", default=False):
            pytest.skip("Skipping download test due to --no-download option")

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-download",
        action="store_true",
        default=False,
        help="Skip tests that require downloading data"
    )
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="Run slow tests (by default they are skipped)"
    )