"""
Pytest configuration and shared fixtures for dataset tests.
"""

import pytest
import sys
import logging
from pathlib import Path

# Add the src directory to Python path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session") 
def data_root(project_root):
    """Get the data directory."""
    return project_root / "data"

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for each test."""
    logger = logging.getLogger(__name__)
    logger.info("Starting test")
    yield
    logger.info("Test completed")