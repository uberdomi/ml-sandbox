# 🧪 ML Sandbox

A modern machine learning playground project for experimentation and learning with PyTorch, featuring clean dependency management with `uv`.

## ✨ Features

- **Modern Dependency Management**: Uses `uv` for fast, reliable Python package management
- **Flexible PyTorch Installation**: Choose between CPU-only or CUDA-enabled PyTorch
- **Dataset Utilities**: Built-in support for MNIST, Fashion-MNIST, and CIFAR-10 datasets
- **Jupyter Notebook Support**: Integrated Jupyter Lab/Notebook environment
- **Testing Suite**: Comprehensive tests with pytest
- **Type Hints & Linting**: Modern Python practices with black and ruff

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- `pip` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/paulhoheisel/ml-sandbox.git
   cd ml-sandbox
   ```

2. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

3. **Choose your installation option:**

   **Option A: With CUDA 12.6 support (for NVIDIA GPUs):**
   ```bash
   uv sync --extra cu126
   ```

   **Option B: CPU-only (lightweight, no GPU):**
   ```bash
   uv sync --extra cpu
   ```

   **Option C: Full development environment with Jupyter:**
   ```bash
   uv sync --extra cu126 --extra jupyter --extra dev
   ```

4. **Verify installation:**
   ```bash
   uv run python test.py
   ```

   You should see all imports successful and CUDA status (if applicable).

### Alternative: Traditional pip Installation

For backwards compatibility, you can use traditional `requirements.txt` (which were compiled using `uv` from the `pyproject.toml` file):

```bash
# With CUDA support
pip install -r requirements.txt

# CPU-only
pip install -r requirements-cpu.txt
```

## 📦 Available Installation Extras

- **`cpu`**: PyTorch CPU version (lightweight, no CUDA dependencies)
- **`cu126`**: PyTorch with CUDA 12.6 support (for NVIDIA GPUs)
- **`jupyter`**: Full Jupyter Lab environment (`jupyterlab`, `notebook`, `ipywidgets`)
- **`dev`**: Development tools (`black`, `ruff`, additional pytest plugins)

**Note**: You cannot install both `cpu` and `cu126` extras simultaneously - choose one based on your hardware.

## 🏃 Running the Project

### Using Scripts

With `uv run`, you don't need to activate the virtual environment:

```bash
# Run a Python script
uv run python your_script.py

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_datasets.py

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing
```

### Activating the Virtual Environment

Alternatively, activate the virtual environment:

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Then run commands normally
python your_script.py
pytest
```

### Jupyter Notebooks

To work with Jupyter notebooks:

```bash
# Make sure Jupyter is installed
uv sync --extra jupyter --extra cu126

# Launch Jupyter Lab
uv run jupyter lab

# Or launch classic Notebook
uv run jupyter notebook
```

The Jupyter kernel will automatically use the project's virtual environment.

## 📂 Project Structure

```
ml-sandbox/
├── .venv/                  # Virtual environment (created by uv sync)
├── notebooks/              # Jupyter notebooks
├── data/                   # Downloaded datasets
├── src/                    # Source code
│   └── input_data/         # Dataset utilities
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Locked dependency versions (auto-generated)
├── requirements.txt        # Generated for pip compatibility (CUDA)
├── requirements-cpu.txt    # Generated for pip compatibility (CPU-only)
├── README.md               # This file
└── test.py                 # Quick import verification script
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test markers
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m "not download"  # Skip tests requiring downloads

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# View the coverage report in your browser
xdg-open htmlcov/index.html  # Linux
# or open htmlcov/index.html  # macOS
# or start htmlcov/index.html  # Windows
# or manually open: file:///path/to/project/htmlcov/index.html
```

The HTML coverage report shows:
- Overall coverage percentage for your project
- Line-by-line coverage (which lines were executed during tests)
- Missing coverage (lines not covered by tests, highlighted in red)
- Coverage by module/file - click on files to see detailed coverage

### Code Formatting and Linting

```bash
# Install dev dependencies
uv sync --extra dev

# Format code with black
uv run black src/ tests/

# Lint with ruff
uv run ruff check src/ tests/

# Fix auto-fixable issues
uv run ruff check --fix src/ tests/
```

### Adding New Dependencies

```bash
# Add a new package
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade

# Generate updated requirements.txt
uv export --format requirements.txt --extra cu126 --no-dev -o requirements.txt
```

## 🔄 Generating requirements.txt

For backwards compatibility with traditional pip workflows:

```bash
# Generate requirements.txt with CUDA support
uv export --format requirements.txt --extra cu126 --no-dev -o requirements.txt

# Generate requirements-cpu.txt for CPU-only
uv export --format requirements.txt --extra cpu --no-dev -o requirements-cpu.txt

# Generate requirements with all extras
uv export --format requirements.txt --all-extras -o requirements-all.txt
```

## ⚙️ How It Works

### PyTorch Installation Strategy

The project uses custom PyTorch package indices defined in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu126", extra = "cu126" },
]
```

This ensures:
- PyTorch packages are fetched from the correct index based on your choice
- NVIDIA CUDA libraries are automatically bundled with the CUDA version
- CPU-only installations remain lightweight without unnecessary CUDA dependencies
- Conflict resolution prevents installing both CPU and CUDA versions

## 🐛 Troubleshooting

### "CUDA not available" when running PyTorch

Make sure you installed with the CUDA extra:
```bash
rm -rf .venv
uv sync --extra cu126
```

### Conflict errors during installation

You're trying to install both `cpu` and `cu126`. Remove the virtual environment and choose one:
```bash
rm -rf .venv
uv sync --extra cu126  # Or --extra cpu
```

### Module not found errors

Ensure you're running scripts with `uv run` or have activated the virtual environment:
```bash
uv run python script.py
# or
source .venv/bin/activate  # Then run normally
```

### Jupyter kernel not using correct environment

After installing with `--extra jupyter`, restart Jupyter:
```bash
uv run jupyter lab
```

The kernel should automatically use the `.venv` environment.

## 🔄 Migration from Old Setup

If you're migrating from an older `pip + requirements.txt` setup:

1. **Back up your old environment**:
   ```bash
   pip freeze > requirements-old.txt
   ```

2. **Install uv**:
   ```bash
   pip install uv
   ```

3. **Create new environment**:
   ```bash
   uv sync --extra cu126  # or --extra cpu
   ```

4. **Test your code**:
   ```bash
   uv run python test.py
   uv run pytest
   ```

5. **Remove old virtualenv** (if everything works):
   ```bash
   deactivate  # if in old venv
   rm -rf venv/  # or whatever your old venv directory was
   ```

## 📚 Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/) - Official uv package manager documentation
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) - PyTorch setup for different platforms
- [Project Repository](https://github.com/paulhoheisel/ml-sandbox) - Source code and issues

## 📄 License

See the repository for license information.

## 👥 Contributors

- Dominik Bereżański
- Paul Hoheisel

