#!/bin/bash
# Setup script for Volatility Estimator

set -e

echo "=========================================="
echo "Volatility Estimator - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   ✓ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "4. Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓ pip upgraded"
echo ""

# Install dependencies
echo "5. Installing dependencies..."
pip install -r requirements.txt
echo "   ✓ Dependencies installed"
echo ""

# Install package in editable mode
echo "6. Installing package in editable mode..."
pip install -e . --quiet
echo "   ✓ Package installed"
echo ""

# Run tests
echo "7. Running tests..."
echo ""
python3 test_volatility.py
echo ""

echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""
echo "To run examples:"
echo "  python3 example_usage.py"
echo ""

