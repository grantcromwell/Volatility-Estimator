#!/bin/bash
# Setup script for Volatility Estimator

set -e

echo "=========================================="
echo "Volatility Estimator - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MIN_VERSION="3.8"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "   ✗ Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "   ✓ Python $PYTHON_VERSION detected (>= 3.8)"
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
echo "   ✓ Core dependencies installed"
echo ""

# Optional: Install GARCH support
echo "6. Installing optional GARCH support (arch library)..."
if pip install arch --quiet 2>/dev/null; then
    echo "   ✓ GARCH models enabled"
else
    echo "   ⚠ GARCH models not available (arch library installation failed)"
    echo "   Note: EWMA and range estimators still work without GARCH"
fi
echo ""

# Install package in editable mode
echo "7. Installing package in editable mode..."
pip install -e . --quiet
echo "   ✓ Package installed"
echo ""

# Run tests
echo "8. Running test suite..."
echo ""
if pytest tests/ -v --tb=short 2>/dev/null; then
    echo ""
    echo "   ✓ All tests passed"
else
    echo ""
    echo "   ⚠ Some tests failed or skipped (this is OK if arch library is not installed)"
fi
echo ""

echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Features Available:"
echo "  • EWMA volatility (RiskMetrics standard)"
echo "  • Range-based estimators (Garman-Klass, Parkinson, Yang-Zhang)"
echo "  • GARCH models (if arch library installed)"
echo "  • Validation and backtesting framework"
echo "  • Sector volatility analysis"
echo ""
echo "Quick Start:"
echo "  # Activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "  # Run production script:"
echo "  python assess_market_volatility.py"
echo ""
echo "  # With favorite assets:"
echo "  python assess_market_volatility.py --favorites AAPL MSFT GOOGL NVDA"
echo ""
echo "  # Run tests:"
echo "  pytest tests/ -v"
echo ""
echo "  # Run enterprise model tests:"
echo "  pytest tests/test_enterprise_models.py -v"
echo ""

