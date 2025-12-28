"""
Setup script for Volatility Estimator package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Separate optional dependencies
optional_requirements = {
    "garch": ["arch>=6.2.0"],  # GARCH models (optional but recommended)
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
    "all": ["arch>=6.2.0"],  # All optional dependencies
}

setup(
    name="volatility-estimator",
    version="1.0.0",
    author="VolEstimator Team",
    description="Volatility estimator with industry-standard models (EWMA, GARCH) and advanced techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/VolEstimator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=optional_requirements,
    entry_points={
        "console_scripts": [
            "assess-volatility=assess_market_volatility:main",
        ],
    },
    keywords=[
        "volatility",
        "finance",
        "risk",
        "garch",
        "ewma",
        "quantitative",
        "trading",
        "investment",
        "regime-detection",
        "correlation-analysis",
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/VolEstimator#readme",
        "Source": "https://github.com/yourusername/VolEstimator",
        "Tracker": "https://github.com/yourusername/VolEstimator/issues",
    },
)

