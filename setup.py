"""
Setup script for TinySearch
"""
from setuptools import setup, find_packages

# Read version from __init__.py
version = {}
with open("tinysearch/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Read README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define dependencies
install_requires = [
    "numpy>=1.19.0",
    "pyyaml>=5.1.0",
    "tqdm>=4.60.0",
    "watchdog>=2.1.0",
]

extras_require = {
    "api": [
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
    ],
    "embedders": [
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "sentence-transformers>=2.0.0",
    ],
    "adapters": [
        "pymupdf>=1.18.0",
        "pypdf2>=1.26.0",
    ],
    "indexers": [
        "faiss-cpu>=1.7.0",
    ],
    "dev": [
        "pytest>=6.0.0",
        "black>=21.5b2",
        "isort>=5.9.0",
        "mypy>=0.812",
        "flake8>=3.9.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "myst-parser>=0.15.0",
    ],
}

# Full dependencies
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name="tinysearch",
    version=version.get("__version__", "0.1.0"),
    author="TinySearch Team",
    author_email="tinysearch@example.com",
    description="A lightweight vector retrieval system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tinysearch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "tinysearch=tinysearch.cli:main",
            "tinysearch-api=tinysearch.api:start_api",
        ],
    },
) 