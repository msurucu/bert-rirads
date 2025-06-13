"""
Setup script for RI-RADS Text Classification Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "RI-RADS Text Classification Tool"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="rirads-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning-based text classification for RI-RADS scoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rirads-scoring-tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pytest-cov>=4.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0,<2.14.0",
        ],
        "medical": [
            "bio-transformers>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rirads-train=train_rirads:main",
            "rirads-predict=predict_rirads:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rirads_classifier": ["*.json", "*.yaml"],
    },
)