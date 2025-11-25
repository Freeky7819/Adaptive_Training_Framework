#!/usr/bin/env python3
"""
Adaptive Training Framework - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements = [
    "torch>=1.12.0",
    "torchvision>=0.13.0",
    "numpy>=1.21.0",
]

optional_requirements = {
    "tensorboard": ["tensorboard>=2.10.0"],
    "full": [
        "tensorboard>=2.10.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "mypy>=0.960",
    ],
}

setup(
    name="adaptive-training-framework",
    version="1.0.0",
    author="Damjan Žakelj",
    author_email="",
    description="Resonance-based neural network training optimization framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Freeky7819/Adaptive_Training_Framework",
    project_urls={
        "Bug Tracker": "https://github.com/Freeky7819/Adaptive_Training_Framework/issues",
        "Documentation": "https://github.com/Freeky7819/Adaptive_Training_Framework#readme",
        "Results": "https://github.com/Freeky7819/Adaptive_Training_Framework/blob/main/RESULTS.md",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=optional_requirements,
    entry_points={
        "console_scripts": [
            "atf-train=cli.run:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
