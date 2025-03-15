"""
NEURO Language Setup Configuration

This file contains the package setup configuration for NEURO.
It defines how the package should be installed and its metadata.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = []
    for line in f:
        line = line.strip()
        if line and not line.startswith(('#', '"', "'")):
            requirements.append(line)

setup(
    name="neuro-lang",
    version="0.1.0",
    description="A domain-specific language for neural network development",
    author="NEURO Team",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'neuro=src.neuro:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires='>=3.8',
) 