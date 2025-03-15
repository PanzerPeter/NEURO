"""
NEURO Language Setup Configuration

This file contains the package setup configuration for NEURO.
It defines how the package should be installed and its metadata.
"""

from setuptools import setup, find_packages

setup(
    name="neuro-lang",
    version="0.1.0",
    description="A domain-specific language for neural network development",
    author="NEURO Team",
    packages=find_packages(),
    install_requires=[
        # Dependencies will be read from requirements.txt
    ],
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
    ],
) 