from setuptools import setup, find_packages

setup(
    name="neuro-lang",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ply>=3.11",
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "pytest>=7.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        'console_scripts': [
            'neuro=src.neuro:main',
        ],
    },
    author="NEURO Development Team",
    description="NEURO Programming Language - A language designed for AI development",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="ai, machine learning, programming language",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
) 