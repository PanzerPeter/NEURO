from setuptools import setup, find_packages

setup(
    name="neuro-lang",
    version="0.1.0-alpha",
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
    description="NEURO Programming Language - A language designed for AI development (ALPHA VERSION)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="ai, machine learning, programming language, alpha, experimental",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.7",
    project_urls={
        "Homepage": "https://github.com/PanzerPeter/NEURO",
        "Bug Tracker": "https://github.com/PanzerPeter/NEURO/issues",
        "Documentation": "https://github.com/PanzerPeter/NEURO#readme",
    },
) 