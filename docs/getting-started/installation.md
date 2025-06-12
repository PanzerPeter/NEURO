# Installation Guide

This guide covers installing NEURO on all supported platforms and setting up your development environment.

## Prerequisites

### Required Dependencies
- **Python 3.8+** - For the compiler toolchain
- **Git** - For downloading the project

### Optional Dependencies (for full functionality)
- **LLVM 14+** - For native code generation (recommended)
- **CUDA SDK** - For GPU acceleration (NVIDIA)
- **OpenCL** - For GPU acceleration (AMD/Intel)

## Platform-Specific Installation

### Windows

To compile native executables on Windows, NEURO requires the Microsoft C++ build toolchain (for the linker) and LLVM/Clang.

#### 1. Install Build Tools
First, install the necessary C++ build tools from Microsoft:
1.  Go to the [Visual Studio downloads page](https://visualstudio.microsoft.com/downloads/) and download "Build Tools for Visual Studio".
2.  Run the installer and select the **"Desktop development with C++"** workload. This will install the required compiler and linker.

#### 2. Install LLVM
1.  Download and install LLVM from the [official LLVM site](https://llvm.org/releases/).
2.  During installation, select the option **"Add LLVM to the system PATH"**.

#### 3. Install NEURO
```powershell
# Clone the NEURO repository
git clone https://github.com/your-org/neuro.git
cd neuro

# Set up a Python virtual environment
python -m venv neuro-env
.\neuro-env\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 4. Setting up the Compilation Environment
On Windows, you **must** use a special developer terminal to compile NEURO programs. This ensures `clang` can find the Microsoft linker.

1.  Open the Windows Start Menu.
2.  Search for **"x64 Native Tools Command Prompt for VS"** and open it.
3.  In this new terminal, navigate to the NEURO project directory:
    ```powershell
    cd path\to\neuro
    ```
4.  Activate the Python environment and set the `PYTHONPATH`:
    ```powershell
    .\neuro-env\Scripts\activate
    set PYTHONPATH=%cd%\src;%PYTHONPATH%
    ```
Now you are ready to compile.

### Linux/macOS

#### Method 1: Shell Script
```bash
# Download and run installation script
curl -sSL https://install.neuro-lang.org/install.sh | bash

# Or manual installation:
git clone https://github.com/your-org/neuro.git
cd neuro

# Set up Python environment
python3 -m venv neuro-env
source neuro-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up PYTHONPATH
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Test installation
python src/neuro/main.py examples/hello_world.nr
```

## LLVM Setup (Optional but Recommended)

### Windows
1. Download LLVM from [llvm.org](https://llvm.org/releases/)
2. Install to default location (`C:\Program Files\LLVM`)
3. Add to PATH: `C:\Program Files\LLVM\bin`

### Linux
```bash
# Ubuntu/Debian
sudo apt-get install llvm-14-dev clang-14

# Fedora/CentOS
sudo dnf install llvm-devel clang-devel

# Arch Linux
sudo pacman -S llvm clang
```

### macOS
```bash
# Using Homebrew
brew install llvm

# Add to PATH
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
```

## GPU Support Setup

### NVIDIA CUDA
```bash
# Download from nvidia.com/cuda
# Follow platform-specific installation

# Verify installation
nvcc --version
```

### AMD ROCm (Linux only)
```bash
# Ubuntu
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm-dev
```

## Verification

### Basic Functionality

To verify your installation, run the following commands **from the correct terminal** (on Windows, this must be the x64 Native Tools Command Prompt):

```bash
# Test compiler
python src/neuro/main.py --version

# Test compilation
python src/neuro/main.py examples/hello_world.nr -o hello_world.exe

# Test with LLVM (if installed)
python src/neuro/main.py examples/hello_world.nr --emit-llvm-ir

# Run compiled output
./hello_world.exe  # Windows
./hello_world      # Linux/macOS
```

### GPU Functionality (if applicable)
```bash
# Test GPU detection
python src/neuro/main.py --list-devices

# Test GPU compilation
python src/neuro/main.py examples/gpu_example.nr --target gpu
```

## Environment Configuration

### Setting Up PATH
```bash
# Add to ~/.bashrc, ~/.zshrc, or equivalent
export PATH="$PATH:/path/to/neuro/bin"
export PYTHONPATH="/path/to/neuro/src:$PYTHONPATH"
```

### IDE Integration
See [Development Environment Setup](development-environment.md) for IDE-specific configuration.

## Troubleshooting

### Common Issues

#### Python Path Issues
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Fix: Set PYTHONPATH explicitly
export PYTHONPATH="/full/path/to/neuro/src:$PYTHONPATH"
```

#### LLVM Not Found
```bash
# Check LLVM installation
llvm-config --version

# Fix: Update PATH
export PATH="/usr/lib/llvm-14/bin:$PATH"
```

#### Permission Issues (Linux/macOS)
```bash
# Fix Python virtual environment permissions
chmod +x neuro-env/bin/activate

# Fix script permissions
chmod +x scripts/*.sh
```

## Next Steps

After successful installation:

1. **[Take the First Steps](first-steps.md)** - Write your first NEURO program
2. **[Set up Development Environment](development-environment.md)** - Configure your IDE
3. **[Explore Examples](../examples/README.md)** - Run example programs

## Getting Help

- **Documentation**: [Complete documentation index](../README.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/neuro/issues)
- **Community**: [Discord Server](https://discord.gg/neuro-lang)
- **Email**: support@neuro-lang.org 