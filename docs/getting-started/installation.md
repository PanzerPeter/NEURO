# Installation and Setup

The NEURO compiler is currently implemented in Python. To use it, you will need a Python environment and the source code.

## Prerequisites

*   Python 3.8 or higher.
*   Git (for cloning the repository).

## 1. Get the Source Code

First, clone the NEURO repository from GitHub to your local machine:

```bash
git clone <repository-url>
cd Neuro1.0
```
*(Replace `<repository-url>` with the actual URL of the repository.)*

## 2. Set up a Virtual Environment (Recommended)

It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other projects.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\\Scripts\\activate
# On macOS/Linux:
source .venv/bin/activate
```

## 3. Running the Compiler

The main entry point for the compiler is the `neurc` command, which is an alias for running the `src/neuro/main.py` script. You can run the compiler using the following command:

```bash
python -m neuro.main [options] <input-file>
```

### Example: Compile a "Hello, World!" program

1.  Create a file named `hello.nr` with the following content:
    ```neuro
    print("Hello, World!")
    ```

2.  Compile it from your terminal:
    ```bash
    python -m neuro.main examples/basics/hello_world.nr
    ```

    This command will compile `hello.nr` and create an executable file in your project directory (e.g., `hello_world.bat` on Windows or `hello_world` on Linux/macOS).

3.  Run the compiled program:
    ```bash
    # On Windows
    .\\hello_world.bat

    # On macOS/Linux
    ./hello_world
    ```

    You should see the output: `Hello, World!`

## Next Steps

Now that you have the compiler running, you can proceed to the next section to learn more about the language.

*   **[Your First Program](./your-first-program.md)** *(Coming Soon)* 