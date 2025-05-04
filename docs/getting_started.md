# Getting Started with NEURO (Alpha)

Welcome to NEURO, a domain-specific language for neural network development. This guide will help you get started quickly.

## 1. Installation

First, ensure you have Python (>= 3.8) and pip installed. Then, follow these steps:

```bash
# Clone the repository
git clone https://github.com/PanzerPeter/NEURO.git # Or your fork's URL
cd NEURO

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install NEURO in editable mode
pip install -e .
```

This makes the `neuro` command available in your environment.

## 2. Your First NEURO Script

Create a file named `hello.nr` with the following content:

```neuro
// hello.nr: Define a simple model and print a message

// Define a basic neural network (won't be trained here)
simple_model = NeuralNetwork() { // Parameters like input/output size are optional
    Dense(units=10, activation="relu");
    Dense(units=1, activation="sigmoid");
};

print("NEURO model defined successfully!");
// print("Model structure (placeholder):", simple_model); // Printing model objects directly might give limited info
```

## 3. Running the Script

From your terminal (in the `NEURO` project directory where you installed the package), run:

```bash
neuro run hello.nr
```

You should see output indicating the script is running and the print statements being executed.

## Next Steps

- Explore the examples in the `examples/` directory.
- Read the `language_guide.md` for detailed syntax information.
- Check the `README.md` for current features and project status.

