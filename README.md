# NEURO: Neural Network Domain-Specific Language

NEURO is a domain-specific language (DSL) designed for creating, training, and evaluating neural networks with a clean and intuitive syntax. Built on top of PyTorch, it combines the flexibility of a high-level language with the performance of modern deep learning frameworks.

> ⚠️ **ALPHA VERSION WARNING**
>
> NEURO is currently in ALPHA stage. This means:
> - The API may change without notice
> - Features might be incomplete or have bugs
> - Production use is NOT recommended
> - No warranty or liability is provided (see LICENSE or `pyproject.toml`)
>
> **USE AT YOUR OWN RISK**

## Features (Current Implementation - Alpha)

- 🧠 **Declarative Neural Network Definition**: Define network architectures using a dedicated syntax.
- 🚀 **PyTorch Backend**: Leverages PyTorch for network construction and training.
- 📊 **Data Handling**: Basic data loading via `NeuroMatrix` YAML format and `load_matrix` function.
- 🧩 **Core Layers**: 
  - `Dense` (Fully Connected)
  - `Conv2D` (2D Convolution)
  - `Flatten`
  - `BatchNorm` (Infers 1D/2D from context, uses `LazyBatchNorm1d` / `BatchNorm2d`)
- ✨ **Activations**: `ReLU`, `Sigmoid`, `Tanh`, `Softmax` (applied within layers).
- 📉 **Loss Functions**: `BCELoss`, `MSELoss`, `CrossEntropyLoss` (configured via `Loss(...)` assignment).
- ⚙️ **Optimizers**: `Adam`, `SGD`, `RMSprop` (configured via `Optimizer(...)` assignment).
- 💪 **Basic Training Loop**: `model.train(data, ...)` with support for epochs, batch size, and optional validation data.
- 📝 **Model Saving/Loading**: Basic checkpointing (`model.save(...)`) and loading (`model.load(...)`) - *Note: Loading currently requires manual model re-definition before loading weights.*
- ↔️ **Tuple Assignment**: Assign results from functions like `data.split()` to multiple variables.

_(Many advanced features listed previously, like RNNs, attention, label smoothing, etc., are planned but not yet implemented)._

## Installation

```bash
# Clone the repository
git clone https://github.com/PanzerPeter/NEURO.git # Replace with your URL if different
cd NEURO

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install NEURO and its dependencies in editable mode
pip install -e .

# To install development tools (for testing, linting):
pip install -e ".[dev]"
```

## Quick Start

Here's a simple example using the current NEURO syntax:

```neuro
# Load data from a .nrm file (YAML format)
data = load_matrix("path/to/your_data.nrm"); 

# (Optional) Split data
# train_data, val_data = data.split(train_frac=0.8);

# Create a simple neural network for binary classification
# Note: input_size/output_size in NeuralNetwork() are currently informational
my_model = NeuralNetwork(input_size=10, output_size=1) {      
    Dense(units=64, activation="relu"); # Layer definitions require semicolons
    Dense(units=32, activation="relu");
    Dense(units=1, activation="sigmoid");
};

# Configure loss function and optimizer via assignment
# These configure the *next* training run
loss_cfg = Loss(type="bce");
opt_cfg = Optimizer(type="adam", learning_rate=0.001);

# Train the model using the loaded data and configured loss/optimizer
# Training parameters like epochs are passed directly to train()
# The variables loss_cfg and opt_cfg are not used directly here;
# the interpreter uses the *last defined* Loss/Optimizer config.
train_history = my_model.train(data=data, epochs=50, batch_size=16);

# Evaluate the model (Evaluation logic might be placeholder)
# test_data = load_matrix("path/to/test_data.nrm"); 
# results = my_model.evaluate(data=test_data);

# Save the trained model (weights and metadata)
# my_model.save("my_trained_model.pt"); 

print("Training complete."); # Example of using built-in print

```

## Project Structure

```
NEURO/
├── src/                 # Source code (lexer, parser, interpreter, models, etc.)
│   ├── __init__.py
│   ├── parser.py
│   ├── interpreter.py
│   ├── neuro_ast.py
│   ├── models.py
│   ├── layers.py 
│   ├── losses.py
│   ├── optimizers.py
│   ├── matrix.py
│   ├── type_checker.py
│   ├── neuro_types.py
│   └── errors.py
├── neuro.py         # Main script logic (used by entry point)
├── examples/            # Example NEURO programs (.nr files)
├── docs/                # Documentation files (Markdown)
├── .gitignore
├── LICENSE              # License file (GPL-3.0)
├── pyproject.toml       # Build configuration and dependencies
└── README.md            # This file
```

## Data Format (`.nrm` files)

NEURO uses a YAML-based format (`.nrm`) managed by the `NeuroMatrix` class. Files should contain `metadata` and `data` sections.

```yaml
metadata:
  description: "Simple dataset for demonstration"
  num_samples: 5
  input_features: 3
  output_features: 1

data:
  - input: [1.0, 2.0, 3.0]
    output: [1]
  - input: [4.0, 5.0, 6.0]
    output: [0]
  # ... more data points
```

Use the built-in `load_matrix("path/to/file.nrm")` function within your `.nr` scripts to load data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Install dev tools (`pip install -e ".[dev]"`) and run tests (`pytest -v`)
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later) - see the [LICENSE](LICENSE) file for details.

When using this software, please adhere to the terms of the GPL-3.0 license, including providing attribution: "Based on NEURO (https://github.com/PanzerPeter/NEURO)".

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)

## Contact

For questions and support, please open an issue in the GitHub repository: [https://github.com/PanzerPeter/NEURO/issues](https://github.com/PanzerPeter/NEURO/issues)

## Usage

### Interactive REPL

If you run the interpreter without any arguments, it will start an interactive Read-Eval-Print Loop (REPL):

```bash
python neuro.py 
```

This will present you with a `neuro>` prompt where you can type Neuro commands one line at a time. The interpreter will execute each line and print the result (if any) or report errors.

To exit the REPL, type `exit()` or `quit()` and press Enter, or press `Ctrl+D` (on Linux/macOS) or `Ctrl+Z` then Enter (on Windows).

## Features

// ... existing code ... 