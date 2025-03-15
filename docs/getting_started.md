# Getting Started with NEURO

Welcome to NEURO, a domain-specific language designed for neural network development and deep learning experimentation. This guide will help you get started with NEURO quickly and effectively.

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for deep learning)
- 4GB RAM minimum (8GB+ recommended)
- Operating Systems: Windows, Linux, or macOS

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/PanzerPeter/NEURO.git
cd NEURO

# Create and activate virtual environment (optional but recommended)
python -m venv venv

# On Windows:
# venv\Scripts\activate

# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

NEURO depends on the following packages:
- PyTorch 2.6+
- NumPy 2.2+
- PLY 3.11+
- Click 8.1+

## Your First Neural Network

Let's create a simple neural network for binary classification:

```neuro
model = NeuralNetwork(input_size=2, output_size=1) {
    Dense(units=64, activation="relu");
    Dense(units=32, activation="relu");
    Dense(units=1, activation="sigmoid");
}

# Configure loss and optimizer
loss = Loss(type="binary_crossentropy");
optimizer = Optimizer(type="adam", learning_rate=0.001);

# Train the model
model.train(data, epochs=10, batch_size=32);
```

## Basic Concepts

NEURO is built around these core concepts:

1. **Neural Networks**: Define models using a clear, intuitive syntax
2. **Layers**: Build networks using pre-defined or custom layers
3. **Training**: Simple interface for model training and evaluation
4. **Data Handling**: Built-in tools for data preprocessing and management

## Key Features

- Intuitive syntax for neural network definition
- Built-in support for common layer types
- Automatic shape inference
- Integrated data preprocessing
- Custom layer support
- Transfer learning capabilities

## Next Steps

- Read the [Language Guide](language_guide.md) for detailed syntax information
- Check out [Examples](examples.md) for common use cases
- Review [Best Practices](best_practices.md)
- See the [FAQ](faq.md) for common questions

## Quick Tips

1. Use the REPL for quick experimentation:
   ```bash
   neuro
   ```

2. Save and load models:
   ```neuro
   model.save("my_model.pt");
   loaded = load_model("my_model.pt");
   ```

3. Data preprocessing:
   ```neuro
   data = load_matrix("data.csv");
   data.normalize();
   ```

## File Extensions

- `.nr`: NEURO source code files
- `.nrm`: NEURO Matrix data files

## Need Help?

- Check our [FAQ](faq.md)
- Report issues on the [GitHub repository](https://github.com/PanzerPeter/NEURO/issues) 