# NEURO: Neural Network Domain-Specific Language

NEURO is a powerful domain-specific language (DSL) designed for creating, training, and evaluating neural networks with a clean and intuitive syntax. Built on top of PyTorch, it combines the flexibility of a high-level language with the performance of modern deep learning frameworks.

## Features

- 🧠 **Intuitive Neural Network Creation**: Define neural networks using a simple, declarative syntax
- 🚀 **PyTorch Integration**: Leverages PyTorch's powerful backend for optimal performance
- 📊 **Built-in Data Handling**: Efficient data management with the NeuroMatrix format
- 🔧 **Flexible Architecture**: Support for various neural network architectures including:
  - Feed-forward Neural Networks
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN, LSTM, GRU)
  - Attention Mechanisms
  - Residual Connections
- 🎯 **Advanced Training Features**:
  - Label Smoothing
  - Gradient Clipping
  - Learning Rate Scheduling
  - Teacher Forcing
  - Beam Search Decoding

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NEURO.git
cd NEURO

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Here's a simple example of creating and training a neural network using NEURO:

```neuro
# Create a simple neural network for binary classification
model = NeuralNetwork(input_size=3, output_size=1) {      
    Dense(units=64, activation="relu")
    Dense(units=32, activation="relu")
    Dense(units=1, activation="sigmoid")
}

# Configure loss function and optimizer
loss(type="bce")
optimizer(type="adam", learning_rate=0.001)

# Train the model
model.train(data, epochs=100)

# Evaluate the model
accuracy = model.evaluate(test_data)
```

## Project Structure

```
NEURO/
├── src/
│   ├── __init__.py
│   ├── interpreter.py    # Core interpreter implementation
│   ├── parser.py        # NEURO language parser
│   ├── matrix.py        # Data handling implementation
│   └── neuro.py         # Main entry point
├── examples/            # Example NEURO programs
├── tests/              # Unit tests
├── docs/               # Documentation
└── requirements.txt    # Project dependencies
```

## Advanced Features

### Label Smoothing

```neuro
# Enable label smoothing with custom smoothing factor
loss(type="cross_entropy", smoothing=0.1)
```

### Gradient Clipping

```neuro
# Set maximum gradient norm
model.clip_gradients(max_norm=1.0)
```

### Learning Rate Scheduling

```neuro
# Train with cosine annealing scheduler
model.train(data, epochs=100, scheduler="cosine", warmup_steps=10)
```

## Data Format

NEURO uses the NeuroMatrix format for efficient data handling:

```python
matrix = NeuroMatrix()
matrix.add_data_point(input_data=[1.0, 2.0, 3.0], output_data=[1])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](LICENSE) file for details.

When using this software, you must:
- Include the original copyright notice and license
- State significant changes made to the software
- Make source code available when distributing
- Include the following attribution in documentation or about section:
  "Based on NEURO (https://github.com/yourusername/NEURO)"

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Inspired by modern deep learning frameworks and DSLs

## Contact

For questions and support, please open an issue in the GitHub repository. 