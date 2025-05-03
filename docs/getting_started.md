# Getting Started with NEURO

Welcome to NEURO, a domain-specific language designed for neural network development and deep learning experimentation. This guide will help you get started with NEURO quickly and effectively.

## Installation

```bash
pip install neuro-lang
```

## Your First Neural Network

Let's create a simple neural network for binary classification:

```neuro
model = NeuralNetwork(input_size=2, output_size=1) {
    Dense(units=64, activation="relu");
    Dense(units=32, activation="relu");
    Dense(units=1, activation="sigmoid");
}

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
- Learn about [Advanced Features](advanced_features.md)
- Review [Best Practices](best_practices.md)

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

## Need Help?

- Check our [FAQ](faq.md)
- Join our community on Discord
- Report issues on GitHub 