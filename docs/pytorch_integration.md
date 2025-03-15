# PyTorch Integration

NEURO is built on top of the PyTorch machine learning framework. This document details how NEURO integrates PyTorch functionality and how users can access PyTorch's deeper capabilities.

## Architecture Overview

NEURO integrates PyTorch in the following ways:

1. **NeuralNetwork Class**: NEURO's `NeuralNetwork` class inherits from PyTorch's `nn.Module`.
2. **Layers**: NEURO layers use PyTorch modules (e.g., `nn.Linear`, `nn.Conv2d`).
3. **Data Structures**: NEURO uses PyTorch tensors for data storage and manipulation.
4. **Training/Optimization**: NEURO utilizes PyTorch optimizers and loss functions.

## Accessing PyTorch Modules

### Layer Definitions and Their PyTorch Equivalents

| NEURO Layer | PyTorch Module |
|-------------|---------------|
| `Dense` | `nn.Linear` |
| `Conv2D` | `nn.Conv2d` |
| `MaxPool` | `nn.MaxPool2d` |
| `LSTM` | `nn.LSTM` |
| `GRU` | `nn.GRU` |
| `Dropout` | `nn.Dropout` |
| `BatchNorm` | `nn.BatchNorm1d`, `nn.BatchNorm2d` |
| `Flatten` | `nn.Flatten` |

### NeuralNetwork and nn.Module

NEURO's `NeuralNetwork` class inherits from PyTorch's `nn.Module`, making all PyTorch module functions available. For example:

```neuro
model = NeuralNetwork(input_size=784, output_size=10) {
    Dense(units=128, activation="relu");
    Dense(units=10, activation="softmax");
}

# Using PyTorch nn.Module methods
model.to("cuda");  # Move to GPU
model.train();     # Set to training mode
model.eval();      # Set to evaluation mode
```

## Using Direct PyTorch Code

NEURO allows embedding direct PyTorch code for greater flexibility in your projects:

### Creating Custom Layers with PyTorch

```neuro
@custom_layer
def ResidualBlock(x) {
    # Direct access to PyTorch modules
    skip = x;
    conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1);
    conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1);
    bn1 = torch.nn.BatchNorm2d(64);
    bn2 = torch.nn.BatchNorm2d(64);
    
    out = conv1(x);
    out = bn1(out);
    out = torch.nn.functional.relu(out);
    out = conv2(out);
    out = bn2(out);
    
    # PyTorch operations
    out = out + skip;
    return torch.nn.functional.relu(out);
}
```

### Accessing PyTorch Functions

NEURO's interpreter allows direct access to PyTorch modules:

```neuro
# Using PyTorch functions
x = torch.randn(10, 10);
y = torch.nn.functional.softmax(x, dim=1);
z = torch.argmax(y, dim=1);
```

## Importing Existing PyTorch Models

NEURO provides the ability to use existing PyTorch models:

```neuro
# Load a pre-trained PyTorch model
@pretrained("resnet18")
backbone = Backbone(trainable=false);

model = NeuralNetwork(input_size=(3, 224, 224), output_size=10) {
    backbone(x);
    Flatten();
    Dense(units=10, activation="softmax");
}
```

## Exporting Models in PyTorch Format

NEURO models are PyTorch models, so they can be directly exported in PyTorch format:

```neuro
model = NeuralNetwork(input_size=784, output_size=10) {
    Dense(units=128, activation="relu");
    Dense(units=10, activation="softmax");
}

# Save in PyTorch format
model.save("model.pt");

# Can be loaded with native PyTorch code
# import torch
# model = torch.load("model.pt")
```

## Performance Optimization

NEURO automatically leverages PyTorch's performance optimization capabilities:

### GPU Acceleration

```neuro
# Using GPU
model.to("cuda");

# Using multiple GPUs
model.to("cuda");
model = torch.nn.DataParallel(model);
```

### Quantization

```neuro
# Model quantization
# First install: pip install torch-quantization
import torch.quantization

# Define the model
model = NeuralNetwork(input_size=784, output_size=10) {
    Dense(units=128, activation="relu");
    Dense(units=10, activation="softmax");
}

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
);
```

## Troubleshooting

If you encounter issues with PyTorch integration, check:

1. **PyTorch Version**: Ensure you're using PyTorch 2.6.0 or newer.
2. **CUDA Compatibility**: If using GPU, check that your CUDA version is compatible with your PyTorch version.
3. **Memory Issues**: For memory problems, use smaller batch sizes or activate gradient checkpointing.

```neuro
# Activate gradient checkpointing
model.train(data, 
    epochs=100, 
    batch_size=32, 
    gradient_checkpoint=true
);
```

## Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [NEURO Examples](examples.md)
- [Advanced Practices](best_practices.md) 