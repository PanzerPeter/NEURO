# Frequently Asked Questions (FAQ)

## General Questions

### What is NEURO?
NEURO is a domain-specific language (DSL) for neural network development. It provides an intuitive, declarative syntax for building and training deep learning models, built on top of PyTorch.

### Why use NEURO instead of direct PyTorch?
NEURO simplifies neural network development in the following ways:
- More intuitive syntax
- Automation of common tasks
- Automated memory management
- Built-in best practices
- Less boilerplate code
- Better error messages

### What are the system requirements?
- Python 3.8 or newer
- CUDA-capable GPU (optional, but recommended)
- At least 4GB RAM (8GB+ recommended)
- Operating Systems: Windows, Linux, or macOS

## Installation and Setup

### How do I install NEURO?
```bash
# From source (recommended for development)
git clone https://github.com/PanzerPeter/NEURO.git
cd NEURO
python -m venv venv
# Windows: venv\Scripts\activate
# Unix/macOS: source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### How do I enable GPU support?
NEURO automatically detects and uses available GPUs. Make sure you have:
1. CUDA-capable GPU
2. Installed CUDA toolkit
3. PyTorch with CUDA support

### Can I use NEURO in Jupyter notebooks?
Yes! Install the Jupyter kernel:
```bash
pip install neuro-jupyter
python -m neuro_jupyter.install
```

## Language Features

### How do I handle different input sizes?
Use dynamic dimensions:
```neuro
model = NeuralNetwork(input_size=(None, 3, 224, 224)) {
    # Model definition
}
```

### Does NEURO support transfer learning?
Yes, NEURO has built-in support for using pre-trained models:
```neuro
@pretrained("resnet18")
backbone = Backbone(trainable=false);

model = NeuralNetwork(input_size=(3, 224, 224), output_size=10) {
    backbone(x);
    Flatten();
    Dense(units=10, activation="softmax");
}
```

### How does NEURO handle memory for large models?
NEURO automatically optimizes memory usage and supports gradient checkpointing:
```neuro
model.train(data, epochs=100, gradient_checkpoint=true);
```

## Performance Questions

### How fast is NEURO compared to native PyTorch?
NEURO adds minimal overhead compared to PyTorch since it's built directly on PyTorch. The performance difference is typically less than 5%.

### How can I optimize NEURO models?
- Use appropriate batch sizes
- Enable gradient checkpointing for large models
- Use `model.to("cuda")` for GPU acceleration
- Apply the `@timer` decorator to identify performance bottlenecks

### Does NEURO support quantization?
Yes, NEURO supports PyTorch's quantization capabilities to reduce model size and speed up inference.

## Troubleshooting

### My model isn't converging. What should I do?
1. Check the learning rate
2. Try different optimizers
3. Normalize your input data
4. Check for vanishing or exploding gradients
5. Use batch normalization

### I'm getting CUDA memory errors. How can I fix them?
1. Reduce the batch size
2. Enable gradient checkpointing
3. Use fewer or narrower layers
4. Avoid using excessively large tensors

### How does NEURO handle dependencies?
NEURO uses the following main dependencies:
- PyTorch 2.6.0 or newer
- NumPy 2.2.0 or newer
- PLY 3.11 or newer
- Click 8.1.8 or newer

## Integrations

### Can I use custom PyTorch modules in NEURO?
Yes, NEURO allows integration of custom PyTorch modules. See the [PyTorch Integration Guide](pytorch_integration.md) for more details.

### How do I export NEURO models?
NEURO models can be directly exported in PyTorch format:
```neuro
model.save("model.pt");
```

### Is NEURO compatible with existing ML pipelines?
Yes, NEURO models are compatible with most ML pipelines that use PyTorch models, including experiment tracking tools, visualization tools, and serving platforms.

## Version Control and Compatibility

### What Python versions does NEURO support?
NEURO supports Python 3.8 or newer, including versions 3.9, 3.10, 3.11, and 3.12.

### What is NEURO's licensing?
NEURO is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](https://github.com/PanzerPeter/NEURO/blob/main/LICENSE) file for more details.

### Where can I report bugs or request features?
You can report bugs and feature requests in the GitHub issue tracker: [https://github.com/PanzerPeter/NEURO/issues](https://github.com/PanzerPeter/NEURO/issues). 