# NEURO Programming Language

**A high-performance, compiled programming language for AI development**

NEURO is a modern programming language designed specifically for artificial intelligence and machine learning applications. It combines the performance of compiled languages like C++ with the productivity and safety of modern languages like Rust, while providing first-class support for AI workloads.

## ✨ Key Features

- 🚀 **High Performance**: Compiles to native machine code via LLVM
- 🧠 **AI-First Design**: Built-in tensors, neural networks, and GPU acceleration  
- 🔧 **Static Typing**: Type inference with compile-time optimization
- ⚡ **Zero-Cost Abstractions**: High-level features with no runtime overhead
- 🎯 **Memory Safety**: Automatic Reference Counting (ARC) for predictable performance
- 🔀 **General Purpose**: Full programming language capabilities beyond AI

## 🚀 Quick Start

### Installation

To get started with NEURO, clone the repository and set up your environment:

```bash
git clone <repository-url>
cd Neuro1.0

# Set up Python path (Windows)
set PYTHONPATH=%cd%\src;%PYTHONPATH%

# Set up Python path (Linux/Mac)
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### Hello, World!

Create a file named `hello.nr`:

```neuro
func main() {
    let message = "Hello, NEURO World!"
    print(message)

    let data: Tensor<f32> = [1.0, 2.0, 3.0, 4.0]
    let result = data @ data
    print("Dot product: " + str(result))
}
```

Compile and run your first NEURO program:

```bash
python src/neuro/main.py hello.nr
# On successful compilation, an executable will be created.
```

## Syntax Examples

**Declarative Neural Network Definition:**
```neuro
// Type inference with tensor operations
let model = NeuralNetwork<f32, (784, 128, 10)> {
    dense_layer<128>(.relu),
    batch_norm(),
    dropout(0.2),
    dense_layer<10>(.softmax)
}
```

**Optimized Training Loop:**
```neuro
// Training with automatic optimization
for batch in training_data.batches(32) {
    let loss = model.forward(batch.inputs).cross_entropy(batch.labels)
    model.backward(loss)
    optimizer.step()
}
```

**GPU Kernel Compilation:**
```neuro
// Generates optimized CUDA/OpenCL kernels
@gpu func matrix_multiply<T>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
    return a @ b
}
```

## Development Architecture

NEURO's compilation pipeline is designed for performance and extensibility:

1.  **Source Code (`.nr`)** → **Lexer/Parser** → **Abstract Syntax Tree (AST)**
2.  **AST** → **Type Checker** → **Type-Annotated AST**
3.  **Type-Annotated AST** → **High-Level IR** → **Optimizer**
4.  **Optimized IR** → **LLVM IR** → **Native Code**

## Contributing

Contributions are welcome. The project is structured to facilitate development:

```
src/neuro/
├── lexer.py
├── parser.py
├── ast_nodes.py
├── type_checker.py
├── compiler.py
├── errors.py
└── main.py

tests/
├── test_lexer.py
├── test_parser.py
└── test_integration.py
```

Please run the test suite to ensure your changes don't break existing functionality:
```bash
python -m pytest tests/ -v
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
