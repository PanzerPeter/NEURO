# First Steps with NEURO

Welcome to NEURO! This guide will get you writing and running NEURO programs quickly. By the end, you'll understand the basics and be ready to explore more advanced features.

## Your First NEURO Program

Let's start with the classic "Hello, World!" program:

### Create hello.nr
```neuro
func main() {
    print("Hello, NEURO World!")
}
```

### Compile and Run

On Linux or macOS, you can compile and run from any standard terminal. On Windows, you **must** use the **x64 Native Tools Command Prompt for VS** that was set up during installation.

```bash
# Compile the program
python src/neuro/main.py hello.nr -o hello.exe

# Run the compiled output
./hello.exe    # Windows
./hello        # Linux/macOS
```

**Expected Output:**
```
Hello, NEURO World!
```

## Understanding NEURO Syntax

### Functions and Variables
```neuro
func greet(name: String) -> String {
    let greeting = "Hello, " + name + "!"
    return greeting
}

func main() {
    let name = "Alice"
    let message = greet(name)
    print(message)
}
```

Key concepts:
- `func` declares functions
- `let` declares immutable variables (type inference)
- `->` indicates return type
- String concatenation with `+`

### Type System
```neuro
func type_examples() {
    // Explicit types
    let integer: int = 42
    let floating: float = 3.14
    let text: String = "NEURO"
    let flag: bool = true
    
    // Type inference (preferred)
    let auto_int = 42           // inferred as int
    let auto_float = 3.14       // inferred as float
    let auto_string = "NEURO"   // inferred as String
    let auto_bool = true        // inferred as bool
    
    // Arrays
    let numbers = [1, 2, 3, 4, 5]              // Array<int>
    let floats: Array<float> = [1.0, 2.0, 3.0]
    
    print("Integer: " + str(integer))
    print("Float: " + str(floating))
    print("String: " + text)
    print("Boolean: " + str(flag))
}
```

### Control Flow
```neuro
func control_flow_examples() {
    // If-else statements
    let age = 25
    if age >= 18 {
        print("Adult")
    } else {
        print("Minor")
    }
    
    // Loops
    for i in 0..5 {
        print("Count: " + str(i))
    }
    
    let numbers = [1, 2, 3, 4, 5]
    for num in numbers {
        print("Number: " + str(num))
    }
    
    // While loop
    let mut counter = 0
    while counter < 3 {
        print("Counter: " + str(counter))
        counter = counter + 1
    }
}
```

## AI-Focused Features

NEURO's strength lies in its AI capabilities. Let's explore some basic AI features:

### Tensors (Multi-dimensional Arrays)
```neuro
func tensor_basics() {
    // Create tensors
    let vector: Tensor<float> = [1.0, 2.0, 3.0, 4.0]
    let matrix: Tensor<float> = [[1.0, 2.0], [3.0, 4.0]]
    
    // Tensor operations
    let scaled = vector * 2.0
    let sum = vector + [0.5, 0.5, 0.5, 0.5]
    
    // Matrix multiplication
    let result = matrix @ matrix  // @ is matrix multiplication
    
    print("Original vector: " + str(vector))
    print("Scaled vector: " + str(scaled))
    print("Matrix result: " + str(result))
}
```

### Simple Neural Network
```neuro
func simple_neural_network() {
    // Define a simple neural network
    let model = NeuralNetwork<float, (2, 3, 1)> {
        dense_layer(units=3, activation=relu),
        dense_layer(units=1, activation=sigmoid)
    }
    
    // Sample input
    let input: Tensor<float> = [0.5, 0.8]
    
    // Forward pass
    let output = model.forward(input)
    print("Neural network output: " + str(output))
}
```

## Working with Files

### Reading Input
```neuro
func file_operations() {
    // Read from file
    let content = read_file("data.txt")
    print("File content: " + content)
    
    // Write to file
    write_file("output.txt", "Hello from NEURO!")
    
    // Process CSV data
    let data = load_csv("dataset.csv")
    print("Loaded " + str(data.rows()) + " rows")
}
```

## Error Handling

```neuro
func error_handling_example() {
    // Result type for operations that might fail
    let result = divide(10.0, 0.0)
    
    match result {
        Ok(value) -> print("Result: " + str(value)),
        Err(error) -> print("Error: " + error.message())
    }
}

func divide(a: float, b: float) -> Result<float, String> {
    if b == 0.0 {
        return Err("Division by zero")
    } else {
        return Ok(a / b)
    }
}
```

## Complete Example Program

Let's put it all together with a practical example:

### data_analysis.nr
```neuro
func analyze_data(filename: String) {
    print("Starting data analysis...")
    
    // Load and preprocess data
    let raw_data = load_csv(filename)
    let cleaned_data = raw_data
        .remove_missing()
        .normalize_columns([0, 1, 2])
    
    print("Loaded " + str(cleaned_data.rows()) + " samples")
    
    // Simple statistics
    let features = cleaned_data.get_columns([0, 1, 2])
    let means = calculate_means(features)
    let stds = calculate_stds(features)
    
    print("Feature means: " + str(means))
    print("Feature stds: " + str(stds))
    
    // Simple linear model
    let model = LinearModel<float> {
        input_size: 3,
        output_size: 1
    }
    
    // Training data
    let X = cleaned_data.get_columns([0, 1, 2])
    let y = cleaned_data.get_column(3)
    
    // Train model
    for epoch in 0..100 {
        let predictions = model.forward(X)
        let loss = mean_squared_error(predictions, y)
        
        if epoch % 20 == 0 {
            print("Epoch " + str(epoch) + ", Loss: " + str(loss))
        }
        
        model.backward(loss)
        model.update_weights(0.01)  // learning rate
    }
    
    print("Training completed!")
}

func main() {
    analyze_data("sample_data.csv")
}
```

### Run the Complete Example
```bash
# Create sample data file
echo "1.0,2.0,3.0,7.0" > sample_data.csv
echo "2.0,3.0,4.0,11.0" >> sample_data.csv
echo "3.0,4.0,5.0,15.0" >> sample_data.csv

# Compile and run
python src/neuro/main.py data_analysis.nr
./data_analysis
```

## Compilation Options

NEURO provides various compilation options for different needs:

### Development Mode
```bash
# Fast compilation for development
python src/neuro/main.py program.nr --dev

# Show compilation steps
python src/neuro/main.py program.nr --verbose

# Check syntax without compilation
python src/neuro/main.py program.nr --check-syntax
```

### Optimization Levels
```bash
# No optimization (fastest compilation)
python src/neuro/main.py program.nr -O0

# Basic optimization
python src/neuro/main.py program.nr -O1

# Standard optimization (default)
python src/neuro/main.py program.nr -O2

# Aggressive optimization (slowest compilation, fastest execution)
python src/neuro/main.py program.nr -O3
```

### Debug Information
```bash
# Include debug symbols
python src/neuro/main.py program.nr --debug

# Show intermediate representations
python src/neuro/main.py program.nr --emit-tokens    # Lexer output
python src/neuro/main.py program.nr --emit-ast       # Parse tree
python src/neuro/main.py program.nr --emit-llvm-ir   # LLVM IR
```

## Common Patterns

### Pattern Matching
```neuro
enum Color {
    Red,
    Green,
    Blue,
    RGB(int, int, int)
}

func describe_color(color: Color) -> String {
    match color {
        Red -> "Pure red",
        Green -> "Pure green", 
        Blue -> "Pure blue",
        RGB(r, g, b) -> "RGB(" + str(r) + ", " + str(g) + ", " + str(b) + ")"
    }
}
```

### Generic Functions
```neuro
func max<T>(a: T, b: T) -> T where T: Comparable {
    if a > b {
        return a
    } else {
        return b
    }
}

func main() {
    let int_max = max(10, 20)        // T = int
    let float_max = max(3.14, 2.71)  // T = float
    print("Max int: " + str(int_max))
    print("Max float: " + str(float_max))
}
```

## Next Steps

Now that you've covered the basics:

1. **[Explore Language Features](../language/basics.md)** - Dive deeper into NEURO syntax
2. **[Learn AI Features](../language/ai-features.md)** - Master tensors and neural networks
3. **[Try More Examples](../examples/README.md)** - Practice with real-world examples
4. **[Set Up Your IDE](development-environment.md)** - Configure syntax highlighting and debugging

## Getting Help

- **Documentation**: Browse the [complete documentation](../README.md)
- **Examples**: Check out more [code examples](../examples/README.md)
- **Community**: Join our [Discord server](https://discord.gg/neuro-lang)
- **Issues**: Report problems on [GitHub](https://github.com/your-org/neuro/issues)

Happy coding with NEURO! 🚀 