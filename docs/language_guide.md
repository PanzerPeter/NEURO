# NEURO Language Guide

## Language Overview

NEURO is a domain-specific language designed for neural network development. It combines the power of PyTorch with an intuitive, declarative syntax that makes deep learning more accessible.

## Syntax Fundamentals

### Neural Network Definition

```neuro
model = NeuralNetwork(input_size=784, output_size=10) {
    Dense(units=128, activation="relu");
    Dropout(rate=0.2);
    Dense(units=10, activation="softmax");
}
```

### Layer Types

1. **Dense (Fully Connected)**
   ```neuro
   Dense(units=64, activation="relu");
   ```

2. **Convolutional**
   ```neuro
   Conv2D(filters=32, kernel_size=3, activation="relu");
   MaxPool(pool_size=2);
   ```

3. **Recurrent**
   ```neuro
   LSTM(units=64, return_sequences=true);
   GRU(units=32);
   ```

4. **Utility Layers**
   ```neuro
   Dropout(rate=0.5);
   BatchNorm();
   Flatten();
   ```

### Custom Layers

```neuro
@custom_layer
def ResidualBlock(x) {
    skip = x;
    Dense(units=64, activation="relu");
    Dense(units=64);
    return x + skip;
}
```

### Model Training

```neuro
# Configure training
loss = Loss(type="categorical_crossentropy");
optimizer = Optimizer(type="adam", learning_rate=0.001);

# Train the model
model.train(
    data=training_data,
    epochs=10,
    batch_size=32,
    validation_data=val_data
);
```

### Data Handling

```neuro
# Load data
data = load_matrix("dataset.csv");

# Preprocess
data.normalize();
train_data, val_data = data.split(0.8);  # 80-20 split

# Create batches
batches = data.create_batches(batch_size=32);
```

### Transfer Learning

```neuro
@pretrained("resnet18")
backbone = Backbone(trainable=false);

model = NeuralNetwork(input_size=224, output_size=10) {
    backbone(x);
    Flatten();
    Dense(units=10, activation="softmax");
}
```

### Model Branching

```neuro
model = NeuralNetwork(input_size=784, output_size=10) {
    # Core layers
    x = Dense(units=128, activation="relu")(x);
    
    # Branch 1: Classification
    branch1 = Dense(units=64, activation="relu")(x);
    output1 = Dense(units=10, activation="softmax")(branch1);
    
    # Branch 2: Reconstruction
    branch2 = Dense(units=256, activation="relu")(x);
    output2 = Dense(units=784, activation="sigmoid")(branch2);
    
    return [output1, output2];
}
```

## Variables and Scope

NEURO uses a block-based scope system, similar to many modern programming languages. Variables are accessible within their defining block and any nested blocks.

```neuro
x = 10;  # Global scope

def function() {
    y = 20;  # Function scope
    print(x);  # Can access global variable
    
    if (true) {
        z = 30;  # Block scope
        print(y);  # Can access function variable
    }
    
    print(z);  # Can access variables defined in inner blocks
}
```

### Constants and Variable Reassignment

```neuro
PI = 3.14159;  # Convention for constants is UPPERCASE
learning_rate = 0.01;  # Variables use snake_case

# Variables can be reassigned
learning_rate = 0.001;
```

## Control Flow

### Conditionals

```neuro
if (condition) {
    # Code to execute if condition is true
} else if (another_condition) {
    # Code to execute if another_condition is true
} else {
    # Code to execute if all conditions are false
}
```

### Loops

```neuro
# For loop
for (i in range(10)) {
    print(i);
}

# Iterating over collections
for (item in collection) {
    print(item);
}

# Loop control
for (i in range(100)) {
    if (i < 10) {
        continue;  # Skip to next iteration
    }
    if (i > 90) {
        break;  # Exit loop
    }
    print(i);
}
```

## Functions

```neuro
def calculate_accuracy(predictions, targets) {
    correct = 0;
    for (i in range(len(predictions))) {
        if (predictions[i] == targets[i]) {
            correct += 1;
        }
    }
    return correct / len(predictions);
}
```

### Decorators

NEURO supports decorators for adding functionality to functions:

```neuro
@timer
def train_model(model, data, epochs) {
    # Function implementation
}
```

Common built-in decorators:
- `@custom_layer`: Define a custom neural network layer
- `@pretrained`: Use a pre-trained model
- `@timer`: Measure execution time
- `@gpu`: Force execution on GPU

## Error Handling

NEURO provides detailed error messages for debugging:

```
Error: Invalid layer configuration at line 5, column 10
Dense(units=-1, activation="relu");
         ^
Details: 'units' must be a positive integer
```

### Handling Syntax Errors

When a syntax error occurs, NEURO will:
1. Highlight the problematic code
2. Indicate the exact position of the error
3. Provide a description of the issue
4. Suggest potential fixes

### Runtime Errors

Runtime errors include:
- `NeuroRuntimeError`: General runtime errors
- `NeuroTypeError`: Type-related errors
- `NeuroValueError`: Invalid value errors
- `NeuroShapeError`: Tensor shape mismatch errors
- `NeuroLayerError`: Neural network layer errors

## PyTorch Integration

NEURO is built on top of PyTorch and can integrate directly with PyTorch components. 

```neuro
# Direct access to PyTorch tensors
x = torch.randn(10, 10);

# Using PyTorch functions
y = torch.nn.functional.softmax(x, dim=1);
```

For more details, see the [PyTorch Integration Guide](pytorch_integration.md).

## File Extensions

NEURO uses the following file extensions:
- `.nr`: NEURO source code files
- `.nrm`: NEURO Matrix data format files

## Comments

NEURO supports single-line and multi-line comments:

```neuro
# This is a single-line comment

/*
  This is a
  multi-line comment
*/
```

## Best Practices

For coding standards and best practices, refer to the [Best Practices Guide](best_practices.md). 