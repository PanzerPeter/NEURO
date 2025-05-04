# NEURO Language Guide (Alpha)

This guide describes the syntax and core features of the NEURO language as currently implemented.

## Basic Structure

A NEURO program (`.nr` file) consists of a sequence of statements. Most statements end with a semicolon `;`.

```neuro
// Comments start with #
# This is a comment

// Variable assignment
my_variable = 10;

// Function call
print("Hello, NEURO!");

// Model definition (ends with })
my_model = NeuralNetwork(input_size=10, output_size=1) {
    Dense(units=64, activation="relu");
    Dense(units=1, activation="sigmoid");
};

// Loss/Optimizer configuration
loss_config = Loss(type="bce");
opt_config = Optimizer(type="adam", learning_rate=0.001);

// Model training call (ends with ;)
my_model.train(data=my_data, epochs=10);
```

## Comments

Single-line comments start with `#`.

```neuro
# This entire line is a comment
variable = 1; # Comment after code
```

## Variables and Assignment

Variables are assigned using `=`. They can hold numbers, strings, loaded data (`NeuroMatrix`), model definitions, or results from function calls.

```neuro
count = 100;
message = "Processing...";
data = load_matrix("data.nrm");

# Tuple assignment is supported for functions returning sequences (like data.split)
data_matrix = load_matrix("my_data.nrm");
train_set, test_set = data_matrix.split(train_frac=0.75);
```

## Data Types

- **Number**: Integers (`10`) and floating-point numbers (`3.14`).
- **String**: Double-quoted strings (`"hello"`).
- **NeuroMatrix**: Represents datasets, loaded via `load_matrix()`. 
- **NeuralNetwork**: Represents a defined model.
- **None**: The special value `None` (used implicitly for Loss/Optimizer assignments).

## Built-in Functions

- `print(...)`: Prints its arguments to the console.
- `load_matrix("filepath.nrm")`: Loads data from a `.nrm` (YAML) file into a `NeuroMatrix` object.

```neuro
x = 10;
print("Value of x:", x);

data = load_matrix("../datasets/iris.nrm");
print("Loaded data:", data); # Note: Printing a matrix might show limited info
```

## Model Definition

Models are defined using the `NeuralNetwork` keyword, assigned to a variable. The definition includes optional parameters in parentheses `()` and a body in curly braces `{}` containing layer definitions.

```neuro
# input_size/output_size are currently informational metadata
my_cnn = NeuralNetwork(input_size=784, output_size=10) {
    # Layers are defined sequentially
    # Each layer definition must end with a semicolon
    Conv2D(out_channels=16, kernel_size=3, padding=1, activation="relu"); # Assumes in_channels if follows another Conv2D
    BatchNorm(); # Infers dim=2 after Conv2D
    # MaxPool(); # MaxPool not yet implemented
    Conv2D(out_channels=32, kernel_size=3, padding=1, activation="relu");
    BatchNorm(); # Infers dim=2 after Conv2D
    Flatten();
    Dense(units=128, activation="relu");
    BatchNorm(); # Infers dim=1 after Dense
    Dense(units=10, activation="softmax");
};
```

### Model Parameters (`NeuralNetwork(...)`)

Currently, only `input_size` and `output_size` are recognized as informational parameters within the `NeuralNetwork(...)` definition. They do not affect the PyTorch model construction directly at this time.

### Layers

Layers are defined within the model body `{...}`. Each layer definition must end with a semicolon `;`. Activations are typically specified within the layer's parameters.

- **`Dense(units=..., activation=...)`**: Fully connected layer.
  - `units`: (Required) Number of output neurons.
  - `activation`: (Optional) String name of activation function (e.g., "relu", "sigmoid").

- **`Conv2D(...)`**: 2D Convolutional layer.
  - `out_channels`: (Required) Number of output channels (filters).
  - `kernel_size`: (Required) Integer or tuple (e.g., `3` or `(3, 3)`).
  - `in_channels`: (Optional) Number of input channels. If omitted, it's inferred from the previous `Conv2D` layer's `out_channels`. **Required for the first Conv2D layer.**
  - `stride`: (Optional) Integer or tuple (default: `1`).
  - `padding`: (Optional) Integer or tuple (default: `0`).
  - `activation`: (Optional) String name of activation function.

- **`BatchNorm(...)`**: Batch Normalization.
  - `momentum`: (Optional) Momentum factor (default: `0.1`).
  - `eps`: (Optional) Epsilon value (default: `1e-5`).
  - `dim`: (Optional) Explicitly specify `1` or `2`. If omitted, dimension (1D or 2D) is inferred based on the preceding layer (`Dense` -> 1D, `Conv2D` -> 2D). Requires `in_channels` to be set for `dim=2` (usually inferred from preceding `Conv2D`). Uses `LazyBatchNorm1d` or `BatchNorm2d` internally.

- **`Flatten(...)`**: Flattens input, typically used between convolutional and dense layers.
  - `start_dim`: (Optional) First dimension to flatten (default: `1`).
  - `end_dim`: (Optional) Last dimension to flatten (default: `-1`).

- **`Dropout(...)`**: (Not shown in example, but likely supported if in `src/models.py` layer parsing logic)
  - `rate`: (Required) Dropout probability.

### Activation Functions

Specify activation as a string parameter (`activation="..."`) within layer definitions. Supported: `"relu"`, `"sigmoid"`, `"tanh"`, `"softmax"`. For `softmax`, an optional `dim` parameter can be added (e.g., `Dense(..., activation="softmax", dim=1)`).

## Loss and Optimizer Configuration

Loss function and optimizer are configured globally for subsequent training calls by assigning the result of `Loss(...)` or `Optimizer(...)` calls to variables. The interpreter uses the *most recently defined* configurations when `model.train()` is called.

```neuro
# The variable names ('loss_cfg', 'opt_cfg') don't matter to the interpreter,
# only that Loss() and Optimizer() were called.
l_cfg = Loss(type="mse"); # Configure Mean Squared Error loss
o_cfg = Optimizer(type="sgd", learning_rate=0.01, momentum=0.9); # Configure SGD

# ... define model ...
# ... load data ...

# This training run will use MSE loss and SGD optimizer
model.train(data=train_data, epochs=20);

# Reconfigure for the next training run
new_loss = Loss(type="crossentropy");
new_opt = Optimizer(type="adam", learning_rate=0.0005);

# This training run will use CrossEntropy loss and Adam optimizer
model.train(data=train_data, epochs=10);
```

### `Loss(...)` Parameters

- `type`: (Required) String name of the loss function. Supported: `"bce"` (Binary Cross Entropy), `"mse"` (Mean Squared Error), `"crossentropy"` (Cross Entropy Loss).

### `Optimizer(...)` Parameters

- `type`: (Required) String name of the optimizer. Supported: `"adam"`, `"sgd"`, `"rmsprop"`.
- `learning_rate`: (Optional) Learning rate (default depends on optimizer, e.g., `0.001` for Adam).
- Other optimizer-specific parameters (e.g., `betas`, `eps`, `weight_decay` for Adam; `momentum` for SGD) might be supported by the backend but may need explicit implementation in the parser/interpreter logic if not standard `torch.optim` arguments.

## Model Training

Train a defined model using the `.train()` method.

```neuro
# Assumes 'my_model' is a defined NeuralNetwork variable
# Assumes 'train_data' is a loaded NeuroMatrix variable
# Assumes Loss() and Optimizer() have been configured previously

history = my_model.train(
    data=train_data, 
    epochs=100, 
    batch_size=32, 
    validation_data=val_data # Optional: requires 'val_data' NeuroMatrix variable
);
```

### `train(...)` Parameters

- `data`: (Required) The `NeuroMatrix` variable containing training data.
- `epochs`: (Optional) Number of training epochs (default: `10`).
- `batch_size`: (Optional) Mini-batch size (default: `32`).
- `validation_data`: (Optional) A `NeuroMatrix` variable containing validation data.

## Model Evaluation

Evaluate a trained model using the `.evaluate()` method. *Note: The current implementation might be a placeholder.*

```neuro
# Assumes 'my_model' is trained
# Assumes 'test_data' is a loaded NeuroMatrix

results = my_model.evaluate(data=test_data);
print("Evaluation results:", results);
```

### `evaluate(...)` Parameters

- `data`: (Required) The `NeuroMatrix` variable containing evaluation data.

## Model Saving and Loading

Save model weights and basic metadata using `.save()`. Load using `NeuralNetwork.load()` (static method).

```neuro
# Saving
my_model.save("my_cnn_model.pt");

# Loading (Requires manual re-definition of the *exact same* architecture first)
reloaded_model_architecture = NeuralNetwork(...) { ... }; # Define same layers as saved model

# This static load method is not yet fully functional for reconstruction
# loaded_model = NeuralNetwork.load("my_cnn_model.pt"); # This likely raises NotImplementedError

# Workaround: Load state dict into the manually created architecture
# (This functionality needs to be exposed/implemented in the language/interpreter)
# --> currently no direct language feature for state_dict loading.
```

- **Saving:** `.save("filepath.pt")` saves the model's state dictionary.
- **Loading:** `NeuralNetwork.load("filepath.pt")` is intended but currently raises a `NotImplementedError` because reconstructing the model architecture from the file is not supported. The current workaround involves defining the model again in the script and manually loading the state dictionary using PyTorch methods (which isn't directly possible *within* the NEURO language yet).

