# NEURO Language Guide (Alpha)

This guide describes the syntax and core features of the NEURO language as currently implemented.

## Basic Structure

A NEURO program (`.nr` file) consists of a sequence of statements. Most statements end with a semicolon `;`.

```neuro
// Comments start with //
// This is a comment

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

// Data Loading
matrix_data = load_matrix("my_data.nrm");

// Data Splitting
train_set, val_set, test_set = matrix_data.split(train_frac=0.7, val_frac=0.15, test_frac=0.15);

// Model training call (ends with ;)
my_model.train(data=train_set, epochs=10, validation_data=val_set);

// Model saving
my_model.save("saved_model.pth");
```

## Comments

Single-line comments start with `//`.

```neuro
// This entire line is a comment
variable = 1; // Comment after code
```

## Variables and Assignment

Variables are assigned using `=`. They can hold numbers, strings, loaded data (`NeuroMatrix`), model definitions, or results from function calls.

```neuro
count = 100;
message = "Processing...";
data = load_matrix("data.nrm");

# Tuple assignment is supported for functions/methods returning sequences
data_matrix = load_matrix("my_data.nrm");
# The split method returns multiple values
train_set, val_set, test_set = data_matrix.split(train_frac=0.7, val_frac=0.15); # Test frac is inferred
print("Training set size:", train_set); # Assumes NeuroMatrix has __str__ or __len__

# General tuple assignment
a, b = some_function_returning_two_values();
```

## Data Types

- **Number**: Integers (`10`) and floating-point numbers (`3.14`).
- **String**: Double-quoted strings (`"hello"`).
- **NeuroMatrix**: Represents datasets, loaded via `load_matrix()` or returned by `split()`. It supports methods like `.split()`.
- **NeuralNetwork**: Represents a defined model. It supports methods like `.train()`, `.evaluate()`, and `.save()`.
- **None**: The special value `None` (currently used implicitly when assigning Loss/Optimizer configurations).
- **Internal Types**: The type system uses internal types like `IntType`, `FloatType`, `StringType`, `TensorType`, `LayerType`, `ModelType`, `LossType`, `OptimizerType`, `FunctionType`, `DataType`, `VoidType`, `AnyType` for static analysis. You don't typically interact with these directly, but error messages might refer to them.

## Built-in Functions

- `print(...)`: Prints its arguments to the console.
- `load_matrix("filepath.nrm")`: Loads data from a `.nrm` (YAML format) file into a `NeuroMatrix` object. See Data Format section.

```neuro
x = 10;
print("Value of x:", x);

data = load_matrix("../datasets/iris.nrm");
print("Loaded data:", data); # Note: Printing a matrix might show limited info
```

## Type System (Static Analysis)

NEURO includes a static type checker that runs automatically before your script is interpreted (when using the `--use-real` flag). Its goal is to catch potential errors early, such as incompatible layer connections, using variables before they are defined, or passing the wrong kind of data to functions or layers.

**Key Checks Performed:**

*   **Variable Usage:** Ensures variables are defined before use and tracks their inferred types (`Int`, `Float`, `String`, `ModelType`, `DataType`, etc.).
*   **Layer Compatibility:** When defining a `NeuralNetwork`, the type checker analyzes the sequence of layers:
    *   It infers the output shape and data type (typically `Float`) of each layer.
    *   It verifies that the output type/shape of one layer is compatible with the expected input type/shape of the next layer.
    *   It checks layer parameters (e.g., `units` must be an `Int`).
    *   Dimensionality is checked (e.g., `Conv2D` expects 4D tensor input, `Dense` expects 2D).
*   **Function/Method Calls:**
    *   Checks if the function or method exists for the given object type.
    *   Validates the number and types of arguments passed against the expected signature.
*   **Training Compatibility:** Before a `train` statement, it checks:
    *   If a `Loss` and `Optimizer` have been defined.
    *   If the model's output type is compatible with the loss function's expected input.
    *   (Future) If the model's input and loss target types are compatible with the training data.

**How it Works:**

The checker traverses the code structure (Abstract Syntax Tree) *without* running the code. It infers types based on assignments and definitions and uses pre-defined signatures for built-in functions, methods, layers, losses, and optimizers to validate usage.

**Example Errors Caught:**

```neuro
my_model = NeuralNetwork() {
    Dense(units=64); # Output: (Batch, 64)
    # ERROR: Conv2D expects 4D input, but Dense output is 2D.
    Conv2D(out_channels=16, kernel_size=3); 
};

x = "not a number";
# ERROR: Dense layer 'units' parameter expects Int, got String.
layer = Dense(units=x);

defined_model = NeuralNetwork() { Dense(units=1); };
# ERROR: Variable 'undefined_model' is not defined.
save(undefined_model, filepath="model.pth"); 

classify_model = NeuralNetwork() { Dense(units=10); }; # Output (Batch, 10)
Loss(type="BCE"); # BCE expects 1D output
Optimizer(type="Adam");
# ERROR: Model output Tensor(shape=(None, 10)) is not compatible
#        with loss expected input Tensor(shape=(None,)).
train(classify_model, data=some_data, epochs=5);
```

If the type checker finds errors, they will be printed after parsing, and the script execution will halt.

## Model Definition

Models are defined using the `NeuralNetwork` keyword, assigned to a variable. The definition includes optional parameters in parentheses `()` and a body in curly braces `{}` containing layer definitions.

```neuro
# input_size/output_size are currently informational metadata
my_cnn = NeuralNetwork(input_size=(1, 28, 28), output_size=10) { # Example input size for image data
    # Layers are defined sequentially
    # Each layer definition must end with a semicolon
    # First Conv2D requires in_channels
    Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1, activation="relu");
    BatchNorm(); # Infers dim=2, features=16 after Conv2D
    MaxPool2D(kernel_size=2, stride=2);
    Conv2D(out_channels=32, kernel_size=3, padding=1, activation="relu"); # Infers in_channels=16
    BatchNorm(); # Infers dim=2, features=32
    AvgPool2D(kernel_size=2, stride=2);
    Flatten();
    Dense(units=128, activation="relu");
    Dropout(rate=0.5);
    BatchNorm(); # Infers dim=1 after Dense (uses LazyBatchNorm1d)
    Dense(units=10, activation="softmax"); # Assuming classification task
};
```

### Model Parameters (`NeuralNetwork(...)`)

Currently, only `input_size` and `output_size` are recognized as informational parameters within the `NeuralNetwork(...)` definition. They do not affect the PyTorch model construction directly at this time.

### Layers

Layers are defined within the model body `{...}`. Each layer definition must end with a semicolon `;`. Activations are typically specified within the layer's parameters.

- **`Dense(units=..., activation=...)`**: Fully connected layer. Uses `LazyLinear` internally, so `input_shape` is not required.
  - `units`: (Required) Number of output neurons.
  - `activation`: (Optional) String name of activation function (e.g., `"relu"`, `"sigmoid"`).

- **`Conv2D(...)`**: 2D Convolutional layer.
  - `out_channels`: (Required) Number of output channels (filters).
  - `kernel_size`: (Required) Integer or comma-separated string/tuple (e.g., `3` or `"3,3"` or `(3, 3)`).
  - `in_channels`: (Optional) Number of input channels. If omitted, it's inferred from the previous `Conv2D` layer's `out_channels`. **Required for the first Conv2D layer.**
  - `stride`: (Optional) Integer or comma-separated string/tuple (default: `1`).
  - `padding`: (Optional) Integer or comma-separated string/tuple (default: `0`).
  - `activation`: (Optional) String name of activation function.

- **`BatchNorm(...)`**: Batch Normalization.
  - `momentum`: (Optional) Momentum factor (default: `0.1`).
  - `eps`: (Optional) Epsilon value (default: `1e-5`).
  - `dim`: (Optional) Explicitly specify `1` or `2`. If omitted, dimension (1D or 2D) is inferred based on the preceding layer (`Dense`/`Flatten` -> 1D, `Conv2D`/`Pool` -> 2D). Uses `LazyBatchNorm1d` for 1D and `BatchNorm2d` for 2D (infers features from previous layer).

- **`Flatten(...)`**: Flattens input, typically used between convolutional/pooling and dense layers.
  - `start_dim`: (Optional) First dimension to flatten (default: `1`, flattens channel, height, width).
  - `end_dim`: (Optional) Last dimension to flatten (default: `-1`, flattens all dims after `start_dim`).

- **`MaxPool2D(...)`**: Applies 2D max pooling.
  - `kernel_size`: (Required) Integer or comma-separated string/tuple.
  - `stride`: (Optional) Integer or comma-separated string/tuple (default: same as `kernel_size`).
  - `padding`: (Optional) Integer or comma-separated string/tuple (default: `0`).

- **`AvgPool2D(...)`**: Applies 2D average pooling.
   - `kernel_size`: (Required) Integer or comma-separated string/tuple.
   - `stride`: (Optional) Integer or comma-separated string/tuple (default: same as `kernel_size`).
   - `padding`: (Optional) Integer or comma-separated string/tuple (default: `0`).

- **`Dropout(...)`**: Applies dropout regularization.
  - `rate`: (Required) Dropout probability (e.g., `0.5`).

### Activation Functions

Specify activation as a string parameter (`activation="..."`) within layer definitions (`Dense`, `Conv2D`). Supported: `"relu"`, `"sigmoid"`, `"tanh"`, `"softmax"`.
For `softmax`, an optional `dim` parameter can be added to the layer definition (e.g., `Dense(..., activation="softmax", dim=1)`), defaulting to `dim=1`.

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
new_loss = Loss(type="bce"); # Only BCE is currently implemented
new_opt = Optimizer(type="adam", learning_rate=0.0005);

# This training run will use BCE loss and Adam optimizer
model.train(data=train_data, epochs=10);
```

### `Loss(...)` Parameters

- `type`: (Required) String name of the loss function. Supported: `"bce"` (Binary Cross Entropy).

### `Optimizer(...)` Parameters

- `type`: (Required) String name of the optimizer. Supported: `"adam"`.
- `learning_rate`: (Optional) Learning rate (default: `0.001` for Adam).
- `betas`: (Optional) Tuple for Adam beta parameters (default: `(0.9, 0.999)`). *Syntax for tuple params TBD.*
- `eps`: (Optional) Epsilon for Adam (default: `1e-8`).
- `weight_decay`: (Optional) Weight decay for Adam (default: `0`).

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

- `data`: (Required) The `NeuroMatrix` variable containing training data (`input` and `output` keys).
- `epochs`: (Optional) Number of training epochs (default: `10`).
- `batch_size`: (Optional) Mini-batch size (default: `32`).
- `validation_data`: (Optional) A `NeuroMatrix` variable containing validation data.

## Model Evaluation

Evaluate a trained model using the `.evaluate()` method. This computes and returns the loss on the provided dataset using the currently configured loss function.

```neuro
# Save model state dictionary and metadata
my_model.save("my_cnn_model.pth");

# Loading requires re-defining the *exact same* architecture first,
# then using the static load method to load weights and metadata.
# NOTE: Reconstructing the model purely from the saved file is not supported.

reloaded_model = NeuralNetwork(input_size=(1, 28, 28), output_size=10) {
    # Must exactly match the saved model's architecture
    Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1, activation="relu");
    BatchNorm();
    MaxPool2D(kernel_size=2, stride=2);
    Conv2D(out_channels=32, kernel_size=3, padding=1, activation="relu");
    BatchNorm();
    AvgPool2D(kernel_size=2, stride=2);
    Flatten();
    Dense(units=128, activation="relu");
    Dropout(rate=0.5);
    BatchNorm();
    Dense(units=10, activation="softmax");
};

# Load the saved state dictionary into the existing architecture
# This load function is defined in models.py
loaded_model_instance = NeuralNetwork.load("my_cnn_model.pth");

# Now 'loaded_model_instance' contains the architecture AND the loaded weights.
# You can use it for evaluation or further training.
eval_results = loaded_model_instance.evaluate(data=test_data);

```

- **Saving:** `.save("filepath.pth")` saves the model's state dictionary and associated metadata (like the Neuro language parameters used to create it, if available).
- **Loading:** `NeuralNetwork.load("filepath.pth")` attempts to load a saved model. It **requires** that you first define a `NeuralNetwork` variable in your script with the **exact same architecture** as the one that was saved. The `.load()` method then creates an instance of this architecture and loads the saved weights (`state_dict`) into it. It does **not** reconstruct the model from the file alone.

## Data Format (`.nrm` Files)

The `load_matrix("filepath.nrm")` function expects data to be in YAML format with two top-level keys:

- `metadata`: A dictionary containing information about the dataset (e.g., name, description, source).
- `data`: A list where each item is a dictionary containing at least two keys:
    - `input`: A list or nested list of numbers representing the input features for one sample.
    - `output`: A list or number representing the target/label for that sample.

Example `data.nrm`:

```yaml
metadata:
  name: "Simple XOR Dataset"
  description: "Dataset for the XOR problem"
  num_samples: 4
data:
  - input: [0.0, 0.0]
    output: [0.0]
  - input: [0.0, 1.0]
    output: [1.0]
  - input: [1.0, 0.0]
    output: [1.0]
  - input: [1.0, 1.0]
    output: [0.0]
```

## NeuroMatrix Methods

Variables holding `NeuroMatrix` objects (loaded via `load_matrix` or returned by `split`) have the following methods:

- **`.split(train_frac=0.7, val_frac=0.15, test_frac=0.15, shuffle=True, random_state=None)`**:
    - Splits the data into training, validation, and testing sets.
    - Returns a tuple of `(train_matrix, validation_matrix, test_matrix)`.
    - If a fraction is 0, the corresponding returned matrix will be `None`.
    - If `test_frac` is omitted, it's calculated as `1 - train_frac - val_frac`.
    - `shuffle`: Whether to shuffle data before splitting (default: `True`).
    - `random_state`: Optional integer seed for reproducible shuffles.

```neuro
full_data = load_matrix("dataset.nrm");

# Split 70% train, 30% test, shuffled
train_data, _, test_data = full_data.split(train_frac=0.7, val_frac=0.0, test_frac=0.3);

# Split 60% train, 20% val, 20% test, not shuffled
train_data, val_data, test_data = full_data.split(0.6, 0.2, 0.2, shuffle=False);

print("Train set:", train_data); # Prints info about the new NeuroMatrix object
```

## Data Formats

NEURO primarily uses the `.nrm` (NeuroMatrix) format for handling datasets. This format is based on YAML for readability and ease of editing.

### .nrm Format Specification

A `.nrm` file consists of two main top-level keys: `metadata` and `data`.

```yaml
metadata:
  name: My Dataset                  # Optional: Human-readable name
  description: "Dataset details..." # Optional: Description of the data
  version: 1.0                     # Optional: Version of the dataset format/contents
  missing_value_token: null        # Optional: How missing values appear in the data section (e.g., null, NA, "", -1). Defaults to `null`.
  # Optional but recommended: Describes the structure of input/output data
  features:
    input:
      - { name: feature1, type: numeric }   # List of input features
      - { name: feature2, type: categorical } # Type can be numeric, categorical, text, etc.
      # ... more input features
    output:
      - { name: target, type: numeric }     # List of output features
      # ... more output features
  # Other custom metadata keys can be added here

data:
  # List of data points (records/rows)
  - input: [value1_1, value1_2] # Must match the order and length of metadata.features.input
    output: [target1]           # Must match the order and length of metadata.features.output
  - input: [value2_1, null]     # Example using `missing_value_token` (here, null)
    output: [target2]
  - input: [value3_1, .nan]     # Example using standard YAML NaN
    output: [target3]
  # ... more data points
```

*   **`metadata`**: A dictionary containing information *about* the dataset.
    *   `name`, `description`, `version`: Self-explanatory identifiers.
    *   `missing_value_token`: Defines the representation of missing data within the `data` list. The `NeuroMatrix` loader will convert these tokens to an internal representation (currently `None`).
    *   `features`: A dictionary with `input` and `output` keys. Each holds a list of dictionaries, where each dictionary defines a feature's `name` and `type`. This metadata is crucial for applying type-specific preprocessing.
*   **`data`**: A list where each element is a dictionary containing `input` and `output` keys. The values are lists of the actual data, corresponding to the features defined in the metadata.
*   **NaN Handling**: Standard YAML NaN representations (`.nan`, `.NaN`, `.NAN`) are automatically parsed by the YAML loader as `float('nan')`. These are treated distinctly from the `missing_value_token`. Methods within `NeuroMatrix` (like `normalize`) are designed to handle both `None` (from the token) and `float('nan')` appropriately, typically by converting them to `numpy.nan` when performing numerical operations.

## Built-in Classes and Modules

NEURO provides built-in classes for common tasks.

### `NeuroMatrix`

This class is the primary way to load, handle, and preprocess tabular data stored in `.nrm` files.

**Loading Data:**

```neuro
# Load data from a file
my_data = NeuroMatrix()
my_data.load("path/to/your_data.nrm")

print(my_data) # Shows basic info
print("Number of samples:", len(my_data))
print("Input Features:", my_data.input_features)
```

**Preprocessing Methods:**

The `NeuroMatrix` class offers methods to preprocess the loaded data *in-place*.

*   **`normalize(method='minmax', columns=None, data_type='input')`**
    *   Normalizes numeric columns.
    *   `method`: `'minmax'` (scales to [0, 1]) or `'zscore'` (scales to zero mean, unit variance).
    *   `columns`: An optional list of specific column names (strings) to normalize. If `None` (default), all columns marked as `numeric` in the specified `data_type`'s features will be normalized.
    *   `data_type`: `'input'` (default) or `'output'` to specify which set of features to normalize.
    *   Example:
        ```neuro
        # Min-max scale all numeric input features
        my_data.normalize(method='minmax', data_type='input')

        # Z-score scale specific output feature 'target1'
        my_data.normalize(method='zscore', columns=['target1'], data_type='output')
        ```

*   **`handle_missing(strategy='mean', columns=None, data_type='input', value=None)`**
    *   Handles missing values (represented internally as `None` or `np.nan`).
    *   `strategy`:
        *   `'mean'`: Fills missing values in numeric columns with the column's mean.
        *   `'median'`: Fills missing values in numeric columns with the column's median.
        *   `'constant'`: Fills missing values with the value specified in the `value` argument.
        *   `'remove'`: Removes entire rows (data points) that contain missing values in any of the specified `columns`.
    *   `columns`: An optional list of specific column names (strings) to process. If `None` (default), all columns in the specified `data_type` are processed.
    *   `data_type`: `'input'` (default) or `'output'`.
    *   `value`: Required only when `strategy='constant'`. Specifies the value to fill missing entries with.
    *   Example:
        ```neuro
        # Fill missing values in input columns 'feature1' and 'feature3' with their respective means
        my_data.handle_missing(strategy='mean', columns=['feature1', 'feature3'], data_type='input')

        # Fill all missing output values with 0
        my_data.handle_missing(strategy='constant', value=0, data_type='output')

        # Remove rows with any missing data in the input features
        my_data.handle_missing(strategy='remove', data_type='input')
        ```

*   **`split(train_frac=0.7, val_frac=0.15, test_frac=0.15, shuffle=True, random_state=None)`**
    *   Splits the dataset into training, validation, and test sets.
    *   Returns a tuple of three `NeuroMatrix` objects (or `None` if a fraction is 0).
    *   See `test_matrix.py` for detailed usage examples.

