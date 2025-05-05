# examples/mlp/simple_mlp.nr
# Defines a simple Multi-Layer Perceptron (MLP), loads data, preprocesses, and prints info.

# --- Data Loading and Preprocessing ---
print("Loading and preprocessing data...");
data_path = "../data/dummy_binary_classification.nrm"; # Relative path

# 1. Load data
matrix = NeuroMatrix();
matrix.load(data_path);
print("Data loaded:", matrix);

# 2. Handle Missing Values (Example: Fill with mean)
# Assuming some features might have missing values (though the dummy doesn't)
matrix.handle_missing(strategy='mean', data_type='input'); 
print("After handling missing (mean):");
# Note: Accessing processed data directly isn't standard Neuro syntax yet
# We might need print methods on NeuroMatrix later. For now, rely on print messages.

# 3. Normalize Features (Example: Min-Max scaling)
matrix.normalize(method='minmax', data_type='input');
print("After normalizing inputs (minmax):");

# 4. Split data (optional step shown)
train_data, val_data, test_data = matrix.split(train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42);
print("Train data size:", len(train_data));
if val_data {
    print("Validation data size:", len(val_data));
}
if test_data {
    print("Test data size:", len(test_data));
}

# --- Model Definition ---
print("\nDefining a simple MLP model...");

# Define the model and assign it to a variable
# Assuming input size matches the number of features in the loaded data
# len(matrix.input_features) is not directly available in Neuro syntax yet
# Hardcoding size based on the dummy data (4 input features)
mlp_model = NeuralNetwork(input_size=4, output_size=1) {
    # Layer definitions require semicolons
    Dense(units=16, activation="relu");
    Dense(units=8, activation="relu");
    Dense(units=1, activation="sigmoid"); # For binary classification output
};

print("MLP Model defined:");
print(mlp_model); # Print the model structure

# Note: Training steps would follow here using the preprocessed train_data, val_data
print("\nExample finished. Training loop not implemented in this example."); 