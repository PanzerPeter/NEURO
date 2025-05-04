# examples/data/load_split.nr
# Demonstrates loading data using load_matrix and splitting it.

print("Loading data...");
# Assumes this script is run from the project root
data_path = "examples/data/dummy_binary_classification.nrm";
full_data = load_matrix(data_path);

print("Data loaded:", full_data);

print("Splitting data (70% train, 15% val, 15% test)...");
# Use tuple assignment to capture the three returned matrices
# Note: If a fraction is 0, the corresponding variable will be None.
train_set, validation_set, test_set = full_data.split(
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    shuffle=True,      # Shuffle before splitting
    random_state=42    # For reproducible split
);

print("Train set:", train_set);
print("Validation set:", validation_set);
print("Test set:", test_set);

# Example of accessing data length (if needed)
# train_len = len(train_set); # Requires len() built-in or method
# print("Number of training samples:", train_len); 