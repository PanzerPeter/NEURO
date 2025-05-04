# examples/training/train_cnn_with_validation.nr
# Full example: Load data, split, define CNN, train with validation, evaluate.

print("--- CNN Training Example with Validation ---");

# 1. Load Data
print("Loading data...");
data_path = "examples/data/dummy_multiclass_images.nrm";
full_data = load_matrix(data_path);
print("Data loaded:", full_data);

# 2. Split Data
print("Splitting data (60% train, 20% val, 20% test)...");
train_set, val_set, test_set = full_data.split(
    train_frac=0.6, 
    val_frac=0.2, 
    test_frac=0.2, 
    random_state=123
);
print("Train set:", train_set);
print("Validation set:", val_set);
print("Test set:", test_set);

# Handle cases where splits might be empty (e.g., small dataset)
if train_set == None or val_set == None:
    print("Error: Dataset too small to create non-empty train/validation splits with current fractions.");
    # Exit or handle error appropriately - requires language support for control flow or exit()
    # For now, the script might fail later if train_set is None.

# 3. Define Model
print("Defining CNN model...");
cnn_model = NeuralNetwork(input_size=[1, 8, 8], output_size=3) {
    Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1, activation="relu");
    BatchNorm(); 
    Conv2D(out_channels=16, kernel_size=3, padding=1, activation="relu"); 
    BatchNorm();
    Flatten(); 
    Dense(units=32, activation="relu");
    BatchNorm();
    Dense(units=3, activation="softmax"); # Output layer for 3 classes
};
print("Model defined:", cnn_model);

# 4. Configure Loss and Optimizer
print("Configuring loss (CrossEntropy) and optimizer (Adam)...");
# Use CrossEntropyLoss for multi-class classification with softmax output
loss_config = Loss(type="crossentropy"); 
opt_config = Optimizer(type="adam", learning_rate=0.002);

# 5. Train Model with Validation
print("Starting training with validation...");
history = cnn_model.train(
    data=train_set,             # Use the training split
    epochs=15, 
    batch_size=2,
    validation_data=val_set     # Provide the validation split
);
print("Training finished.");
print("Training history:", history);

# 6. Evaluate Model (Placeholder)
print("Evaluating model on test set (placeholder)...");
if test_set != None:
    results = cnn_model.evaluate(data=test_set);
    print("Evaluation results (placeholder):", results);
else:
    print("Skipping evaluation as test set is empty.");

# 7. Save Model
print("Saving model...");
save_path = "./cnn_model_trained.pt";
cnn_model.save(save_path);
print("Model saved to:", save_path);

print("--- CNN Training Example Complete ---"); 