# examples/training/train_mlp.nr
# Full example: Load data, define MLP, train, evaluate, save.

print("--- MLP Training Example ---");

# 1. Load Data
print("Loading data...");
data_path = "examples/data/dummy_binary_classification.nrm";
full_data = load_matrix(data_path);
print("Data loaded:", full_data);

# 2. Define Model
print("Defining MLP model...");
mlp_model = NeuralNetwork(input_size=4, output_size=1) {
    Dense(units=16, activation="relu");
    Dense(units=8, activation="relu");
    Dense(units=1, activation="sigmoid");
};
print("Model defined:", mlp_model);

# 3. Configure Loss and Optimizer
print("Configuring loss (BCE) and optimizer (Adam)...");
loss_config = Loss(type="bce");
opt_config = Optimizer(type="adam", learning_rate=0.005);

# 4. Train Model
print("Starting training...");
# The interpreter uses the last defined loss/optimizer config
history = mlp_model.train(
    data=full_data, 
    epochs=20, 
    batch_size=4
    # validation_data=... # Could split data and provide validation set here
);
print("Training finished.");
print("Training history:", history);

# 5. Evaluate Model (Placeholder)
print("Evaluating model (placeholder)...");
# Evaluation currently returns dummy data in the interpreter
# In a real scenario, load a separate test set:
# test_data = load_matrix("examples/data/dummy_binary_test.nrm");
# results = mlp_model.evaluate(data=test_data);
results = mlp_model.evaluate(data=full_data); # Evaluate on training data for demo
print("Evaluation results (placeholder):", results);

# 6. Save Model
print("Saving model...");
save_path = "./mlp_model_trained.pt"; # Save in the current directory
mlp_model.save(save_path);
print("Model saved to:", save_path);

print("--- MLP Training Example Complete ---"); 