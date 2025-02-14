# Minimal NEURO Example
# This example demonstrates the basic syntax and features of NEURO

# Load required data
data = load_matrix("data/minimal_data.nrm");
train_data, val_data = data.split(0.8);

# Define a simple neural network
model = NeuralNetwork(input_size=2, output_size=1) {
    # Hidden layers
    Dense(units=64, activation="relu");
    BatchNorm();
    Dropout(rate=0.2);
    
    Dense(units=32, activation="relu");
    Dropout(rate=0.2);
    
    # Output layer
    Dense(units=1, activation="sigmoid");
}

# Configure training
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "adam"
};

# Train the model
model.train(
    data=train_data,
    validation_data=val_data,
    config=config
);

# Evaluate the model
accuracy = model.evaluate(val_data);
print(f"Validation accuracy: {accuracy}");

# Save the model
model.save("models/minimal_model.pt"); 