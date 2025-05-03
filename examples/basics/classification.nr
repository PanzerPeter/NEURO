# Binary Classification Example
# This example demonstrates how to build a binary classifier using NEURO

# 1. Data Loading and Preprocessing
data = load_matrix("data/heart_disease.nrm");

# Preprocess the data
data.normalize();
data.handle_missing(strategy="mean");
data.shuffle(seed=42);

# Split the data
train_data, temp = data.split(0.7);
val_data, test_data = temp.split(0.5);

# 2. Model Definition
model = NeuralNetwork(input_size=13, output_size=1) {
    # Feature extraction layers
    Dense(units=128, activation="relu");
    BatchNorm();
    Dropout(rate=0.3);
    
    Dense(units=64, activation="relu");
    BatchNorm();
    Dropout(rate=0.2);
    
    # Output layer for binary classification
    Dense(units=1, activation="sigmoid");
}

# 3. Training Configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": {
        "type": "adam",
        "beta1": 0.9,
        "beta2": 0.999
    },
    "loss": "binary_crossentropy",
    "metrics": ["accuracy", "precision", "recall", "f1"]
};

# 4. Callbacks
@before_training
def prepare_data(data) {
    return data.balance(strategy="class_weight");
}

@after_epoch
def check_metrics(model, metrics) {
    # Log metrics
    print(f"Epoch {metrics.epoch}:");
    print(f"  Loss: {metrics.loss:.4f}");
    print(f"  Accuracy: {metrics.accuracy:.4f}");
    
    # Save best model
    if metrics.val_accuracy > best_accuracy {
        model.save_checkpoint("models/best_classifier.pt");
    }
    
    # Early stopping
    if not improved_for(10) {
        return "StopTraining";
    }
}

# 5. Train the model
print("Starting training...");
history = model.train(
    data=train_data,
    validation_data=val_data,
    config=config
);

# 6. Evaluate on test set
test_metrics = model.evaluate(test_data);
print("\nTest Set Performance:");
print(f"Accuracy: {test_metrics.accuracy:.4f}");
print(f"Precision: {test_metrics.precision:.4f}");
print(f"Recall: {test_metrics.recall:.4f}");
print(f"F1 Score: {test_metrics.f1:.4f}");

# 7. Save the model
model.save("models/final_classifier.pt");

# 8. Visualize Results
plot = Plot() {
    add_line(history.accuracy, label="Training Accuracy");
    add_line(history.val_accuracy, label="Validation Accuracy");
    set_title("Training History");
    set_xlabel("Epoch");
    set_ylabel("Accuracy");
};
plot.save("plots/training_history.png"); 