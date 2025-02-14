# MNIST Classification Example
# This example demonstrates how to build a CNN for MNIST digit classification

# 1. Data Loading and Preprocessing
mnist = load_dataset("mnist");
mnist.normalize(scale=255.0);

# Split into train, validation, and test sets
train_data, val_data = mnist.train.split(0.9);
test_data = mnist.test;

# Enable data augmentation for training
train_data.enable_augmentation({
    "rotation_range": 10,
    "width_shift": 0.1,
    "height_shift": 0.1,
    "zoom_range": 0.1
});

# 2. Model Definition
model = NeuralNetwork(input_size=(28, 28, 1), output_size=10) {
    # First convolutional block
    Conv2D(filters=32, kernel_size=3, padding="same");
    BatchNorm();
    ReLU();
    Conv2D(filters=32, kernel_size=3, padding="same");
    BatchNorm();
    ReLU();
    MaxPool(pool_size=2);
    Dropout(rate=0.25);
    
    # Second convolutional block
    Conv2D(filters=64, kernel_size=3, padding="same");
    BatchNorm();
    ReLU();
    Conv2D(filters=64, kernel_size=3, padding="same");
    BatchNorm();
    ReLU();
    MaxPool(pool_size=2);
    Dropout(rate=0.25);
    
    # Classification head
    Flatten();
    Dense(units=512, activation="relu");
    BatchNorm();
    Dropout(rate=0.5);
    Dense(units=10, activation="softmax");
}

# 3. Training Configuration
config = {
    "learning_rate": {
        "initial": 0.001,
        "scheduler": "cosine_annealing",
        "min_lr": 1e-6,
        "cycles": 5
    },
    "batch_size": 128,
    "epochs": 50,
    "optimizer": {
        "type": "adam",
        "beta1": 0.9,
        "beta2": 0.999
    },
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy", "top_k_accuracy"]
};

# 4. Custom Callbacks
@before_training
def setup_training(model) {
    model.enable_mixed_precision();
    model.enable_checkpointing();
}

@after_epoch
def monitor_training(model, metrics) {
    # Log metrics
    print(f"Epoch {metrics.epoch}:");
    print(f"  Accuracy: {metrics.accuracy:.4f}");
    print(f"  Val Accuracy: {metrics.val_accuracy:.4f}");
    
    # Save best model
    if metrics.val_accuracy > best_accuracy {
        model.save_checkpoint("models/best_mnist.pt");
        print("  Saved new best model");
    }
    
    # Learning rate cycling
    if is_cycle_end(metrics.epoch) {
        print("  Starting new learning rate cycle");
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
print(f"Top-5 Accuracy: {test_metrics.top_k_accuracy:.4f}");

# 7. Confusion Matrix and Analysis
model.plot_confusion_matrix(test_data);
model.analyze_misclassifications(test_data, max_samples=10);

# 8. Save the model
model.save("models/final_mnist.pt");

# 9. Visualization
plots = MultiPlot(rows=2, cols=2) {
    # Training history
    subplot(0, 0) {
        add_line(history.accuracy, label="Training Accuracy");
        add_line(history.val_accuracy, label="Validation Accuracy");
        set_title("Accuracy History");
    }
    
    # Learning rate history
    subplot(0, 1) {
        add_line(history.learning_rate, label="Learning Rate");
        set_title("Learning Rate Schedule");
    }
    
    # Sample predictions
    subplot(1, 0) {
        add_prediction_grid(test_data, samples=16);
        set_title("Sample Predictions");
    }
    
    # Feature maps
    subplot(1, 1) {
        add_feature_maps(test_data[0], layer="conv2d_1");
        set_title("First Layer Feature Maps");
    }
};
plots.save("plots/mnist_analysis.png");

# 10. Model Summary and Statistics
model.summary();
model.compute_flops();
model.measure_inference_time(batch_size=1);
model.measure_inference_time(batch_size=32); 