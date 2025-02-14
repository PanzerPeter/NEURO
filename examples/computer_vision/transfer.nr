# Transfer Learning Example
# This example demonstrates transfer learning using a pre-trained ResNet model

# 1. Data Loading and Preprocessing
data = load_dataset("custom_images", path="data/flower_photos");

# Image preprocessing pipeline
data.resize(size=(224, 224));
data.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]);  # ImageNet stats

# Data augmentation for training
train_augmentation = {
    "rotation_range": 30,
    "width_shift": 0.2,
    "height_shift": 0.2,
    "horizontal_flip": true,
    "vertical_flip": false,
    "zoom_range": 0.2,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest"
};

# Split and prepare data
train_data, temp = data.split(0.8);
val_data, test_data = temp.split(0.5);
train_data.enable_augmentation(train_augmentation);

# 2. Model Definition with Transfer Learning
@pretrained("resnet50")
backbone = Backbone(weights="imagenet", trainable=false);

model = NeuralNetwork(input_size=(224, 224, 3), output_size=5) {
    # Pre-trained backbone
    backbone(x);
    
    # Custom classification head
    GlobalAveragePooling2D();
    Dropout(rate=0.5);
    Dense(units=512, activation="relu");
    BatchNorm();
    Dropout(rate=0.3);
    Dense(units=5, activation="softmax");
}

# 3. Training Strategy
training_strategy = {
    "phase1": {  # Train only the custom head
        "epochs": 10,
        "learning_rate": 0.001,
        "trainable_layers": ["dense", "batch_norm"]
    },
    "phase2": {  # Fine-tune last few backbone layers
        "epochs": 20,
        "learning_rate": 0.0001,
        "unfreeze_layers": ["layer4", "layer3"]
    },
    "phase3": {  # Fine-tune entire network
        "epochs": 30,
        "learning_rate": 0.00001,
        "unfreeze_all": true
    }
};

# 4. Training Configuration
base_config = {
    "batch_size": 32,
    "optimizer": {
        "type": "adam",
        "beta1": 0.9,
        "beta2": 0.999
    },
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy", "top_k_accuracy"]
};

# 5. Custom Callbacks
@before_training
def prepare_phase(model, phase) {
    if phase == "phase1" {
        model.freeze_backbone();
    } elif phase == "phase2" {
        model.unfreeze_layers(["layer4", "layer3"]);
    } elif phase == "phase3" {
        model.unfreeze_all();
    }
    
    model.enable_mixed_precision();
}

@after_epoch
def monitor_training(model, metrics) {
    # Log metrics
    print(f"Epoch {metrics.epoch} ({current_phase}):");
    print(f"  Accuracy: {metrics.accuracy:.4f}");
    print(f"  Val Accuracy: {metrics.val_accuracy:.4f}");
    
    # Save best model for each phase
    if metrics.val_accuracy > best_accuracy[current_phase] {
        model.save_checkpoint(f"models/best_{current_phase}.pt");
        print(f"  Saved best model for {current_phase}");
    }
}

# 6. Training Execution
print("Starting transfer learning...");

for phase, config in training_strategy.items() {
    current_phase = phase;
    print(f"\nStarting {phase}...");
    
    # Update configuration
    phase_config = base_config.copy();
    phase_config["learning_rate"] = config["learning_rate"];
    phase_config["epochs"] = config["epochs"];
    
    # Train for this phase
    history = model.train(
        data=train_data,
        validation_data=val_data,
        config=phase_config
    );
    
    # Save phase results
    model.save(f"models/{phase}_model.pt");
    save_history(history, f"history/{phase}_history.json");
}

# 7. Final Evaluation
test_metrics = model.evaluate(test_data);
print("\nFinal Test Set Performance:");
print(f"Accuracy: {test_metrics.accuracy:.4f}");
print(f"Top-5 Accuracy: {test_metrics.top_k_accuracy:.4f}");

# 8. Model Analysis
model.analyze_feature_maps(test_data[0]);
model.visualize_class_activation_maps(test_data[0:5]);
model.plot_confusion_matrix(test_data);

# 9. Save Final Model
model.save("models/final_transfer_model.pt", format="torchscript");

# 10. Visualization
plots = MultiPlot(rows=2, cols=2) {
    # Training history across phases
    subplot(0, 0) {
        for phase in ["phase1", "phase2", "phase3"] {
            add_line(histories[phase].accuracy, 
                    label=f"{phase} Accuracy");
        }
        set_title("Training Accuracy by Phase");
    }
    
    # Learning rates across phases
    subplot(0, 1) {
        for phase in ["phase1", "phase2", "phase3"] {
            add_line(histories[phase].learning_rate,
                    label=f"{phase} LR");
        }
        set_title("Learning Rate by Phase");
    }
    
    # Sample predictions
    subplot(1, 0) {
        add_prediction_grid(test_data, samples=16);
        set_title("Sample Predictions");
    }
    
    # Class activation maps
    subplot(1, 1) {
        add_activation_maps(test_data[0:4]);
        set_title("Class Activation Maps");
    }
};
plots.save("plots/transfer_learning_analysis.png");

# 11. Export Model Report
model.generate_report({
    "architecture": true,
    "performance_metrics": true,
    "training_history": true,
    "sample_predictions": true,
    "export_format": "pdf"
}); 