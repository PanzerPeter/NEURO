# Sentiment Analysis Example
# This example demonstrates sentiment analysis using NEURO with a BERT-based model

# 1. Data Loading and Preprocessing
data = load_dataset("imdb_reviews");

# Text preprocessing pipeline
text_processor = TextProcessor() {
    lowercase();
    remove_special_chars();
    truncate_length(max_length=512);
};

# Apply preprocessing
data.preprocess(text_processor);
data.tokenize(tokenizer="bert-base-uncased", max_length=512);

# Split datasets
train_data, temp = data.split(0.8);
val_data, test_data = temp.split(0.5);

# 2. Model Definition with Pre-trained BERT
@pretrained("bert-base-uncased")
bert = BERTBackbone(trainable=false);

model = NeuralNetwork(input_size=(512,), output_size=2) {
    # BERT Encoder
    bert(x);
    
    # Pooling and classification head
    GlobalAveragePooling1D();
    Dropout(rate=0.2);
    
    Dense(units=256, activation="relu");
    BatchNorm();
    Dropout(rate=0.2);
    
    Dense(units=64, activation="relu");
    Dropout(rate=0.1);
    
    Dense(units=2, activation="softmax");
}

# 3. Training Strategy
training_strategy = {
    "phase1": {  # Train only classification head
        "epochs": 3,
        "learning_rate": 2e-4,
        "trainable_layers": ["dense", "batch_norm"]
    },
    "phase2": {  # Fine-tune last BERT layers
        "epochs": 2,
        "learning_rate": 2e-5,
        "unfreeze_layers": ["encoder.layer.11", "encoder.layer.10"]
    }
};

# 4. Training Configuration
base_config = {
    "batch_size": 16,
    "optimizer": {
        "type": "adamw",
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "eps": 1e-8
    },
    "loss": {
        "type": "cross_entropy",
        "label_smoothing": 0.1
    },
    "metrics": ["accuracy", "precision", "recall", "f1"]
};

# 5. Custom Callbacks
@before_training
def prepare_phase(model, phase) {
    if phase == "phase1" {
        model.freeze_backbone();
    } elif phase == "phase2" {
        model.unfreeze_layers(["encoder.layer.11", "encoder.layer.10"]);
    }
    
    model.enable_mixed_precision();
    model.gradient_checkpointing(enabled=true);
}

@after_batch
def monitor_batch(model, batch_metrics) {
    if batch_metrics.batch_idx % 100 == 0 {
        print(f"Batch {batch_metrics.batch_idx}:");
        print(f"  Loss: {batch_metrics.loss:.4f}");
        print(f"  Accuracy: {batch_metrics.accuracy:.4f}");
    }
}

@after_epoch
def monitor_epoch(model, metrics) {
    # Log metrics
    print(f"Epoch {metrics.epoch} ({current_phase}):");
    print(f"  Accuracy: {metrics.accuracy:.4f}");
    print(f"  F1 Score: {metrics.f1:.4f}");
    
    # Save best model
    if metrics.val_accuracy > best_accuracy {
        model.save_checkpoint(f"models/best_sentiment.pt");
        print("  Saved new best model");
    }
}

# 6. Training Execution
print("Starting sentiment analysis training...");

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
    model.save(f"models/{phase}_sentiment.pt");
}

# 7. Evaluation
test_metrics = model.evaluate(test_data);
print("\nTest Set Performance:");
print(f"Accuracy: {test_metrics.accuracy:.4f}");
print(f"F1 Score: {test_metrics.f1:.4f}");
print(f"Precision: {test_metrics.precision:.4f}");
print(f"Recall: {test_metrics.recall:.4f}");

# 8. Model Analysis
# Analyze attention patterns
model.visualize_attention_patterns(test_data[0:5]);

# Analyze misclassifications
model.analyze_errors(test_data, max_samples=10);

# Generate confusion matrix
model.plot_confusion_matrix(test_data);

# 9. Inference Examples
test_sentences = [
    "This movie was absolutely fantastic!",
    "I've never been so disappointed.",
    "It was okay, nothing special.",
    "A masterpiece of modern cinema."
];

predictions = model.predict(test_sentences);
model.explain_predictions(test_sentences, predictions);

# 10. Save Final Model
model.save("models/final_sentiment.pt", format="torchscript");

# 11. Visualization
plots = MultiPlot(rows=2, cols=2) {
    # Training history
    subplot(0, 0) {
        add_line(history.accuracy, label="Train Accuracy");
        add_line(history.val_accuracy, label="Val Accuracy");
        set_title("Accuracy History");
    }
    
    # Loss history
    subplot(0, 1) {
        add_line(history.loss, label="Train Loss");
        add_line(history.val_loss, label="Val Loss");
        set_title("Loss History");
    }
    
    # Attention visualization
    subplot(1, 0) {
        add_attention_heatmap(test_sentences[0]);
        set_title("Attention Patterns");
    }
    
    # Confidence distribution
    subplot(1, 1) {
        add_confidence_histogram(test_data);
        set_title("Prediction Confidence");
    }
};
plots.save("plots/sentiment_analysis.png");

# 12. Export Analysis Report
model.generate_report({
    "model_architecture": true,
    "training_history": true,
    "performance_metrics": true,
    "error_analysis": true,
    "attention_analysis": true,
    "example_predictions": true,
    "export_format": "pdf"
}); 