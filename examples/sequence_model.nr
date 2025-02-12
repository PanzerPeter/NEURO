# Create a sequence-to-sequence model with attention and residual connections
model = NeuralNetwork(input_size=512, output_size=10000, sequence_length=20) {
    # Embedding layer for input tokens with dropout
    Embedding(vocab_size=10000, embedding_dim=768)  # Increased embedding dimension
    Dropout(rate=0.2)
    
    # Bidirectional LSTM encoder with increased capacity
    LSTM(hidden_size=768, num_layers=3, bidirectional=true)
    normalize()
    Dropout(rate=0.3)
    
    # Multi-head self-attention mechanism
    Attention(num_heads=12, dropout=0.3)  # Increased number of heads
    normalize()
    
    # GRU decoder with increased capacity
    GRU(hidden_size=768, num_layers=3)
    normalize()
    Dropout(rate=0.3)
    
    # Output layers with residual connections and layer normalization
    Dense(units=1536, activation="relu")  # Increased units
    normalize()
    Dropout(rate=0.4)
    Dense(units=768, activation="relu")
    normalize()
    Dropout(rate=0.4)
    Dense(units=10000)
    normalize()
    
    # Configure loss and optimizer with regularization
    loss(type="cross_entropy")
    optimizer(type="adam", learning_rate=0.001, weight_decay=0.01)  # Added weight decay
}

# Set gradient clipping
model.clip_gradients(max_norm=1.0)

# Load training data
data = load_matrix("examples/sequence_data.nrm")

# Train the model with curriculum learning and label smoothing
model.train(
    data,
    epochs=100,  # Increased epochs
    learning_rate=0.001,
    teacher_forcing_ratio=0.6,  # Slightly increased teacher forcing
    label_smoothing=0.1,
    scheduler="cosine",  # Added learning rate scheduler
    warmup_steps=10
)

# Save the trained model
save_model(model, "sequence_model.pt")

# Evaluate on test data with beam search
test = load_matrix("examples/sequence_test.nrm")
accuracy = model.evaluate(test, beam_size=5)  # Added beam search for evaluation

# Print results
print("Model accuracy: ", accuracy) 