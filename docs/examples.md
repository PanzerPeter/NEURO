# NEURO Examples

This guide provides practical examples of using NEURO for various deep learning tasks.

## Basic Examples

### 1. Binary Classification

```neuro
# Load and preprocess data
data = load_matrix("heart_disease.csv");
data.normalize();
train_data, val_data = data.split(0.8);

# Create model
model = NeuralNetwork(input_size=13, output_size=1) {
    Dense(units=64, activation="relu");
    Dropout(rate=0.3);
    Dense(units=32, activation="relu");
    Dense(units=1, activation="sigmoid");
}

# Configure training
loss = Loss(type="binary_crossentropy");
optimizer = Optimizer(type="adam", learning_rate=0.001);

# Train
model.train(
    data=train_data,
    validation_data=val_data,
    epochs=50,
    batch_size=32
);
```

### 2. Image Classification (MNIST)

```neuro
model = NeuralNetwork(input_size=784, output_size=10) {
    # Reshape input to 28x28x1
    Reshape(shape=(28, 28, 1));
    
    # Convolutional layers
    Conv2D(filters=32, kernel_size=3, activation="relu");
    MaxPool(pool_size=2);
    Conv2D(filters=64, kernel_size=3, activation="relu");
    MaxPool(pool_size=2);
    
    # Fully connected layers
    Flatten();
    Dense(units=128, activation="relu");
    Dropout(rate=0.5);
    Dense(units=10, activation="softmax");
}
```

### 3. Sequence Prediction (Time Series)

```neuro
model = NeuralNetwork(input_size=(30, 1), output_size=1) {
    LSTM(units=64, return_sequences=true);
    LSTM(units=32);
    Dense(units=16, activation="relu");
    Dense(units=1, activation="linear");
}
```

## Advanced Examples

### 1. Transfer Learning with ResNet

```neuro
@pretrained("resnet18")
backbone = Backbone(trainable=false);

model = NeuralNetwork(input_size=224, output_size=100) {
    backbone(x);
    GlobalAveragePooling2D();
    Dense(units=512, activation="relu");
    Dropout(rate=0.5);
    Dense(units=100, activation="softmax");
}

# Fine-tuning
backbone.trainable = true;
model.train(data, epochs=10, learning_rate=0.0001);
```

### 2. Custom Layer with Residual Connection

```neuro
@custom_layer
def ResidualBlock(x) {
    skip = x;
    
    Conv2D(filters=64, kernel_size=3, padding="same");
    BatchNorm();
    ReLU();
    
    Conv2D(filters=64, kernel_size=3, padding="same");
    BatchNorm();
    
    return x + skip;
}

model = NeuralNetwork(input_size=(32, 32, 3), output_size=10) {
    Conv2D(filters=64, kernel_size=3);
    ResidualBlock();
    ResidualBlock();
    GlobalAveragePooling2D();
    Dense(units=10, activation="softmax");
}
```

### 3. Multi-Branch Architecture

```neuro
model = NeuralNetwork(input_size=224, output_size=1000) {
    # Image branch
    Branch("image_branch") {
        Conv2D(filters=64, kernel_size=7, stride=2);
        MaxPool(pool_size=3, stride=2);
        ResidualBlock();
    }
    
    # Metadata branch
    Branch("metadata_branch") {
        Dense(units=256, activation="relu");
        Dropout(rate=0.3);
    }
    
    # Merge branches
    Concatenate(branches=["image_branch", "metadata_branch"]);
    Dense(units=1000, activation="softmax");
}
```

### 4. Sequence-to-Sequence Model

```neuro
model = NeuralNetwork(input_size=(None, 256), output_size=(None, 256)) {
    # Encoder
    LSTM(units=512, return_sequences=true);
    LSTM(units=512);
    
    # Decoder
    RepeatVector(n=max_sequence_length);
    LSTM(units=512, return_sequences=true);
    TimeDistributed(Dense(units=256));
}
```

## Training Examples

### 1. Custom Training Loop

```neuro
@before_training
def prepare_batch(batch) {
    # Augment images
    return batch.augment(
        rotation=15,
        flip=true,
        zoom=0.1
    );
}

@after_batch
def log_metrics(metrics) {
    print(f"Batch loss: {metrics.loss}");
}

@after_epoch
def check_improvement(model, metrics) {
    if metrics.val_loss < best_loss {
        model.save("best_model.pt");
    }
}
```

### 2. Memory-Efficient Training

```neuro
# Enable memory optimization
model.enable_checkpointing();
model.set_memory_limit(max_gb=4);

# Configure dynamic batch sizing
model.train(
    data=large_dataset,
    batch_size="auto",
    max_memory_usage=0.8
);
```

### 3. Multi-GPU Training

```neuro
model.distribute(strategy="data_parallel");
model.train(
    data=huge_dataset,
    batch_size=256 * num_gpus,
    epochs=100
);
```

## Deployment Examples

### 1. Model Export

```neuro
# Save for production
model.save("production_model.pt", format="torchscript");

# Export ONNX format
model.export("model.onnx", input_shape=(1, 3, 224, 224));
```

### 2. Model Serving

```neuro
# Create serving endpoint
model.serve(
    port=8080,
    max_batch_size=32,
    timeout=100
);
```

### 3. Model Optimization

```neuro
# Quantize model
model.quantize(precision="int8");

# Optimize for inference
model.optimize(target="inference");
```

Each example includes best practices and common patterns used in real-world applications. For more detailed explanations and additional examples, check our [tutorials](tutorials.md) section. 