# Simple binary classification model

model = NeuralNetwork(input_size=2, output_size=1) {
    Dense(units=4, activation="relu");
    Dense(units=1, activation="sigmoid");
}

# Configure training
loss = Loss(type="bce");
optimizer = Optimizer(type="adam", learning_rate=0.001);

# Train the model
model.train(data, epochs=10); 

# Simple binary classification model 