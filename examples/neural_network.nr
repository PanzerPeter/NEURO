// NEURO Type Inference Engine Demo
// This example demonstrates defining and using a neural network.

// Define the model globally
let model = NeuralNetwork<f32, (784, 128, 10)> {
    dense_layer<128>(.relu),
    batch_norm(),
    dropout(0.2),
    dense_layer<10>(.softmax)
}

// Global variable for input data
let input: Tensor<float, (1, 784)> = create_input_data()

// Function to create some dummy data
func create_input_data() -> Tensor<float, (1, 784)> {
    return zeros(1, 784)
}

func main() {
    let output = model.forward(input) // `model` is now defined globally
    print("Model output calculated.")
} 