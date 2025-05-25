// Simple Hello World in NEURO
// This demonstrates basic language syntax

func main() {
    let message = "Hello, NEURO World!"
    print(message)
    
    // Basic arithmetic
    let x = 42
    let y = 3.14
    let z = x + y
    
    print("The answer is: " + str(z))
}

// Example function with parameters and return type
func add(a: int, b: int) -> int {
    return a + b
}

// Example with tensor operations (AI-focused)
func create_simple_tensor() -> Tensor<float> {
    let data = [1.0, 2.0, 3.0, 4.0]
    return data
} 