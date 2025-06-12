// NEURO Type Inference Engine Demo
// This example demonstrates basic type inference for variables,
// tensor, and control flow.

func main() {
    // Variable type inference
    let x = 42              // Inferred: int
    let y = 3.14           // Inferred: float  
    let name = "NEURO"     // Inferred: string
    let is_ready = true    // Inferred: bool
    
    // Mixed operations with type checking
    let sum = x + 10       // Inferred: int (x is int, 10 is int)
    let product = y * 2.5  // Inferred: float (y is float, 2.5 is float)
    
    // Function calls with type checking
    print("Hello " + name + "!")
    
    // Tensor type inference with shape checking
    let vector: Tensor<float> = [1.0, 2.0, 3.0, 4.0]  // Shape: (4,)
    let matrix = [[1.0, 2.0], [3.0, 4.0]]             // Inferred: Tensor<float, (2, 2)>
    
    // Tensor operations
    let dot_product = vector @ vector  // Inferred: float (dot product)
    
    // Control flow with type checking
    if x > 0 {
        print("x is positive")
    }
    
    // Loop with iterator type inference
    for element in vector {
        // element inferred as float (element type of vector)
        print("Element: " + str(element))
    }
} 