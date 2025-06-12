// NEURO Type Inference Engine Demo
// This example demonstrates complex tensor operations with shape checking.

// Complex tensor operations with shape checking
func matrix_operations() -> Tensor<float, (3, 3)> {
    let a: Tensor<float, (3, 3)> = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0], 
        [7.0, 8.0, 9.0]
    ]
    
    let b: Tensor<float, (3, 3)> = [
        [9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0]
    ]
    
    // Element-wise operations (type checker ensures same shape)
    let sum_matrix = a + b      // Tensor<float, (3, 3)>
    let diff_matrix = a - b     // Tensor<float, (3, 3)>
    
    // Matrix multiplication (type checker verifies compatible shapes)
    let product_matrix = a @ b  // Tensor<float, (3, 3)>
    
    return product_matrix
}

func main() {
    let product = matrix_operations()
    print("Product of matrices:")
    print(str(product))
} 