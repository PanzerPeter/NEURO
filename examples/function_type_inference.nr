// NEURO Type Inference Engine Demo
// This example demonstrates type inference in functions.

// Function with parameter and return type inference
func add_numbers(a: int, b: int) -> int {
    return a + b  // Type checker verifies return type matches declaration
}

// Generic function with type inference
func identity<T>(value: T) -> T {
    return value  // T is inferred from usage
}

// Function with inferred return type
func calculate_average(numbers: Tensor<float>) -> float {
    let sum = 0.0
    let count = 0
    
    for num in numbers {
        sum = sum + num     // sum remains float
        count = count + 1   // count remains int
    }
    
    return sum / float(count)  // Return type inferred as float
}

func main() {
    let sum = add_numbers(5, 7)
    print("Sum: " + str(sum))

    let five = identity(5)
    let hello = identity("hello")
    print("Identity (int): " + str(five))
    print("Identity (string): " + hello)

    let numbers: Tensor<float> = [1.0, 2.0, 3.0, 4.0, 5.0]
    let avg = calculate_average(numbers)
    print("Average: " + str(avg))
} 