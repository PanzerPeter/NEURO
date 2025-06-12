// NEURO Type Inference Engine Demo
// This example file documents common type errors that the NEURO
// compiler would catch.

func error_examples() {
    // The following lines are examples of code that would fail to compile.
    // They are commented out to allow this file to be parsed.
    
    // Error: Cannot unify int with string
    // let bad_assignment: int = "string"
    
    // Error: Cannot perform addition between int and string
    // let bad_math = 5 + "hello"
    
    // Error: Conditional expression must be of type bool
    // if "not bool" {
    //     print("This won't work.")
    // }
    
    // Error: Cannot unify int with float in Tensor element type
    // let bad_tensor: Tensor<int> = [1.0, 2.0]
}

func main() {
    // Calling error_examples() would result in compile-time errors.
    // error_examples()
    print("This file contains examples of type errors.")
} 