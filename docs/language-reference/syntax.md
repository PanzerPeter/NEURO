# Basic Syntax

This page covers the fundamental syntax of the NEURO programming language.

## Comments

Comments are ignored by the compiler and are used to add explanatory notes to the code. NEURO supports single-line comments.

```neuro
// This is a single-line comment.
let x = 10 // This is an inline comment.
```

## Variables and Constants

You can declare variables and constants using the `let` keyword. NEURO is statically typed, but it has powerful type inference, so you often don't need to write the types explicitly.

```neuro
// The compiler infers the type of 'x' as an integer.
let x = 10

// You can also explicitly annotate the type.
let y: f32 = 3.14

// Variables are immutable by default. To make them mutable, use 'mut'.
let mut z = 5
z = z + 1 // This is valid.
```

## Data Types

NEURO has a rich set of built-in data types.

### Primitive Types
*   **Integers**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
*   **Floating-Point**: `f32`, `f64`
*   **Boolean**: `bool` (`true` or `false`)
*   **String**: `string`

```neuro
let is_active: bool = true
let name: string = "Neuro"
```

### AI-Specific Types
*   **Tensor**: A first-class citizen in NEURO. Tensors are multi-dimensional arrays used in AI computations.
    ```neuro
    let vector: Tensor<f32> = [1.0, 2.0, 3.0]
    let matrix: Tensor<i32> = [[1, 2], [3, 4]]
    ```
*   **NeuralNetwork**: A special type for defining neural network architectures.

## Functions

Functions are defined with the `func` keyword. They can have parameters and return values.

```neuro
// A simple function with no parameters or return value.
func do_something() {
    print("Doing something!")
}

// A function with parameters and a return type.
func add(a: i32, b: i32) -> i32 {
    return a + b
}

// Calling functions
do_something()
let sum = add(5, 10)
```

## Control Flow

NEURO supports common control flow statements.

### If-Else
```neuro
let number = 10
if number > 5 {
    print("Number is greater than 5")
} else {
    print("Number is not greater than 5")
}
```

### For Loops
```neuro
// Loop over a range
for i in 0..5 {
    print("Iteration: " + str(i))
}

// Loop over items in a tensor
let data: Tensor<f32> = [1.0, 2.0, 3.0]
for item in data {
    print(item)
}
``` 