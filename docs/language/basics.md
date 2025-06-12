# Language Basics

This guide covers the fundamental concepts and syntax of the NEURO programming language. These concepts form the foundation for building both AI applications and general-purpose programs.

## Type System

### Primitive Types

NEURO has a strong, static type system with automatic type inference:

```neuro
// Integer types
let small: i8 = 127                    // 8-bit signed integer
let medium: i16 = 32767                // 16-bit signed integer  
let normal: i32 = 2147483647           // 32-bit signed integer (default int)
let large: i64 = 9223372036854775807   // 64-bit signed integer

// Unsigned integers
let ubyte: u8 = 255
let ushort: u16 = 65535
let uint: u32 = 4294967295
let ulong: u64 = 18446744073709551615

// Floating point
let single: f32 = 3.14159              // 32-bit float
let double: f64 = 3.141592653589793    // 64-bit float (default float)

// Boolean and character
let flag: bool = true
let character: char = 'A'

// String (UTF-8)
let text: String = "Hello, NEURO!"
```

### Type Inference

NEURO's type inference reduces verbosity while maintaining safety:

```neuro
// Type inference at work
let age = 25                    // inferred as i32
let height = 175.5              // inferred as f64
let name = "Alice"              // inferred as String
let active = true               // inferred as bool

// Arrays with inference
let numbers = [1, 2, 3, 4, 5]           // Array<i32>
let scores = [95.5, 87.2, 92.1]         // Array<f64>
let names = ["Alice", "Bob", "Charlie"]  // Array<String>

// Complex types with inference
let coordinates = [(0, 0), (1, 2), (3, 4)]  // Array<(i32, i32)>
```

### Collection Types

```neuro
// Arrays (fixed size, stack allocated)
let fixed_array: [i32; 5] = [1, 2, 3, 4, 5]
let inferred_array = [10, 20, 30]  // Array<i32>

// Dynamic arrays (heap allocated)
let mut dynamic: Array<i32> = Array::new()
dynamic.push(1)
dynamic.push(2)
dynamic.push(3)

// Tuples (heterogeneous, fixed size)
let point: (f64, f64) = (3.14, 2.71)
let person: (String, i32, bool) = ("Alice", 25, true)

// Accessing tuple elements
let x = point.0
let y = point.1
let (name, age, is_student) = person  // destructuring

// Hash maps
let mut scores: HashMap<String, i32> = HashMap::new()
scores.insert("Alice", 95)
scores.insert("Bob", 87)
```

## Variables and Mutability

### Immutable by Default

```neuro
// Immutable variables (default)
let name = "Alice"
let age = 25

// This would cause a compile error:
// age = 26  // Error: cannot assign to immutable variable

// Mutable variables (explicit)
let mut counter = 0
counter = counter + 1  // OK

let mut scores = [0, 0, 0]
scores[0] = 95  // OK
```

### Variable Shadowing

```neuro
func shadowing_example() {
    let x = 5          // x is i32
    print(str(x))      // prints "5"
    
    let x = "hello"    // x is now String (shadowing)
    print(x)           // prints "hello"
    
    {
        let x = true   // x is now bool in this scope
        print(str(x))  // prints "true"
    }
    
    print(x)           // prints "hello" (back to String)
}
```

## Functions

### Function Definition

```neuro
// Basic function
func add(a: i32, b: i32) -> i32 {
    return a + b
}

// Function with type inference
func greet(name: String) {  // return type inferred as ()
    print("Hello, " + name + "!")
}

// Multiple return values
func divide_with_remainder(a: i32, b: i32) -> (i32, i32) {
    return (a / b, a % b)
}

// Default parameters
func create_user(name: String, age: i32 = 18, active: bool = true) -> User {
    return User { name: name, age: age, active: active }
}
```

### Generic Functions

```neuro
// Generic function with type parameters
func swap<T>(a: T, b: T) -> (T, T) {
    return (b, a)
}

// Generic function with constraints
func max<T>(a: T, b: T) -> T where T: Comparable {
    if a > b {
        return a
    } else {
        return b
    }
}

// Multiple type parameters
func map<T, U>(array: Array<T>, transform: func(T) -> U) -> Array<U> {
    let mut result: Array<U> = Array::new()
    for item in array {
        result.push(transform(item))
    }
    return result
}
```

### Higher-Order Functions

```neuro
// Functions as first-class values
func apply_twice(f: func(i32) -> i32, x: i32) -> i32 {
    return f(f(x))
}

func double(x: i32) -> i32 {
    return x * 2
}

func main() {
    let result = apply_twice(double, 5)  // result = 20
    print(str(result))
}

// Lambda expressions
func lambda_example() {
    let numbers = [1, 2, 3, 4, 5]
    
    // Lambda with explicit types
    let doubled = numbers.map(|x: i32| -> i32 { x * 2 })
    
    // Lambda with inference
    let squared = numbers.map(|x| x * x)
    
    // Lambda with multiple statements
    let processed = numbers.map(|x| {
        let temp = x * 2
        return temp + 1
    })
}
```

## Control Flow

### Conditional Statements

```neuro
// If-else statements
func check_grade(score: i32) -> String {
    if score >= 90 {
        return "A"
    } else if score >= 80 {
        return "B"
    } else if score >= 70 {
        return "C"
    } else if score >= 60 {
        return "D"
    } else {
        return "F"
    }
}

// If as expression
func abs(x: i32) -> i32 {
    return if x >= 0 { x } else { -x }
}
```

### Pattern Matching

```neuro
enum Color {
    Red,
    Green,
    Blue,
    RGB(u8, u8, u8),
    HSL { hue: f32, saturation: f32, lightness: f32 }
}

func describe_color(color: Color) -> String {
    match color {
        Red -> "Pure red",
        Green -> "Pure green",
        Blue -> "Pure blue",
        RGB(r, g, b) -> "RGB(" + str(r) + ", " + str(g) + ", " + str(b) + ")",
        HSL { hue: h, saturation: s, lightness: l } -> 
            "HSL(h=" + str(h) + ", s=" + str(s) + ", l=" + str(l) + ")"
    }
}

// Pattern matching with guards
func categorize_number(x: i32) -> String {
    match x {
        n if n < 0 -> "negative",
        0 -> "zero",
        n if n % 2 == 0 -> "positive even",
        _ -> "positive odd"
    }
}

// Matching arrays and tuples
func process_data(data: (String, Array<i32>)) -> String {
    match data {
        ("empty", []) -> "Empty dataset",
        ("single", [x]) -> "Single value: " + str(x),
        (name, numbers) if numbers.len() > 10 -> name + " (large dataset)",
        (name, _) -> name + " (small dataset)"
    }
}
```

### Loops

```neuro
// For loops with ranges
func range_loops() {
    // Inclusive range
    for i in 0..=5 {  // 0, 1, 2, 3, 4, 5
        print(str(i))
    }
    
    // Exclusive range  
    for i in 0..5 {   // 0, 1, 2, 3, 4
        print(str(i))
    }
    
    // Step ranges
    for i in (0..10).step(2) {  // 0, 2, 4, 6, 8
        print(str(i))
    }
}

// For loops with collections
func collection_loops() {
    let numbers = [1, 2, 3, 4, 5]
    
    // Iterate over values
    for num in numbers {
        print(str(num))
    }
    
    // Iterate with index
    for (index, value) in numbers.enumerate() {
        print(str(index) + ": " + str(value))
    }
    
    // Iterate over references (for large objects)
    for ref item in large_objects {
        process(item)  // no copying
    }
}

// While loops
func while_loops() {
    let mut counter = 0
    while counter < 10 {
        print(str(counter))
        counter = counter + 1
    }
    
    // Infinite loop with break
    loop {
        let input = read_line()
        if input == "quit" {
            break
        }
        process_input(input)
    }
}

// Loop control
func loop_control() {
    for i in 0..10 {
        if i % 2 == 0 {
            continue  // skip even numbers
        }
        
        if i > 7 {
            break     // stop at 8
        }
        
        print(str(i))  // prints 1, 3, 5, 7
    }
}
```

## Structs and Enums

### Struct Definition

```neuro
// Basic struct
struct Point {
    x: f64,
    y: f64
}

// Struct with methods
struct Rectangle {
    width: f64,
    height: f64
}

impl Rectangle {
    // Associated function (constructor)
    func new(width: f64, height: f64) -> Rectangle {
        return Rectangle { width: width, height: height }
    }
    
    // Instance method
    func area(self) -> f64 {
        return self.width * self.height
    }
    
    // Mutable method
    func scale(mut self, factor: f64) {
        self.width = self.width * factor
        self.height = self.height * factor
    }
}

// Generic struct
struct Container<T> {
    value: T
}

impl<T> Container<T> {
    func new(value: T) -> Container<T> {
        return Container { value: value }
    }
    
    func get(self) -> T {
        return self.value
    }
}
```

### Enum Definition

```neuro
// Simple enum
enum Direction {
    North,
    South,
    East,
    West
}

// Enum with data
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(u8, u8, u8)
}

// Generic enum
enum Option<T> {
    Some(T),
    None
}

enum Result<T, E> {
    Ok(T),
    Err(E)
}

// Methods on enums
impl<T> Option<T> {
    func is_some(self) -> bool {
        match self {
            Some(_) -> true,
            None -> false
        }
    }
    
    func unwrap(self) -> T {
        match self {
            Some(value) -> value,
            None -> panic("called unwrap on None")
        }
    }
}
```

## Traits (Interfaces)

```neuro
// Define a trait
trait Drawable {
    func draw(self);
    func area(self) -> f64;
}

// Implement trait for struct
impl Drawable for Rectangle {
    func draw(self) {
        print("Drawing rectangle " + str(self.width) + "x" + str(self.height))
    }
    
    func area(self) -> f64 {
        return self.width * self.height
    }
}

// Generic trait constraints
func print_area<T>(shape: T) where T: Drawable {
    print("Area: " + str(shape.area()))
    shape.draw()
}

// Trait with associated types
trait Iterator {
    type Item;
    
    func next(mut self) -> Option<Self::Item>;
}

// Trait inheritance
trait Shape: Drawable {
    func perimeter(self) -> f64;
}
```

## Error Handling

### Result Type

```neuro
// Built-in Result type for error handling
func divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        return Err("Division by zero")
    } else {
        return Ok(a / b)
    }
}

// Using Result
func handle_division() {
    let result = divide(10.0, 2.0)
    
    match result {
        Ok(value) -> print("Result: " + str(value)),
        Err(error) -> print("Error: " + error)
    }
}

// Propagating errors with ?
func calculate_average(numbers: Array<String>) -> Result<f64, String> {
    let mut sum = 0.0
    
    for num_str in numbers {
        let num = parse_float(num_str)?  // propagate error if parsing fails
        sum = sum + num
    }
    
    if numbers.len() == 0 {
        return Err("Cannot calculate average of empty array")
    }
    
    return Ok(sum / numbers.len() as f64)
}
```

### Option Type

```neuro
// Built-in Option type for nullable values
func find_index<T>(array: Array<T>, target: T) -> Option<usize> where T: Eq {
    for (index, item) in array.enumerate() {
        if item == target {
            return Some(index)
        }
    }
    return None
}

// Using Option
func search_example() {
    let numbers = [1, 2, 3, 4, 5]
    let index = find_index(numbers, 3)
    
    match index {
        Some(i) -> print("Found at index: " + str(i)),
        None -> print("Not found")
    }
    
    // Using Option methods
    let doubled = index.map(|i| i * 2)
    let value = index.unwrap_or(0)  // default value if None
}
```

## Memory Management

### Ownership and Borrowing

```neuro
// Ownership transfer
func ownership_example() {
    let data = "Hello"          // data owns the string
    let moved_data = data       // ownership transferred
    // print(data)              // Error: data has been moved
    print(moved_data)           // OK
}

// Borrowing (references)
func borrowing_example() {
    let data = "Hello"
    let reference = &data       // borrow data
    print(reference)            // OK
    print(data)                 // OK, data still owned
}

// Mutable borrowing
func mutable_borrowing() {
    let mut numbers = [1, 2, 3]
    let reference = &mut numbers
    reference.push(4)           // OK
    print(str(numbers.len()))   // OK after reference is done
}
```

## Modules and Imports

```neuro
// Define a module
mod math {
    pub func add(a: i32, b: i32) -> i32 {
        return a + b
    }
    
    func private_helper() -> i32 {
        return 42
    }
    
    pub struct Calculator {
        pub precision: i32
    }
    
    impl Calculator {
        pub func new() -> Calculator {
            return Calculator { precision: 2 }
        }
    }
}

// Using modules
func module_usage() {
    let result = math::add(5, 3)
    let calc = math::Calculator::new()
    
    // Import for convenience
    use math::add;
    let result2 = add(10, 20)
}
```

This covers the fundamental concepts of NEURO's language basics. These features provide a solid foundation for building both AI applications and general-purpose programs with strong type safety and performance characteristics. 