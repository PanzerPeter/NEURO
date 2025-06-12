# NEURO Language Specification

This document provides the complete, authoritative specification for the NEURO programming language. It covers syntax, semantics, and behavior in detail.

## Language Overview

NEURO is a statically-typed, compiled programming language designed for high-performance AI development. It features:

- **Static typing** with type inference
- **Memory safety** through ownership and borrowing
- **Zero-cost abstractions** for high-level programming without runtime overhead
- **First-class AI support** with built-in tensors and neural network constructs
- **LLVM compilation** for native performance

## Lexical Structure

### Comments

```neuro
// Line comment extends to end of line

/* Block comment
   can span multiple lines */

/// Documentation comment for the following item
func documented_function() {}

/**
 * Multi-line documentation comment
 * with formatting preserved
 */
struct DocumentedStruct {}
```

### Identifiers

```bnf
identifier = letter (letter | digit | '_')*
letter = 'a'..'z' | 'A'..'Z' | '_'
digit = '0'..'9'
```

**Rules:**
- Must start with letter or underscore
- Case-sensitive
- Cannot be reserved keywords
- Convention: `snake_case` for variables/functions, `PascalCase` for types

### Keywords

**Reserved Keywords:**
```
let mut func struct enum trait impl where
if else match for while loop break continue return
true false pub mod use as super self
async await static const unsafe extern
type alias union macro
```

**Primitive Types:**
```
i8 i16 i32 i64 isize
u8 u16 u32 u64 usize  
f32 f64
bool char String
```

**AI-Specific Keywords:**
```
Tensor NeuralNetwork Dataset
@gpu @cpu @parallel
forward backward
```

### Literals

#### Integer Literals
```neuro
// Decimal
42
1_000_000  // underscores for readability

// Binary
0b1010_1100

// Octal  
0o755

// Hexadecimal
0xFF_A0

// Type suffixes
42i64      // 64-bit signed integer
100u32     // 32-bit unsigned integer
```

#### Floating Point Literals
```neuro
3.14
2.5e10     // scientific notation
1.23E-4
6.022e23f32  // with type suffix
```

#### Boolean Literals
```neuro
true
false
```

#### Character Literals
```neuro
'a'
'Z'
'\n'       // newline
'\t'       // tab
'\''       // single quote
'\\'       // backslash
'\u{1F600}' // Unicode code point (😀)
```

#### String Literals
```neuro
"Hello, world!"
"Line 1\nLine 2"           // escape sequences
"Unicode: \u{1F680}"       // Unicode escapes
r"Raw string with \n"      // raw string (no escapes)
r#"Raw with "quotes""#     // raw string with delimiter
```

### Operators

#### Arithmetic Operators
```neuro
+    // addition
-    // subtraction, unary minus
*    // multiplication
/    // division
%    // remainder
@    // matrix multiplication (tensors)
**   // exponentiation
```

#### Comparison Operators
```neuro
==   // equal
!=   // not equal
<    // less than
<=   // less than or equal
>    // greater than
>=   // greater than or equal
```

#### Logical Operators
```neuro
&&   // logical AND
||   // logical OR
!    // logical NOT
```

#### Bitwise Operators
```neuro
&    // bitwise AND
|    // bitwise OR
^    // bitwise XOR
<<   // left shift
>>   // right shift
~    // bitwise NOT
```

#### Assignment Operators
```neuro
=    // assignment
+=   // add and assign
-=   // subtract and assign
*=   // multiply and assign
/=   // divide and assign
%=   // remainder and assign
```

#### Other Operators
```neuro
&    // reference (borrow)
*    // dereference
.    // field access
::   // path separator
..   // exclusive range
..=  // inclusive range
?    // error propagation
```

## Type System

### Primitive Types

#### Integer Types
```neuro
i8    // 8-bit signed integer (-128 to 127)
i16   // 16-bit signed integer
i32   // 32-bit signed integer (default int)
i64   // 64-bit signed integer
isize // pointer-sized signed integer

u8    // 8-bit unsigned integer (0 to 255)
u16   // 16-bit unsigned integer
u32   // 32-bit unsigned integer
u64   // 64-bit unsigned integer
usize // pointer-sized unsigned integer
```

#### Floating Point Types
```neuro
f32   // 32-bit IEEE 754 floating point
f64   // 64-bit IEEE 754 floating point (default float)
```

#### Other Primitive Types
```neuro
bool   // Boolean: true or false
char   // Unicode scalar value (32-bit)
()     // Unit type (empty tuple)
!      // Never type (functions that never return)
```

### Compound Types

#### Tuples
```neuro
()              // unit type
(i32,)          // single-element tuple
(i32, f64)      // two-element tuple
(String, i32, bool)  // multi-element tuple

// Destructuring
let (x, y) = (10, 20)
let (name, age, _) = person  // ignore third element
```

#### Arrays
```neuro
[i32; 5]        // fixed-size array of 5 integers
[0; 10]         // array of 10 zeros
[1, 2, 3, 4, 5] // array literal

// Dynamic arrays
Array<i32>      // growable array
```

#### References
```neuro
&T              // immutable reference to T
&mut T          // mutable reference to T
```

#### Function Types
```neuro
func(i32) -> i32              // function taking i32, returning i32
func(String, bool) -> ()      // function taking String and bool, returning unit
func<T>(T) -> T               // generic function type
```

### Generic Types

#### Type Parameters
```neuro
struct Container<T> {
    value: T
}

enum Option<T> {
    Some(T),
    None
}

func identity<T>(x: T) -> T {
    return x
}
```

#### Type Constraints
```neuro
func max<T>(a: T, b: T) -> T where T: Comparable {
    if a > b { a } else { b }
}

func print_debug<T>(value: T) where T: Debug {
    print(debug(value))
}

// Multiple constraints
func complex_function<T, U>(x: T, y: U) -> T 
where 
    T: Clone + Debug,
    U: Into<T>
{
    // implementation
}
```

### AI-Specific Types

#### Tensor Types
```neuro
Tensor<f32>                    // 1D tensor of f32
Tensor<f64, (3, 4)>           // 2D tensor with shape (3, 4)
Tensor<i32, (2, 3, 4)>        // 3D tensor with shape (2, 3, 4)
Tensor<bool, _>               // tensor with inferred shape
```

#### Neural Network Types
```neuro
NeuralNetwork<f32, (784, 128, 10)>    // network with specified layer sizes
Layer<f32>                            // generic layer type
Optimizer<f32>                        // optimizer type
```

## Syntax

### Variable Declarations

#### Immutable Variables
```neuro
let x = 42              // type inferred as i32
let y: f64 = 3.14       // explicit type annotation
let name = "Alice"      // type inferred as String
```

#### Mutable Variables
```neuro
let mut counter = 0     // mutable integer
let mut data: Array<i32> = Array::new()  // mutable array
```

#### Pattern Matching in Let
```neuro
let (x, y) = (10, 20)           // tuple destructuring
let Point { x, y } = point      // struct destructuring
let [first, second, ..rest] = array  // array pattern
```

### Function Definitions

#### Basic Functions
```neuro
func greet() {
    print("Hello!")
}

func add(a: i32, b: i32) -> i32 {
    return a + b
}

// Expression body (no return needed)
func double(x: i32) -> i32 { x * 2 }
```

#### Generic Functions
```neuro
func swap<T>(a: T, b: T) -> (T, T) {
    return (b, a)
}

func map<T, U>(array: Array<T>, f: func(T) -> U) -> Array<U> {
    let mut result = Array::new()
    for item in array {
        result.push(f(item))
    }
    return result
}
```

#### Methods
```neuro
struct Rectangle { width: f64, height: f64 }

impl Rectangle {
    // Associated function (constructor)
    func new(width: f64, height: f64) -> Rectangle {
        Rectangle { width, height }
    }
    
    // Instance method
    func area(self) -> f64 {
        self.width * self.height
    }
    
    // Mutable method
    func scale(mut self, factor: f64) {
        self.width *= factor
        self.height *= factor
    }
}
```

### Control Flow

#### Conditional Expressions
```neuro
if condition {
    // then branch
} else if other_condition {
    // else if branch  
} else {
    // else branch
}

// If expression
let result = if x > 0 { "positive" } else { "non-positive" }
```

#### Pattern Matching
```neuro
match value {
    pattern1 => expression1,
    pattern2 if guard => expression2,
    _ => default_expression,
}

// Exhaustive matching required
match color {
    Red => "red",
    Green => "green", 
    Blue => "blue",
    RGB(r, g, b) => "custom color",
}
```

#### Loops

**For Loops:**
```neuro
// Range iteration
for i in 0..10 { /* ... */ }      // 0 to 9
for i in 0..=10 { /* ... */ }     // 0 to 10

// Collection iteration
for item in collection { /* ... */ }
for (index, item) in collection.enumerate() { /* ... */ }

// Reference iteration
for ref item in large_collection { /* ... */ }
```

**While Loops:**
```neuro
while condition {
    // loop body
}

// Infinite loop
loop {
    // body
    if break_condition {
        break
    }
}
```

**Loop Control:**
```neuro
for i in 0..10 {
    if i % 2 == 0 {
        continue  // skip to next iteration
    }
    
    if i > 7 {
        break     // exit loop
    }
}
```

### Struct Definitions

#### Basic Structs
```neuro
struct Point {
    x: f64,
    y: f64,
}

struct User {
    name: String,
    age: i32,
    active: bool,
}
```

#### Generic Structs
```neuro
struct Container<T> {
    value: T,
}

struct Pair<T, U> {
    first: T,
    second: U,
}
```

#### Tuple Structs
```neuro
struct Color(u8, u8, u8)        // RGB color
struct Meters(f64)              // newtype pattern
```

#### Unit Structs
```neuro
struct Marker;                  // zero-sized type
struct GlobalState;             // singleton pattern
```

### Enum Definitions

#### Simple Enums
```neuro
enum Direction {
    North,
    South,
    East,
    West,
}
```

#### Enums with Data
```neuro
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(u8, u8, u8),
}
```

#### Generic Enums
```neuro
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

### Trait Definitions

#### Basic Traits
```neuro
trait Display {
    func fmt(self) -> String;
}

trait Clone {
    func clone(self) -> Self;
}
```

#### Traits with Associated Types
```neuro
trait Iterator {
    type Item;
    
    func next(mut self) -> Option<Self::Item>;
}
```

#### Trait Inheritance
```neuro
trait Eq: PartialEq {
    // Eq extends PartialEq
}

trait Debug: Display {
    func debug_fmt(self) -> String;
}
```

### Module System

#### Module Declaration
```neuro
mod utils {
    pub func helper() -> i32 { 42 }
    
    func private_function() {}
    
    pub mod nested {
        pub struct PublicStruct {}
    }
}
```

#### Imports
```neuro
use std::collections::HashMap;
use std::io::{Read, Write};
use utils::helper;
use utils::nested::PublicStruct as PS;
```

#### Visibility
```neuro
pub struct PublicStruct {}      // public
struct PrivateStruct {}         // private to module

pub(crate) func crate_visible() {}  // visible in crate
pub(super) func parent_visible() {} // visible to parent module
```

## AI-Specific Language Features

### Tensor Operations

#### Tensor Creation
```neuro
// Literals
let vector: Tensor<f32> = [1.0, 2.0, 3.0, 4.0]
let matrix: Tensor<f32> = [[1.0, 2.0], [3.0, 4.0]]

// Explicit construction
let zeros = Tensor::zeros<f32, (3, 4)>()
let ones = Tensor::ones<f32, (2, 2)>()
let random = Tensor::random<f32, (5, 5)>()
```

#### Tensor Arithmetic
```neuro
let a: Tensor<f32> = [1.0, 2.0, 3.0]
let b: Tensor<f32> = [4.0, 5.0, 6.0]

let sum = a + b           // element-wise addition
let product = a * b       // element-wise multiplication
let dot_product = a @ b   // matrix/dot product
let scaled = a * 2.0      // scalar multiplication
```

### Neural Network DSL

#### Network Definition
```neuro
let model = NeuralNetwork<f32, (784, 128, 64, 10)> {
    dense_layer(units=128, activation=relu),
    batch_norm(),
    dropout(rate=0.2),
    dense_layer(units=64, activation=relu),
    batch_norm(),
    dense_layer(units=10, activation=softmax)
}
```

#### Training Loop
```neuro
for epoch in 0..num_epochs {
    for batch in training_data.batches(batch_size) {
        let predictions = model.forward(batch.inputs)
        let loss = cross_entropy_loss(predictions, batch.targets)
        
        model.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    }
}
```

### GPU Programming

#### GPU Function Annotation
```neuro
@gpu func matrix_multiply<T>(a: Tensor<T>, b: Tensor<T>) -> Tensor<T> {
    // Automatically compiled to CUDA/OpenCL
    return a @ b
}

@cpu func cpu_computation() {
    // Explicitly run on CPU
}
```

#### Device Management
```neuro
let device = Device::cuda(0)  // First CUDA device
let tensor = tensor.to_device(device)

// Parallel execution
@parallel func parallel_map<T, U>(data: Array<T>, f: func(T) -> U) -> Array<U> {
    // Automatically parallelized
}
```

## Memory Model

### Ownership Rules

1. **Each value has exactly one owner**
2. **When the owner goes out of scope, the value is dropped**
3. **Values can be moved or borrowed, but not both simultaneously**

#### Move Semantics
```neuro
let s1 = String::from("hello")
let s2 = s1              // s1 is moved to s2
// print(s1)             // Error: s1 has been moved
print(s2)                // OK
```

#### Borrowing
```neuro
func calculate_length(s: &String) -> usize {
    s.len()  // s is a reference, doesn't take ownership
}

let s = String::from("hello")
let len = calculate_length(&s)  // borrow s
print(s)                        // still valid
```

#### Mutable Borrowing
```neuro
func append_world(s: &mut String) {
    s.push_str(", world!")
}

let mut s = String::from("hello")
append_world(&mut s)
print(s)  // "hello, world!"
```

### Lifetime Annotations

#### Basic Lifetimes
```neuro
func longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

#### Struct Lifetimes
```neuro
struct ImportantExcerpt<'a> {
    part: &'a str,
}
```

## Error Handling

### Result Type
```neuro
enum Result<T, E> {
    Ok(T),
    Err(E),
}

func divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("division by zero".to_string())
    } else {
        Ok(a / b)
    }
}
```

### Error Propagation
```neuro
func complex_operation() -> Result<i32, String> {
    let x = risky_operation()?  // propagate error
    let y = another_operation(x)?
    Ok(y * 2)
}
```

### Option Type
```neuro
enum Option<T> {
    Some(T),
    None,
}

func find_user(id: i32) -> Option<User> {
    // return Some(user) if found, None otherwise
}
```

## Concurrency

### Async/Await
```neuro
async func fetch_data(url: String) -> Result<String, Error> {
    let response = http_get(url).await?
    Ok(response.text().await?)
}

async func main() {
    let data = fetch_data("https://api.example.com").await
    match data {
        Ok(content) => print(content),
        Err(e) => print("Error: " + e.to_string()),
    }
}
```

### Parallel Processing
```neuro
use std::thread

func parallel_computation() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8]
    
    let results: Array<i32> = data
        .par_iter()                    // parallel iterator
        .map(|x| expensive_computation(*x))
        .collect()
}
```

This specification provides the complete syntax and semantics for the NEURO programming language, serving as the authoritative reference for language implementation and usage. 