// NEURO Type Inference Engine Demo
// This example demonstrates struct definition and instantiation
// with generic types.

// Struct with type inference
struct Point<T> {
    x: T,
    y: T
}

func create_point() -> Point<int> {
    let int_point = Point { x: 10, y: 20 }        // Point<int>
    let float_point = Point { x: 1.5, y: 2.7 }   // Point<float>
    
    return int_point
}

// Function to create a Point
func create_point(x: int, y: int) -> Point {
    return Point{x: x, y: y}
}

func main() {
    // Create a point using a function
    let p = create_point(10, 20)
    print("Created point at (" + str(p.x) + ", " + str(p.y) + ")")
} 