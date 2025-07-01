// NEURO Type Inference Engine Demo
// This example demonstrates the type constraint solving process.

// Demonstrating type constraint solving
func constraint_solving_demo() -> string {
    // Type variables are created for unknowns, and constraints are gathered.
    let a = unknown_function()      // `a` is given a type variable, say ?T0
    let b = a + 5                   // Adds constraint: ?T0 must support `+` with int. This resolves ?T0 to `int`.
    let c = str(a)                  // Adds constraint: `a` (now int) must be convertible to string.
    
    // The type checker solves the constraints, deducing that:
    // a: int
    // b: int
    // c: string
    return c
}

// A function whose return type is not immediately obvious to the parser.
// The type checker will infer it.
func unknown_function() -> int {
    return 42  // The body reveals the return type is `int`.
}

func main() {
    let result = constraint_solving_demo()
    print("Constraint solving result: " + result)
} 