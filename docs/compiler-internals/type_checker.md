# Type Checker

The type checker, or semantic analyzer, is the third stage of the NEURO compiler. After the parser has built the Abstract Syntax Tree (AST), the type checker traverses this tree to ensure that the code is semantically correct and that all types are used consistently.

- **Implementation**: `src/neuro/type_checker.py`
- **Main Class**: `NeuroTypeChecker`

## Key Responsibilities

The type checker is responsible for catching a class of errors that the parser cannot see. The parser only checks if the code follows the language's grammatical rules, while the type checker verifies if the code *makes sense*.

Its primary responsibilities include:

1.  **Type Checking**: Verifying that operations are performed on compatible types. For example, it will raise an error if you try to add a string to a tensor (`"hello" + my_tensor`) or use a boolean in a mathematical calculation.

2.  **Type Inference**: Deducing the types of variables and expressions when they are not explicitly annotated by the user. For a declaration like `let x = 10;`, the type checker infers that `x` has the type `i32` (default integer).

3.  **Symbol Table Management**: Maintaining a "symbol table" to keep track of all declared variables, functions, and their types within the current scope. This is how it detects errors like using a variable before it has been declared or redeclaring a variable in the same scope.

4.  **AST Annotation**: As it traverses the tree, the type checker annotates nodes in the AST with their determined types. This annotated AST is then passed to the code generator, which uses the type information to produce correct and optimized code.

## How it Works

The type checker uses the **Visitor design pattern** to traverse the AST. It has a `visit` method for each type of AST node (e.g., `visit_VariableDeclaration`, `visit_BinaryOperation`, `visit_FunctionCall`).

- The process starts with the `check_program` method, which sets up a global scope and then visits each statement in the program.
- When visiting an expression, the checker recursively visits the sub-expressions to determine their types first. For example, to find the type of `a + b`, it first finds the types of `a` and `b`, then checks if the `+` operator is valid for those types.
- For scope management (e.g., inside functions), it creates nested symbol tables. When a scope is exited, its symbol table is discarded.

## Type Inference

Type inference is a key feature of NEURO's type system.
- When a variable is declared with an initial value (e.g., `let x = 10.5;`), the type checker determines the type of the right-hand side expression (`f32`) and assigns it to the variable `x`.
- It can also infer function return types if they are not specified, by analyzing the types of the `return` statements within the function body.
- This provides the safety of a static type system without the verbosity of explicitly writing down every single type. 