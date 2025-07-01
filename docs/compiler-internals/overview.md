# Compiler Architecture Overview

The NEURO compiler (`neurc`) is designed with a classic, multi-stage architecture that transforms NEURO source code into an executable format. This modular design allows for clear separation of concerns, making the compiler easier to develop, debug, and extend.

The compilation process is orchestrated by the `NeuroCompiler` class found in `src/neuro/compiler.py`.

## Compilation Pipeline

The compiler processes a source file in a sequential pipeline. Each stage takes the output of the previous one as its input.

1.  **Lexical Analysis (Lexing)**
    *   **Input**: Raw source code text.
    *   **Output**: A stream of tokens.
    *   **Implementation**: `src/neuro/lexer.py`
    *   The lexer scans the source code and breaks it down into a series of atomic units called tokens. Examples of tokens include identifiers (`my_variable`), keywords (`let`, `func`), operators (`+`, `=`), and literals (`123`, `"hello"`).

2.  **Syntactic Analysis (Parsing)**
    *   **Input**: A stream of tokens from the lexer.
    *   **Output**: An Abstract Syntax Tree (AST).
    *   **Implementation**: `src/neuro/parser.py`
    *   The parser takes the flat list of tokens and organizes them into a hierarchical structure called an AST. This tree represents the grammatical structure of the code, making it easier for subsequent stages to understand. The nodes of this tree are defined in `src/neuro/ast_nodes.py`.

3.  **Semantic Analysis (Type Checking)**
    *   **Input**: The Abstract Syntax Tree.
    *   **Output**: A validated and annotated AST.
    *   **Implementation**: `src/neuro/type_checker.py`
    *   The type checker traverses the AST to verify that the code is semantically correct. It checks for things like type mismatches, use of undeclared variables, and correct function call signatures. It also performs type inference, deducing the types of variables where they are not explicitly stated.

4.  **Code Generation**
    *   **Input**: The validated AST.
    *   **Output**: The final compiled artifact (e.g., an executable, LLVM IR).
    *   **Implementation**: `src/neuro/compiler.py` (`_generate_code` method)
    *   This is the final stage where the high-level AST is translated into low-level code. The current implementation is primarily for demonstration and can generate:
        *   A Python script that interprets the AST (for creating a runnable "executable").
        *   Basic LLVM Intermediate Representation (IR).
        *   Stub assembly code.

## Error Handling

Each stage of the compiler can detect and report errors. The `ErrorReporter` class (`src/neuro/errors.py`) is used throughout the compiler to accumulate errors and warnings, which are then displayed to the user at the end of the compilation process. This ensures that the user receives a comprehensive list of issues at once, rather than the compilation halting at the first error. 