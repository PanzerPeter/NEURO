# Lexer

The lexer, or lexical analyzer, is the first stage of the NEURO compiler. Its job is to read the raw source code and convert it into a sequence of "tokens."

- **Implementation**: `src/neuro/lexer.py`
- **Main Class**: `NeuroLexer`

## What is a Token?

A token is a small, atomic unit of the language's syntax. It has three main components:

- **Type**: The category of the token (e.g., `IDENTIFIER`, `KEYWORD`, `INTEGER_LITERAL`).
- **Value**: The actual text from the source code that this token represents (e.g., `"my_variable"`, `"let"`, `"123"`).
- **Position**: The line and column number where the token appeared. This is crucial for providing meaningful error messages.

For example, the line of code `let x = 10;` would be broken down into the following tokens:
1. `Token(KEYWORD, 'let', ...)`
2. `Token(IDENTIFIER, 'x', ...)`
3. `Token(OPERATOR, '=', ...)`
4. `Token(INTEGER_LITERAL, '10', ...)`
5. `Token(PUNCTUATION, ';', ...)`

## How it Works

The `NeuroLexer` class is initialized with the full source code string. The primary method is `tokenize()`, which iterates through the source code character by character, building up tokens.

The lexer maintains a current position and uses a series of helper methods to identify different kinds of tokens:

- It recognizes keywords (like `func`, `let`, `if`) by comparing identifiers against a predefined set of reserved words.
- It identifies numeric literals (integers and floats).
- It handles string literals, including support for escape sequences.
- It correctly identifies single and multi-character operators (e.g., `=`, `==`, `+`, `->`).
- It skips over whitespace and comments, as they are not meaningful to the later stages of compilation (other than for separating tokens).

## Error Handling

If the lexer encounters a character or sequence of characters that it cannot recognize (e.g., an unsupported symbol like `@` in the wrong context or an unterminated string), it creates an error token and adds a detailed error message to the `ErrorReporter`. This allows the compilation to continue, potentially finding more lexical errors in the same pass. 