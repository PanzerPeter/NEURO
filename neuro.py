# NEURO Language Main Entry Point

import argparse
import sys
import os
# Add built-in input function for REPL
import builtins 

# Now we can import from other modules within src
try:
    from src.parser import NeuroParser
    from src.interpreter import NeuroInterpreter
    from src.errors import NeuroError # Import base Neuro error
    # Import the TypeChecker
    from src.type_checker import TypeChecker
    # from src.matrix import NeuroMatrix # Import if needed directly here
except ImportError as e:
    print(f"Error importing NEURO components: {e}", file=sys.stderr)
    # Provide helpful path information if imports fail
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print("Ensure you are running from the project root or using the top-level 'neuro.py' script.", file=sys.stderr)
    sys.exit(1)

def start_repl():
    """Starts the NEURO Read-Eval-Print Loop (REPL)."""
    print("NEURO Language REPL (v0.1)")
    print("Type 'exit()' or 'quit()' to exit.")

    # Instantiate components once for the session
    # Assuming use_real should be True for REPL for now
    neuro_parser = NeuroParser(use_real_lexer=True)
    neuro_parser.set_use_real_parser(True)
    type_checker = TypeChecker()
    neuro_interpreter = NeuroInterpreter() # Add environment sharing if needed later

    while True:
        try:
            # Use builtins.input to avoid conflict if user defines 'input'
            line = builtins.input("neuro> ")
            if line.strip().lower() in ["exit()", "quit()"]:
                break
            if not line.strip(): # Skip empty lines
                continue

            # Process the line
            ast_root = neuro_parser.parse(line)
            type_checker.check(ast_root) # Check types for the single line/statement

            if type_checker.errors:
                for error in type_checker.errors:
                    print(f"Type Error: {error}", file=sys.stderr)
                type_checker.errors.clear() # Clear errors for the next input
                continue # Don't interpret if type errors occurred

            # Interpret the AST
            result = neuro_interpreter.interpret(ast_root)
            if result is not None: # Only print if interpret returns something
                 print(repr(result)) # Use repr for clearer output of objects/types

        except NeuroError as e:
            print(e, file=sys.stderr) # Print NEURO-specific errors
            # Clear any potential parser/lexer state if needed (depends on implementation)
            # neuro_parser.reset() # Assuming NeuroParser has a reset method - REMOVED, parse() resets state
            type_checker.errors.clear() # Also clear type errors here
        except EOFError: # Handle Ctrl+D
            print("\nExiting REPL.")
            break
        except Exception as e:
            print(f"Internal Error: {type(e).__name__}: {e}", file=sys.stderr)
            # Consider resetting state here too if applicable
            # neuro_parser.reset() - REMOVED, parse() resets state
            type_checker.errors.clear()

def main():
    """Main function to parse and execute NEURO code."""
    parser = argparse.ArgumentParser(description="NEURO Language Interpreter")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Create the parser for the "run" command
    parser_run = subparsers.add_parser("run", help="Run a .nr file")
    parser_run.add_argument("filepath", help="Path to the .nr file to execute")
    parser_run.add_argument("--use-real", action="store_true",
                           help="Use real implementation instead of placeholder")
    # Add more arguments specific to the 'run' command here if needed

    # Add other subparsers here if needed (e.g., 'compile', 'test')

    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        # If no command, start the REPL
        start_repl()
        sys.exit(0) # Exit cleanly after REPL finishes

    # Handle the 'run' command
    if args.command == "run":
        filepath = args.filepath
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'", file=sys.stderr)
            sys.exit(1)

        if not filepath.endswith('.nr'):
            print(f"Warning: File '{filepath}' does not have a .nr extension.", file=sys.stderr)

        print(f"--- Running NEURO script: {filepath} ---")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            # 1. Parsing
            # Pass the use_real flag from the run command's args
            neuro_parser = NeuroParser(use_real_lexer=args.use_real)
            neuro_parser.set_use_real_parser(args.use_real)
            ast_root = neuro_parser.parse(code)

            # 2. Type Checking (inserted step)
            print("--- Performing Type Checking ---")
            type_checker = TypeChecker()
            type_checker.check(ast_root)

            if type_checker.errors:
                print("\n--- Type Errors Found --- ", file=sys.stderr)
                for error in type_checker.errors:
                    print(f"- {error}", file=sys.stderr)
                print("\nType checking failed. Halting execution.", file=sys.stderr)
                sys.exit(1)
            else:
                print("--- Type Checking Passed ---")

            # 3. Interpretation / Execution (was step 2)
            print("--- Starting Interpretation ---")
            neuro_interpreter = NeuroInterpreter()
            neuro_interpreter.interpret(ast_root) # Pass potentially type-annotated AST

        except NeuroError as e:
            # Catch any NEURO-specific errors (Syntax, Interpreter, Type, etc.)
            print(f"\n--- NEURO Error ---", file=sys.stderr)
            print(e, file=sys.stderr) # Use the custom __str__ method from NeuroError
            sys.exit(1)
        except FileNotFoundError: # Catching standard Python errors is still good
            print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            # Catch any other unexpected errors
            print(f"\n--- Unexpected Internal Error ---", file=sys.stderr)
            print(f"An unexpected error occurred: {type(e).__name__}: {e}", file=sys.stderr)
            # Consider adding traceback here for debugging internal errors
            # import traceback
            # traceback.print_exc()
            sys.exit(1)

        print(f"--- Finished NEURO script: {filepath} ---")

# This allows running the script directly, e.g., python src/neuro.py examples/minimal.nr
# Although using the top-level `neuro.py` is preferred.
if __name__ == "__main__":
    main() 