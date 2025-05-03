# NEURO Language Main Entry Point

import argparse
import sys
import os

# Ensure importing from sibling directories works correctly
# Add the parent directory of 'src' (the project root) to sys.path
# This is necessary if running this script directly (e.g., python src/neuro.py)
# instead of using the top-level neuro.py or `python -m src.neuro`
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from other modules within src
try:
    from src.parser import NeuroParser
    from src.interpreter import NeuroInterpreter
    from src.errors import NeuroError # Import base Neuro error
    # from src.matrix import NeuroMatrix # Import if needed directly here
except ImportError as e:
    print(f"Error importing NEURO components: {e}", file=sys.stderr)
    # Provide helpful path information if imports fail
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    print("Ensure you are running from the project root or using the top-level 'neuro.py' script.", file=sys.stderr)
    sys.exit(1)

def main():
    """Main function to parse and execute NEURO code."""
    parser = argparse.ArgumentParser(description="NEURO Language Interpreter")
    parser.add_argument("filepath", help="Path to the .nr file to execute")
    parser.add_argument("--use-real", action="store_true", 
                        help="Use real implementation instead of placeholder")
    # Add more arguments as needed (e.g., verbosity, specific options)

    args = parser.parse_args()

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
        neuro_parser = NeuroParser(use_real_lexer=args.use_real)
        neuro_parser.set_use_real_parser(args.use_real)
        ast_root = neuro_parser.parse(code)

        # 2. Interpretation / Execution
        neuro_interpreter = NeuroInterpreter()
        neuro_interpreter.interpret(ast_root)

    except NeuroError as e:
        # Catch any NEURO-specific errors (Syntax, Interpreter, etc.)
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