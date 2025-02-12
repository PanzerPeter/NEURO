#!/usr/bin/env python3
import sys
import argparse
from .interpreter import NeuroInterpreter
from .syntax_highlighter import highlight_code

def main():
    parser = argparse.ArgumentParser(description='NEURO Programming Language Interpreter')
    parser.add_argument('file', help='NEURO source file (.nr)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-color', action='store_true', help='Disable syntax highlighting')
    args = parser.parse_args()

    try:
        # Read the source file
        with open(args.file, 'r') as f:
            source = f.read()

        # Display the code with syntax highlighting
        if not args.no_color:
            print("\nSource code:")
            print(highlight_code(source))
            print("\nExecution output:")

        # Create interpreter and execute the code
        interpreter = NeuroInterpreter()
        
        if args.verbose:
            print(f"Executing {args.file}...")
        
        result = interpreter.interpret(source)
        
        if args.verbose:
            print(f"Execution completed. Result: {result}")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 