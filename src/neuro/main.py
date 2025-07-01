"""
NEURO Programming Language Compiler
Command-line interface for the NEURO compiler.
"""

import sys
import argparse
import os
from typing import List, Optional

from .compiler import NeuroCompiler, CompileOptions, OptimizationLevel
from .errors import NeuroError


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='neurc',
        description='NEURO Programming Language Compiler',
        epilog='''Examples:
  neurc hello.nr                 # Compile to executable
  neurc hello.nr -o hello        # Compile with custom output name
  neurc hello.nr --emit-llvm-ir  # Generate LLVM IR
  neurc hello.nr -O3 --verbose   # Compile with aggressive optimization
  neurc hello.nr --emit-tokens   # Show lexer tokens for debugging
'''
    )
    
    # Input file
    parser.add_argument(
        'input',
        help='NEURO source file to compile'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        help='Output file name'
    )
    
    parser.add_argument(
        '--target',
        choices=['cpu', 'cuda', 'opencl', 'metal'],
        default='cpu',
        help='Target platform (default: cpu)'
    )
    
    parser.add_argument(
        '--format',
        choices=['executable', 'llvm-ir', 'assembly'],
        default='executable',
        help='Output format (default: executable)'
    )
    
    # Optimization options
    parser.add_argument(
        '-O0', '--no-optimization',
        action='store_const',
        const=OptimizationLevel.O0,
        dest='optimization',
        help='Disable optimizations'
    )
    
    parser.add_argument(
        '-O1', '--basic-optimization',
        action='store_const',
        const=OptimizationLevel.O1,
        dest='optimization',
        help='Basic optimizations'
    )
    
    parser.add_argument(
        '-O2', '--standard-optimization',
        action='store_const',
        const=OptimizationLevel.O2,
        dest='optimization',
        help='Standard optimizations (default)'
    )
    
    parser.add_argument(
        '-O3', '--aggressive-optimization',
        action='store_const',
        const=OptimizationLevel.O3,
        dest='optimization',
        help='Aggressive optimizations'
    )
    
    # Debug and analysis options
    parser.add_argument(
        '--emit-tokens',
        action='store_true',
        help='Print lexer tokens'
    )
    
    parser.add_argument(
        '--emit-ast',
        action='store_true',
        help='Print abstract syntax tree'
    )
    
    parser.add_argument(
        '--emit-llvm-ir',
        action='store_true',
        help='Print LLVM IR'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Include debug information'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='NEURO Compiler 0.1.0'
    )
    
    # Set default optimization level
    parser.set_defaults(optimization=OptimizationLevel.O2)
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return False
    
    # Check input file has .nr extension
    if not args.input.endswith('.nr'):
        print(f"Warning: Input file '{args.input}' does not have .nr extension", file=sys.stderr)
    
    # Check output directory is writable if output specified
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if not os.access(output_dir, os.W_OK):
            print(f"Error: Cannot write to output directory '{output_dir}'", file=sys.stderr)
            return False
    
    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the NEURO compiler."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Create compile options
        options = CompileOptions(
            optimization_level=args.optimization,
            target_platform=args.target,
            output_format=args.format,
            debug_info=args.debug,
            emit_tokens=args.emit_tokens,
            emit_ast=args.emit_ast,
            emit_llvm_ir=args.emit_llvm_ir,
            verbose=args.verbose,
            output_file=args.output
        )
        
        # Create compiler and compile
        compiler = NeuroCompiler(options)
        result = compiler.compile_file(args.input)
        
        # Print results
        if result.success:
            if args.verbose:
                print(f"Compilation successful!")
                if result.output_file:
                    print(f"Output written to: {result.output_file}")
                print(f"Compilation time: {result.compilation_time:.3f}s")
            
            # Print warnings if any
            if result.warnings:
                print(f"\nWarnings ({len(result.warnings)}):", file=sys.stderr)
                for warning in result.warnings:
                    print(f"  {warning}", file=sys.stderr)
            
            return 0
        else:
            print(f"Compilation failed!", file=sys.stderr)
            
            # Print errors
            if result.errors:
                print(f"\nErrors ({len(result.errors)}):", file=sys.stderr)
                for error in result.errors:
                    print(f"  {error}", file=sys.stderr)
            
            return 1
    
    except KeyboardInterrupt:
        print("\nCompilation interrupted by user", file=sys.stderr)
        return 130
    
    except NeuroError as e:
        print(f"Compiler error: {e}", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"Internal error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_interactive_mode() -> None:
    """Run NEURO compiler in interactive mode."""
    print("NEURO Programming Language Compiler")
    print("Interactive Mode - Type 'exit' to quit")
    print()
    
    while True:
        try:
            # Get input
            line = input("neuro> ").strip()
            
            if line in ['exit', 'quit', 'q']:
                break
            
            if not line:
                continue
            
            if line.startswith('help'):
                print("Available commands:")
                print("  compile <file>  - Compile a NEURO file")
                print("  tokens <file>   - Show tokens for a file")
                print("  ast <file>      - Show AST for a file")
                print("  help            - Show this help")
                print("  exit            - Exit interactive mode")
                continue
            
            parts = line.split()
            command = parts[0]
            
            if command == 'compile' and len(parts) > 1:
                filename = parts[1]
                main([filename, '--verbose'])
            elif command == 'tokens' and len(parts) > 1:
                filename = parts[1]
                main([filename, '--emit-tokens'])
            elif command == 'ast' and len(parts) > 1:
                filename = parts[1]
                main([filename, '--emit-ast'])
            else:
                print(f"Unknown command: {line}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nType 'exit' to quit")
        except EOFError:
            break
    
    print("\nGoodbye!")


if __name__ == '__main__':
    # Check if running interactively
    if len(sys.argv) == 1:
        run_interactive_mode()
    else:
        sys.exit(main()) 