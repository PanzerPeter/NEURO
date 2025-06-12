#!/usr/bin/env python3
"""
NEURO Compiler (neurc) - Main Entry Point

The NEURO compiler transforms .nr source files into optimized native code
through multiple compilation phases:
1. Lexical Analysis (Tokenization)
2. Parsing (AST Generation)
3. Type Checking & Inference
4. IR Generation
5. Optimization
6. Code Generation (LLVM)

Usage:
    neurc <source_file.nr> [options]
    neurc --help
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .lexer import NeuroLexer
from .parser import NeuroParser
from .ast_nodes import Program
from .compiler import NeuroCompiler
from .errors import NeuroError


def setup_argument_parser() -> argparse.ArgumentParser:
    """Configure command line argument parsing for the NEURO compiler."""
    parser = argparse.ArgumentParser(
        prog='neurc',
        description='NEURO Programming Language Compiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neurc hello.nr                    # Compile to executable
  neurc model.nr -O2                # Compile with optimizations
  neurc --emit-ast program.nr       # Show AST for debugging
  neurc --emit-llvm-ir program.nr   # Show LLVM IR
        """
    )
    
    # Input file
    parser.add_argument(
        'source_file',
        type=str,
        help='NEURO source file (.nr) to compile'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file name (default: inferred from input)'
    )
    
    # Optimization levels
    parser.add_argument(
        '-O', '--optimize',
        choices=['0', '1', '2', '3'],
        default='1',
        help='Optimization level (default: 1)'
    )
    
    # Debug and development options
    parser.add_argument(
        '--emit-ast',
        action='store_true',
        help='Print the Abstract Syntax Tree and exit'
    )
    
    parser.add_argument(
        '--emit-tokens',
        action='store_true',
        help='Print lexer tokens and exit'
    )
    
    parser.add_argument(
        '--emit-llvm-ir',
        action='store_true',
        help='Print LLVM IR and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (implies verbose) for detailed internal logging and tracebacks'
    )
    
    return parser


def compile_file(source_path: Path, args: argparse.Namespace) -> int:
    """
    Compile a single NEURO source file through all compilation phases.
    
    Args:
        source_path: Path to the .nr source file
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Read source code
        if args.verbose:
            print(f"Reading source file: {source_path}")
        
        with open(source_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # If debug mode is on, ensure verbose is also on
        if args.debug:
            args.verbose = True
        
        # Phase 1: Lexical Analysis
        if args.verbose:
            print("Phase 1: Lexical Analysis")
        
        lexer = NeuroLexer()
        tokens = lexer.tokenize(source_code, str(source_path))
        
        if args.emit_tokens:
            print("=== TOKENS ===")
            for token in tokens:
                print(f"{token.type.name:<15} {token.value!r:<20} at {token.line}:{token.column}")
            return 0
        
        # Phase 2: Parsing
        if args.verbose:
            print("Phase 2: Parsing")
        
        parser = NeuroParser()
        ast = parser.parse(tokens)
        
        if args.emit_ast:
            print("=== ABSTRACT SYNTAX TREE ===")
            print(ast.pretty_print())
            return 0
        
        # Phase 3+: Full Compilation
        if args.verbose:
            print("Phase 3+: Type Checking, IR Generation, and Code Generation")
        
        compiler = NeuroCompiler(
            optimization_level=int(args.optimize),
            verbose=args.verbose,
            debug=args.debug
        )
        
        # Determine output file name
        output_path = args.output
        if output_path is None:
            output_path = str(source_path.with_suffix('.exe' if sys.platform == 'win32' else ''))
        
        # Compile to executable
        compiler.compile(ast, output_path, emit_llvm_ir=args.emit_llvm_ir)
        
        # This message is now conditional on not emitting IR, as the function returns early
        if not args.emit_llvm_ir:
            print(f"Compilation successful: {output_path}")
        
        return 0
        
    except NeuroError as e:
        # Provide a cleaner error message for compilation phases
        print(f"Error: {e.message}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"Error: File not found: {source_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Internal compiler error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        elif args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for the NEURO compiler."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate input file
    source_path = Path(args.source_file)
    if not source_path.exists():
        print(f"Error: File does not exist: {source_path}", file=sys.stderr)
        return 1
    
    if source_path.suffix != '.nr':
        print(f"Error: Expected .nr file, got: {source_path.suffix}", file=sys.stderr)
        return 1
    
    return compile_file(source_path, args)


if __name__ == '__main__':
    sys.exit(main()) 