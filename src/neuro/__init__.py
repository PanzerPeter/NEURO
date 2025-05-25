"""
NEURO Programming Language Implementation

A high-performance, compiled programming language for AI development.
This package contains the complete compiler implementation including:
- Lexical analysis and parsing
- Type checking and inference  
- LLVM code generation
- Standard library
"""

__version__ = "0.1.0"
__author__ = "NEURO Language Team"

# Core compiler components
from .lexer import NeuroLexer
from .parser import NeuroParser
from .compiler import NeuroCompiler
from .errors import NeuroError

__all__ = [
    'NeuroLexer',
    'NeuroParser', 
    'NeuroCompiler',
    'NeuroError',
    '__version__'
] 