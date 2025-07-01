"""
NEURO Programming Language Compiler
A high-performance, AI-focused programming language with type inference.
"""

__version__ = "0.1.0"
__author__ = "NEURO Language Team"

from .lexer import NeuroLexer, Token, TokenType
from .parser import NeuroParser
from .ast_nodes import *
from .type_checker import NeuroTypeChecker
from .compiler import NeuroCompiler, CompileOptions, OptimizationLevel
from .errors import NeuroError, NeuroSyntaxError, NeuroTypeError
from .main import main

__all__ = [
    'NeuroLexer', 'Token', 'TokenType',
    'NeuroParser', 
    'NeuroTypeChecker',
    'NeuroCompiler', 'CompileOptions', 'OptimizationLevel',
    'NeuroError', 'NeuroSyntaxError', 'NeuroTypeError',
    'main'
] 