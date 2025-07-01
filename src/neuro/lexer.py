"""
Lexical Analysis Slice
Tokenizes NEURO source code into a stream of tokens.
"""

import re
from enum import Enum, auto
from typing import List, Optional, Iterator, NamedTuple
from dataclasses import dataclass

from .errors import SourceLocation, ErrorReporter, NeuroSyntaxError


class TokenType(Enum):
    """Token types for the NEURO language."""
    
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers and keywords
    IDENTIFIER = auto()
    
    # Keywords
    FUNC = auto()
    LET = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    STRUCT = auto()
    TRUE = auto()
    FALSE = auto()
    
    # Types
    INT_TYPE = auto()
    FLOAT_TYPE = auto()
    STRING_TYPE = auto()
    BOOL_TYPE = auto()
    TENSOR = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    MATRIX_MULT = auto()  # @
    
    # Comparison
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_THAN = auto()
    GREATER_EQUAL = auto()
    
    # Assignment
    ASSIGN = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Punctuation
    SEMICOLON = auto()
    COLON = auto()
    COMMA = auto()
    DOT = auto()
    ARROW = auto()  # ->
    
    # Brackets
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LESS_BRACKET = auto()  # <
    GREATER_BRACKET = auto()  # >
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()


@dataclass
class Token:
    """Represents a token in the source code."""
    type: TokenType
    value: str
    location: SourceLocation
    
    def __str__(self) -> str:
        return f"Token({self.type.name}, '{self.value}', {self.location})"


class NeuroLexer:
    """Lexical analyzer for the NEURO programming language."""
    
    # Keywords mapping
    KEYWORDS = {
        'func': TokenType.FUNC,
        'let': TokenType.LET,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'return': TokenType.RETURN,
        'struct': TokenType.STRUCT,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'int': TokenType.INT_TYPE,
        'float': TokenType.FLOAT_TYPE,
        'string': TokenType.STRING_TYPE,
        'bool': TokenType.BOOL_TYPE,
        'Tensor': TokenType.TENSOR,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
    }
    
    # Two-character operators
    TWO_CHAR_OPERATORS = {
        '->': TokenType.ARROW,
        '==': TokenType.EQUAL,
        '!=': TokenType.NOT_EQUAL,
        '<=': TokenType.LESS_EQUAL,
        '>=': TokenType.GREATER_EQUAL,
        '&&': TokenType.AND,
        '||': TokenType.OR,
    }
    
    # Single-character operators and punctuation
    SINGLE_CHAR_TOKENS = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.MULTIPLY,
        '/': TokenType.DIVIDE,
        '%': TokenType.MODULO,
        '@': TokenType.MATRIX_MULT,
        '=': TokenType.ASSIGN,
        '<': TokenType.LESS_THAN,
        '>': TokenType.GREATER_THAN,
        '!': TokenType.NOT,
        ';': TokenType.SEMICOLON,
        ':': TokenType.COLON,
        ',': TokenType.COMMA,
        '.': TokenType.DOT,
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE,
        '[': TokenType.LBRACKET,
        ']': TokenType.RBRACKET,
    }
    
    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.position = 0
        self.line = 1
        self.column = 1
        self.error_reporter = ErrorReporter()
    
    def current_char(self) -> Optional[str]:
        """Get the current character."""
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at a character ahead."""
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> None:
        """Move to the next character."""
        if self.position < len(self.source):
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1
    
    def current_location(self) -> SourceLocation:
        """Get the current source location."""
        return SourceLocation(self.filename, self.line, self.column)
    
    def skip_whitespace(self) -> None:
        """Skip whitespace characters except newlines."""
        while (self.current_char() is not None and 
               self.current_char() in ' \t\r'):
            self.advance()
    
    def read_string(self) -> str:
        """Read a string literal."""
        quote_char = self.current_char()
        value = quote_char  # Start with the opening quote
        self.advance()  # Skip opening quote
        
        while self.current_char() is not None and self.current_char() != quote_char:
            if self.current_char() == '\\':
                value += self.current_char()  # Add backslash
                self.advance()
                if self.current_char() is None:
                    break
                value += self.current_char()  # Add escaped character as-is
            else:
                value += self.current_char()
            self.advance()
        
        if self.current_char() == quote_char:
            value += quote_char  # Add closing quote
            self.advance()  # Skip closing quote
        else:
            self.error_reporter.syntax_error(
                "Unterminated string literal",
                self.current_location()
            )
        
        return value
    
    def read_number(self) -> tuple[str, TokenType]:
        """Read a numeric literal."""
        value = ""
        has_dot = False
        
        while (self.current_char() is not None and 
               (self.current_char().isdigit() or self.current_char() == '.')):
            if self.current_char() == '.':
                if has_dot:
                    break  # Second dot, stop here
                has_dot = True
            value += self.current_char()
            self.advance()
        
        if has_dot:
            return value, TokenType.FLOAT
        else:
            return value, TokenType.INTEGER
    
    def read_identifier(self) -> str:
        """Read an identifier or keyword."""
        value = ""
        while (self.current_char() is not None and 
               (self.current_char().isalnum() or self.current_char() == '_')):
            value += self.current_char()
            self.advance()
        return value
    
    def read_comment(self) -> str:
        """Read a comment."""
        value = ""
        # Skip the //
        self.advance()
        self.advance()
        
        while self.current_char() is not None and self.current_char() != '\n':
            value += self.current_char()
            self.advance()
        
        return value.strip()
    
    def next_token(self) -> Token:
        """Get the next token from the source."""
        self.skip_whitespace()
        
        location = self.current_location()
        char = self.current_char()
        
        if char is None:
            return Token(TokenType.EOF, "", self.current_location())
        
        if char == '\n':
            self.advance()
            return Token(TokenType.NEWLINE, '\n', location)
        
        if char == '/' and self.peek_char() == '/':
            comment = self.read_comment()
            return Token(TokenType.COMMENT, comment, location)
        
        if char in '"\'':
            string_value = self.read_string()
            return Token(TokenType.STRING, string_value, location)
        
        if char.isdigit():
            number_value, token_type = self.read_number()
            return Token(token_type, number_value, location)
        
        if char.isalpha() or char == '_':
            identifier = self.read_identifier()
            token_type = self.KEYWORDS.get(identifier, TokenType.IDENTIFIER)
            return Token(token_type, identifier, location)
        
        two_char = char + (self.peek_char() or '')
        if two_char in self.TWO_CHAR_OPERATORS:
            self.advance()
            self.advance()
            return Token(self.TWO_CHAR_OPERATORS[two_char], two_char, location)
        
        if char in self.SINGLE_CHAR_TOKENS:
            self.advance()
            return Token(self.SINGLE_CHAR_TOKENS[char], char, location)
        
        self.error_reporter.syntax_error(f"Unknown character '{char}'", location)
        self.advance()
        return self.next_token()

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code."""
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens
    
    def get_error_reporter(self) -> ErrorReporter:
        """Get the error reporter."""
        return self.error_reporter 