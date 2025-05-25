"""
NEURO Lexer (Tokenizer)

Converts NEURO source code into a stream of tokens for parsing.
Handles all language constructs including:
- Keywords and identifiers
- Literals (numbers, strings, booleans)
- Operators and punctuation
- Comments and whitespace
- Tensor syntax and AI-specific constructs
"""

import re
from enum import Enum, auto
from typing import List, Optional, Iterator, NamedTuple, Union
from dataclasses import dataclass

from .errors import LexerError, SourceLocation


class TokenType(Enum):
    """Enumeration of all token types in the NEURO language."""
    
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers and keywords
    IDENTIFIER = auto()
    
    # Keywords - Control Flow
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    MATCH = auto()
    CASE = auto()
    
    # Keywords - Declarations
    LET = auto()
    FUNC = auto()
    STRUCT = auto()
    ENUM = auto()
    IMPL = auto()
    TRAIT = auto()
    IMPORT = auto()
    
    # Keywords - Types
    INT = auto()
    FLOAT_TYPE = auto()
    BOOL = auto()
    STRING_TYPE = auto()
    TENSOR = auto()
    
    # Keywords - AI/ML specific
    MODEL = auto()
    LAYER = auto()
    NEURALNETWORK = auto()
    DATASET = auto()
    OPTIMIZER = auto()
    LOSS = auto()
    ACTIVATION = auto()
    
    # Keywords - Memory and concurrency
    REF = auto()
    MUT = auto()
    ASYNC = auto()
    AWAIT = auto()
    
    # Keywords - GPU/Meta-programming
    GPU = auto()
    KERNEL = auto()
    DEVICE = auto()
    
    # Operators - Arithmetic
    PLUS = auto()          # +
    MINUS = auto()         # -
    MULTIPLY = auto()      # *
    DIVIDE = auto()        # /
    MODULO = auto()        # %
    POWER = auto()         # **
    
    # Operators - Comparison
    EQUAL = auto()         # ==
    NOT_EQUAL = auto()     # !=
    LESS = auto()          # <
    LESS_EQUAL = auto()    # <=
    GREATER = auto()       # >
    GREATER_EQUAL = auto() # >=
    
    # Operators - Logical
    AND = auto()           # &&
    OR = auto()            # ||
    NOT = auto()           # !
    
    # Operators - Bitwise
    BIT_AND = auto()       # &
    BIT_OR = auto()        # |
    BIT_XOR = auto()       # ^
    BIT_NOT = auto()       # ~
    BIT_LSHIFT = auto()    # <<
    BIT_RSHIFT = auto()    # >>
    
    # Operators - Assignment
    ASSIGN = auto()        # =
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    MULTIPLY_ASSIGN = auto() # *=
    DIVIDE_ASSIGN = auto() # /=
    
    # Operators - Matrix/Tensor
    MATRIX_MUL = auto()    # @
    DOT = auto()           # .
    RANGE = auto()         # ..
    RANGE_INCLUSIVE = auto() # ..=
    
    # Punctuation
    SEMICOLON = auto()     # ;
    COMMA = auto()         # ,
    COLON = auto()         # :
    DOUBLE_COLON = auto()  # ::
    ARROW = auto()         # ->
    FAT_ARROW = auto()     # =>
    QUESTION = auto()      # ?
    
    # Brackets
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LANGLE = auto()        # <
    RANGLE = auto()        # >
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()


@dataclass
class Token:
    """Represents a single token in the source code."""
    type: TokenType
    value: str
    line: int
    column: int
    filename: str = ""
    
    @property
    def location(self) -> SourceLocation:
        """Get the source location of this token."""
        return SourceLocation(self.filename, self.line, self.column)


class NeuroLexer:
    """
    Lexical analyzer for the NEURO programming language.
    
    Converts source code text into a sequence of tokens that can be
    parsed into an Abstract Syntax Tree (AST).
    """
    
    # Keywords mapping
    KEYWORDS = {
        # Control flow
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'elif': TokenType.ELIF,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'return': TokenType.RETURN,
        'match': TokenType.MATCH,
        'case': TokenType.CASE,
        
        # Declarations
        'let': TokenType.LET,
        'func': TokenType.FUNC,
        'struct': TokenType.STRUCT,
        'enum': TokenType.ENUM,
        'impl': TokenType.IMPL,
        'trait': TokenType.TRAIT,
        'import': TokenType.IMPORT,
        
        # Types
        'int': TokenType.INT,
        'float': TokenType.FLOAT_TYPE,
        'bool': TokenType.BOOL,
        'string': TokenType.STRING_TYPE,
        'tensor': TokenType.TENSOR,
        
        # Literals
        'true': TokenType.BOOLEAN,
        'false': TokenType.BOOLEAN,
        
        # AI/ML specific
        'model': TokenType.MODEL,
        'layer': TokenType.LAYER,
        'NeuralNetwork': TokenType.NEURALNETWORK,
        'Dataset': TokenType.DATASET,
        'optimizer': TokenType.OPTIMIZER,
        'loss': TokenType.LOSS,
        'activation': TokenType.ACTIVATION,
        
        # Memory and concurrency
        'ref': TokenType.REF,
        'mut': TokenType.MUT,
        'async': TokenType.ASYNC,
        'await': TokenType.AWAIT,
        
        # GPU/Meta-programming
        'gpu': TokenType.GPU,
        'kernel': TokenType.KERNEL,
        'device': TokenType.DEVICE,
    }
    
    # Operator patterns (order matters for longest match)
    OPERATORS = [
        ('==', TokenType.EQUAL),
        ('!=', TokenType.NOT_EQUAL),
        ('<=', TokenType.LESS_EQUAL),
        ('>=', TokenType.GREATER_EQUAL),
        ('&&', TokenType.AND),
        ('||', TokenType.OR),
        ('<<', TokenType.BIT_LSHIFT),
        ('>>', TokenType.BIT_RSHIFT),
        ('+=', TokenType.PLUS_ASSIGN),
        ('-=', TokenType.MINUS_ASSIGN),
        ('*=', TokenType.MULTIPLY_ASSIGN),
        ('/=', TokenType.DIVIDE_ASSIGN),
        ('->', TokenType.ARROW),
        ('=>', TokenType.FAT_ARROW),
        ('::', TokenType.DOUBLE_COLON),
        ('..=', TokenType.RANGE_INCLUSIVE),
        ('..', TokenType.RANGE),
        ('**', TokenType.POWER),
        ('+', TokenType.PLUS),
        ('-', TokenType.MINUS),
        ('*', TokenType.MULTIPLY),
        ('/', TokenType.DIVIDE),
        ('%', TokenType.MODULO),
        ('@', TokenType.MATRIX_MUL),
        ('<', TokenType.LESS),
        ('>', TokenType.GREATER),
        ('=', TokenType.ASSIGN),
        ('!', TokenType.NOT),
        ('&', TokenType.BIT_AND),
        ('|', TokenType.BIT_OR),
        ('^', TokenType.BIT_XOR),
        ('~', TokenType.BIT_NOT),
        ('.', TokenType.DOT),
        (';', TokenType.SEMICOLON),
        (',', TokenType.COMMA),
        (':', TokenType.COLON),
        ('?', TokenType.QUESTION),
        ('(', TokenType.LPAREN),
        (')', TokenType.RPAREN),
        ('[', TokenType.LBRACKET),
        (']', TokenType.RBRACKET),
        ('{', TokenType.LBRACE),
        ('}', TokenType.RBRACE),
    ]
    
    def __init__(self):
        self.text = ""
        self.pos = 0
        self.line = 1
        self.column = 1
        self.filename = ""
    
    def tokenize(self, text: str, filename: str = "") -> List[Token]:
        """
        Tokenize the input text into a list of tokens.
        
        Args:
            text: Source code to tokenize
            filename: Optional filename for error reporting
            
        Returns:
            List of tokens
            
        Raises:
            LexerError: If invalid syntax is encountered
        """
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.filename = filename
        
        tokens = []
        
        while self.pos < len(self.text):
            # Skip whitespace (except newlines)
            if self.current_char() in ' \t\r':
                self.advance()
                continue
            
            # Handle newlines
            if self.current_char() == '\n':
                tokens.append(self.make_token(TokenType.NEWLINE, '\n'))
                self.advance_newline()
                continue
            
            # Handle comments
            if self.current_char() == '/' and self.peek() == '/':
                self.skip_line_comment()
                continue
            
            if self.current_char() == '/' and self.peek() == '*':
                self.skip_block_comment()
                continue
            
            # Handle string literals
            if self.current_char() in '"\'':
                tokens.append(self.read_string())
                continue
            
            # Handle numeric literals
            if self.current_char().isdigit():
                tokens.append(self.read_number())
                continue
            
            # Handle identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                tokens.append(self.read_identifier())
                continue
            
            # Handle operators and punctuation
            operator_token = self.read_operator()
            if operator_token:
                tokens.append(operator_token)
                continue
            
            # Unknown character
            raise LexerError(
                f"Unexpected character",
                SourceLocation(self.filename, self.line, self.column),
                self.current_char()
            )
        
        # Add EOF token
        tokens.append(self.make_token(TokenType.EOF, ""))
        return tokens
    
    def current_char(self) -> str:
        """Get the current character, or empty string if at end."""
        if self.pos >= len(self.text):
            return ''
        return self.text[self.pos]
    
    def peek(self, offset: int = 1) -> str:
        """Peek at character ahead by offset, or empty string if past end."""
        peek_pos = self.pos + offset
        if peek_pos >= len(self.text):
            return ''
        return self.text[peek_pos]
    
    def advance(self) -> None:
        """Move to the next character."""
        if self.pos < len(self.text):
            self.pos += 1
            self.column += 1
    
    def advance_newline(self) -> None:
        """Move to the next character, handling newline."""
        self.pos += 1
        self.line += 1
        self.column = 1
    
    def make_token(self, token_type: TokenType, value: str) -> Token:
        """Create a token at the current position."""
        return Token(token_type, value, self.line, self.column - len(value), self.filename)
    
    def read_string(self) -> Token:
        """Read a string literal."""
        quote_char = self.current_char()
        start_column = self.column
        value = ""
        self.advance()  # Skip opening quote
        
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                if not self.current_char():
                    raise LexerError(
                        "Unterminated string literal",
                        SourceLocation(self.filename, self.line, start_column)
                    )
                
                # Handle escape sequences
                escape_char = self.current_char()
                if escape_char == 'n':
                    value += '\n'
                elif escape_char == 't':
                    value += '\t'
                elif escape_char == 'r':
                    value += '\r'
                elif escape_char == '\\':
                    value += '\\'
                elif escape_char == quote_char:
                    value += quote_char
                else:
                    value += escape_char
                
                self.advance()
            elif self.current_char() == '\n':
                # Multi-line strings are allowed
                value += self.current_char()
                self.advance_newline()
            else:
                value += self.current_char()
                self.advance()
        
        if not self.current_char():
            raise LexerError(
                "Unterminated string literal",
                SourceLocation(self.filename, self.line, start_column)
            )
        
        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, value, self.line, start_column, self.filename)
    
    def read_number(self) -> Token:
        """Read a numeric literal (integer or float)."""
        start_column = self.column
        value = ""
        has_dot = False
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            if self.current_char() == '.':
                # Check for range operator (..)
                if self.peek() == '.':
                    break
                
                if has_dot:
                    raise LexerError(
                        "Invalid number format",
                        SourceLocation(self.filename, self.line, self.column)
                    )
                has_dot = True
            
            value += self.current_char()
            self.advance()
        
        # Handle scientific notation (e.g., 1.5e-10)
        if self.current_char() and self.current_char().lower() == 'e':
            has_dot = True  # Scientific notation makes it a float
            value += self.current_char()
            self.advance()
            
            if self.current_char() in '+-':
                value += self.current_char()
                self.advance()
            
            if not self.current_char() or not self.current_char().isdigit():
                raise LexerError(
                    "Invalid scientific notation",
                    SourceLocation(self.filename, self.line, self.column)
                )
            
            while self.current_char() and self.current_char().isdigit():
                value += self.current_char()
                self.advance()
        
        token_type = TokenType.FLOAT if has_dot else TokenType.INTEGER
        return Token(token_type, value, self.line, start_column, self.filename)
    
    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_column = self.column
        value = ""
        
        while (self.current_char() and 
               (self.current_char().isalnum() or self.current_char() == '_')):
            value += self.current_char()
            self.advance()
        
        # Check if it's a keyword
        token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, self.line, start_column, self.filename)
    
    def read_operator(self) -> Optional[Token]:
        """Read an operator or punctuation."""
        start_column = self.column
        
        # Try to match operators (longest first)
        for op_text, op_type in self.OPERATORS:
            if self.text[self.pos:].startswith(op_text):
                # Advance by operator length
                for _ in range(len(op_text)):
                    self.advance()
                return Token(op_type, op_text, self.line, start_column, self.filename)
        
        return None
    
    def skip_line_comment(self) -> None:
        """Skip a line comment (//)."""
        while self.current_char() and self.current_char() != '\n':
            self.advance()
    
    def skip_block_comment(self) -> None:
        """Skip a block comment (/* */)."""
        self.advance()  # Skip '/'
        self.advance()  # Skip '*'
        
        while self.current_char():
            if self.current_char() == '*' and self.peek() == '/':
                self.advance()  # Skip '*'
                self.advance()  # Skip '/'
                return
            
            if self.current_char() == '\n':
                self.advance_newline()
            else:
                self.advance()
        
        raise LexerError(
            "Unterminated block comment",
            SourceLocation(self.filename, self.line, self.column)
        ) 