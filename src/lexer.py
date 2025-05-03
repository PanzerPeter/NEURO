# NEURO Lexer: Converts .nr code into a stream of tokens.
import re
from src.errors import NeuroSyntaxError

class Token:
    """Represents a single token identified by the lexer."""
    def __init__(self, type, value, line, column):
        self.type = type      # Token type (e.g., 'IDENTIFIER', 'NUMBER', 'LBRACE')
        self.value = value    # Token value (e.g., 'model', 10, '{')
        self.line = line      # Line number where the token starts
        self.column = column  # Column number where the token starts

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, L{self.line}, C{self.column})"

class NeuroLexer:
    def __init__(self):
        # Token specification for the NEURO language
        self.token_specs = [
            # Comments and whitespace
            ('COMMENT',     r'#.*'),                          # Comments start with # and go to end of line
            ('WHITESPACE',  r'[ \t\r\n]+'),                   # Whitespace (space, tab, newline)
            
            # Keywords - will be checked after parsing identifiers
            # Keywords would go here if we had any reserved words
            
            # Literals
            ('NUMBER',      r'(?:0|[1-9]\d*)(?:\.\d+)?'),      # Integers and floats
            ('STRING',      r'"(?:[^"\\]|\\.)*"'),             # Double-quoted strings
            
            # Operators, delimiters and punctuation
            ('ASSIGN',      r'='),                             # Assignment operator
            ('LPAREN',      r'\('),                            # Left parenthesis
            ('RPAREN',      r'\)'),                            # Right parenthesis
            ('LBRACE',      r'\{'),                            # Left brace
            ('RBRACE',      r'\}'),                            # Right brace
            ('COMMA',       r','),                             # Comma
            ('SEMICOLON',   r';'),                             # Semicolon
            ('DOT',         r'\.'),                            # Dot for method calls
            
            # Identifiers (must come after keywords to ensure keywords are recognized first)
            ('IDENTIFIER',  r'[a-zA-Z_]\w*'),                  # Identifiers
            
            # Invalid token (anything not recognized)
            ('INVALID',     r'.'),                             # Catch-all for invalid characters
        ]
        
        # Combine all token specifications into a single regex pattern
        self.token_regex = '|'.join('(?P<%s>%s)' % (name, pattern) for name, pattern in self.token_specs)
        self.token_regex = re.compile(self.token_regex)
        
        # Keywords to check for after identifying identifiers
        self.keywords = {
            # Reserved keywords would go here if we had any
            # 'if': 'IF',
            # 'else': 'ELSE',
            # etc.
        }
        
        self._use_placeholder = True

    def tokenize(self, code, use_placeholder=None):
        """
        Tokenizes the NEURO code string and yields Tokens.
        
        Args:
            code (str): The NEURO code to tokenize
            use_placeholder (bool, optional): Whether to use the placeholder implementation
                If None, uses the default set in the class.
        
        Returns:
            Generator yielding Token objects
        """
        # Determine whether to use placeholder or real implementation
        if use_placeholder is None:
            use_placeholder = self._use_placeholder
            
        if use_placeholder:
            # Use the placeholder implementation
            print("Tokenizing code... (placeholder)")
            # Placeholder: In a real implementation, this would iterate through the code,
            # match patterns (using regex or manually), and yield Token objects.
            # For now, yield a few dummy tokens representing a simple assignment.
            # Example: model = NeuralNetwork
            yield Token('IDENTIFIER', 'model', 1, 1)
            yield Token('ASSIGN', '=', 1, 7)
            yield Token('IDENTIFIER', 'NeuralNetwork', 1, 9)
            yield Token('LPAREN', '(', 1, 22) # Dummy position
            # ... more tokens ...
            yield Token('RPAREN', ')', 1, 50) # Dummy position
            yield Token('LBRACE', '{', 1, 52) # Dummy position
            # ... tokens for layers ...
            yield Token('RBRACE', '}', 5, 1)   # Dummy position
            print("Tokenization complete (dummy tokens yielded).")
        else:
            # Use the real implementation
            yield from self.tokenize_real(code)

    def tokenize_real(self, code):
        """Tokenizes the NEURO code string using the actual token specifications."""
        line_num = 1
        line_start = 0
        
        # Add a newline at the end to help with parsing the last token
        if not code.endswith('\n'):
            code += '\n'
            
        tokens = []
        for mo in self.token_regex.finditer(code):
            # Get the token type and value
            token_type = mo.lastgroup
            token_value = mo.group()
            column = mo.start() - line_start + 1
            
            # Skip comments and whitespace
            if token_type == 'COMMENT' or token_type == 'WHITESPACE':
                # Update line number and line_start if we encounter a newline
                newlines = token_value.count('\n')
                if newlines > 0:
                    line_num += newlines
                    line_start = mo.start() + token_value.rindex('\n') + 1
                continue
                
            # Check for invalid tokens
            if token_type == 'INVALID':
                error_msg = f"Unexpected character: '{token_value}'"
                raise NeuroSyntaxError(error_msg, line_num, column)
                
            # Check if an identifier is actually a keyword
            if token_type == 'IDENTIFIER' and token_value in self.keywords:
                token_type = self.keywords[token_value]
                
            # Process literals
            if token_type == 'NUMBER':
                # Convert to int or float as appropriate
                if '.' in token_value:
                    token_value = float(token_value)
                else:
                    token_value = int(token_value)
            elif token_type == 'STRING':
                # Remove the quotes and process escape sequences
                token_value = token_value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
                
            # Create and yield the token
            token = Token(token_type, token_value, line_num, column)
            tokens.append(token)
            
            # Update line number and line_start if we encounter a newline in a token
            # (shouldn't happen in most cases, but included for completeness)
            if isinstance(token_value, str) and '\n' in token_value:
                newlines = token_value.count('\n')
                line_num += newlines
                line_start = mo.start() + token_value.rindex('\n') + 1
        
        # Add an EOF token at the end
        tokens.append(Token('EOF', None, line_num, 1))
        
        # Return all tokens at once
        for token in tokens:
            yield token
            
    def set_use_real_implementation(self, use_real=True):
        """
        Sets whether to use the real implementation or placeholder by default.
        
        Args:
            use_real (bool): If True, uses the real implementation by default.
                             If False, uses the placeholder implementation.
        """
        self._use_placeholder = not use_real

# Example usage (for testing)
if __name__ == '__main__':
    test_code = """
    model = NeuralNetwork(input_size=2, output_size=1) {
        Dense(units=64, activation="relu");
    }
    """
    lexer = NeuroLexer()
    print("Using placeholder implementation:")
    for token in lexer.tokenize(test_code, use_placeholder=True):
        print(token)
        
    print("\nUsing real implementation:")
    for token in lexer.tokenize(test_code, use_placeholder=False):
        print(token) 