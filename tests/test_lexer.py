"""
Tests for the NEURO lexer.
"""

import pytest
from src.neuro.lexer import NeuroLexer, TokenType, Token
from src.neuro.errors import NeuroSyntaxError


class TestNeuroLexer:
    """Test cases for the NEURO lexer."""
    
    def test_basic_tokens(self):
        """Test basic token recognition."""
        source = "func let if else for in return struct"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.FUNC, TokenType.LET, TokenType.IF, TokenType.ELSE,
            TokenType.FOR, TokenType.IN, TokenType.RETURN, TokenType.STRUCT,
            TokenType.EOF
        ]
        
        assert len(tokens) == len(expected_types)
        for token, expected_type in zip(tokens, expected_types):
            assert token.type == expected_type
    
    def test_literals(self):
        """Test literal token recognition."""
        source = '42 3.14 "hello world" true false'
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == "42"
        
        assert tokens[1].type == TokenType.FLOAT
        assert tokens[1].value == "3.14"
        
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == '"hello world"'
        
        assert tokens[3].type == TokenType.TRUE
        assert tokens[3].value == "true"
        
        assert tokens[4].type == TokenType.FALSE
        assert tokens[4].value == "false"
    
    def test_identifiers(self):
        """Test identifier recognition."""
        source = "variable_name myVar someFunc123 _private"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        expected_names = ["variable_name", "myVar", "someFunc123", "_private"]
        
        for i, expected_name in enumerate(expected_names):
            assert tokens[i].type == TokenType.IDENTIFIER
            assert tokens[i].value == expected_name
    
    def test_operators(self):
        """Test operator recognition."""
        source = "+ - * / % @ = == != < <= > >= && || !"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.DIVIDE,
            TokenType.MODULO, TokenType.MATRIX_MULT, TokenType.ASSIGN,
            TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.LESS_THAN,
            TokenType.LESS_EQUAL, TokenType.GREATER_THAN, TokenType.GREATER_EQUAL,
            TokenType.AND, TokenType.OR, TokenType.NOT, TokenType.EOF
        ]
        
        for token, expected_type in zip(tokens, expected_types):
            assert token.type == expected_type
    
    def test_punctuation(self):
        """Test punctuation recognition."""
        source = "; : , . -> ( ) { } [ ]"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.SEMICOLON, TokenType.COLON, TokenType.COMMA, TokenType.DOT,
            TokenType.ARROW, TokenType.LPAREN, TokenType.RPAREN, TokenType.LBRACE,
            TokenType.RBRACE, TokenType.LBRACKET, TokenType.RBRACKET, TokenType.EOF
        ]
        
        for token, expected_type in zip(tokens, expected_types):
            assert token.type == expected_type
    
    def test_type_annotations(self):
        """Test type annotation keywords."""
        source = "int float string bool Tensor"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        expected_types = [
            TokenType.INT_TYPE, TokenType.FLOAT_TYPE, TokenType.STRING_TYPE,
            TokenType.BOOL_TYPE, TokenType.TENSOR, TokenType.EOF
        ]
        
        for token, expected_type in zip(tokens, expected_types):
            assert token.type == expected_type
    
    def test_comments(self):
        """Test comment handling."""
        source = """
        // This is a comment
        let x = 42  // Another comment
        /* Multi-line comment
           spanning multiple lines */
        let y = 3.14
        """
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Comments should be tokenized but can be filtered out
        non_comment_tokens = [t for t in tokens if t.type not in (TokenType.COMMENT, TokenType.NEWLINE)]
        
        assert non_comment_tokens[0].type == TokenType.LET
        assert non_comment_tokens[1].type == TokenType.IDENTIFIER
        assert non_comment_tokens[1].value == "x"
    
    def test_strings_with_escapes(self):
        """Test string literals with escape sequences."""
        source = r'"hello\nworld" "quote: \"test\"" "backslash: \\"'
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        assert tokens[0].type == TokenType.STRING
        assert tokens[1].type == TokenType.STRING
        assert tokens[2].type == TokenType.STRING
    
    def test_function_definition(self):
        """Test tokenizing a complete function definition."""
        source = """
        func add(a: int, b: int) -> int {
            return a + b
        }
        """
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Filter out newlines and comments for easier testing
        tokens = [t for t in tokens if t.type not in (TokenType.NEWLINE, TokenType.COMMENT)]
        
        expected_sequence = [
            TokenType.FUNC, TokenType.IDENTIFIER, TokenType.LPAREN,
            TokenType.IDENTIFIER, TokenType.COLON, TokenType.INT_TYPE,
            TokenType.COMMA, TokenType.IDENTIFIER, TokenType.COLON,
            TokenType.INT_TYPE, TokenType.RPAREN, TokenType.ARROW,
            TokenType.INT_TYPE, TokenType.LBRACE, TokenType.RETURN,
            TokenType.IDENTIFIER, TokenType.PLUS, TokenType.IDENTIFIER,
            TokenType.RBRACE, TokenType.EOF
        ]
        
        assert len(tokens) == len(expected_sequence)
        for token, expected_type in zip(tokens, expected_sequence):
            assert token.type == expected_type
    
    def test_tensor_operations(self):
        """Test tokenizing tensor operations."""
        source = """
        let a: Tensor<float, (3, 3)> = [[1.0, 2.0], [3.0, 4.0]]
        let result = a @ b
        """
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Find the matrix multiply operator
        matrix_mult_token = next(t for t in tokens if t.type == TokenType.MATRIX_MULT)
        assert matrix_mult_token.value == "@"
    
    def test_neural_network_syntax(self):
        """Test tokenizing neural network syntax."""
        source = """
        let model = NeuralNetwork<f32, (784, 128, 10)> {
            dense_layer<128>(.relu),
            batch_norm(),
            dropout(0.2)
        }
        """
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Should successfully tokenize without errors
        assert not lexer.get_error_reporter().has_errors()
        
        # Check for specific tokens
        neural_network_token = next(t for t in tokens if t.value == "NeuralNetwork")
        assert neural_network_token.type == TokenType.IDENTIFIER
    
    def test_generic_syntax(self):
        """Test tokenizing generic type syntax."""
        source = "Point<T> Vector<float> Map<string, int>"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Should have proper sequence of < and > tokens
        angle_brackets = [t for t in tokens if t.type in (TokenType.LESS_THAN, TokenType.GREATER_THAN)]
        assert len(angle_brackets) == 6  # 3 pairs of < >
    
    def test_error_handling(self):
        """Test lexer error handling."""
        source = 'let x = "unterminated string'
        lexer = NeuroLexer(source)
        
        # Should handle unterminated string gracefully
        tokens = lexer.tokenize()
        assert lexer.get_error_reporter().has_errors()
    
    def test_location_tracking(self):
        """Test source location tracking."""
        source = """line 1
        line 2
        line 3"""
        lexer = NeuroLexer(source, "test.nr")
        tokens = lexer.tokenize()
        
        # Check that locations are tracked correctly
        identifier_tokens = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert len(identifier_tokens) >= 3
        
        # Check line numbers increase
        for i in range(len(identifier_tokens) - 1):
            assert identifier_tokens[i + 1].location.line >= identifier_tokens[i].location.line
    
    def test_whitespace_handling(self):
        """Test whitespace handling."""
        source = "  \t  let   \t x    =    42   \n  "
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Filter out whitespace tokens
        content_tokens = [t for t in tokens if t.type not in (TokenType.NEWLINE,)]
        
        assert content_tokens[0].type == TokenType.LET
        assert content_tokens[1].type == TokenType.IDENTIFIER
        assert content_tokens[2].type == TokenType.ASSIGN
        assert content_tokens[3].type == TokenType.INTEGER
    
    def test_complex_expression(self):
        """Test tokenizing a complex expression."""
        source = "result = (a + b) * c.field[index] @ matrix"
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        
        # Should tokenize successfully
        assert not lexer.get_error_reporter().has_errors()
        
        # Check for all expected token types
        token_types = [t.type for t in tokens]
        assert TokenType.IDENTIFIER in token_types
        assert TokenType.ASSIGN in token_types
        assert TokenType.LPAREN in token_types
        assert TokenType.PLUS in token_types
        assert TokenType.MULTIPLY in token_types
        assert TokenType.DOT in token_types
        assert TokenType.LBRACKET in token_types
        assert TokenType.MATRIX_MULT in token_types


if __name__ == "__main__":
    pytest.main([__file__]) 