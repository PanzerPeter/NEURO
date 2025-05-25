"""
Tests for the NEURO Lexer

Comprehensive test suite covering all token types and edge cases.
"""

import pytest
from neuro.lexer import NeuroLexer, TokenType, Token
from neuro.errors import LexerError


class TestNeuroLexer:
    """Test suite for the NEURO lexer."""
    
    def setup_method(self):
        """Set up a fresh lexer for each test."""
        self.lexer = NeuroLexer()
    
    def test_empty_input(self):
        """Test lexing empty input."""
        tokens = self.lexer.tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        tokens = self.lexer.tokenize("   \t\r  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF
    
    def test_newlines(self):
        """Test newline handling."""
        tokens = self.lexer.tokenize("hello\nworld\n")
        expected = [
            TokenType.IDENTIFIER,  # hello
            TokenType.NEWLINE,
            TokenType.IDENTIFIER,  # world
            TokenType.NEWLINE,
            TokenType.EOF
        ]
        assert [t.type for t in tokens] == expected
    
    def test_identifiers_and_keywords(self):
        """Test identifier and keyword recognition."""
        source = "func let if else while for true false"
        tokens = self.lexer.tokenize(source)
        
        expected = [
            TokenType.FUNC,
            TokenType.LET,
            TokenType.IF,
            TokenType.ELSE,
            TokenType.WHILE,
            TokenType.FOR,
            TokenType.BOOLEAN,  # true
            TokenType.BOOLEAN,  # false
            TokenType.EOF
        ]
        assert [t.type for t in tokens] == expected
    
    def test_ai_keywords(self):
        """Test AI-specific keywords."""
        source = "NeuralNetwork tensor gpu model layer optimizer"
        tokens = self.lexer.tokenize(source)
        
        expected = [
            TokenType.NEURALNETWORK,
            TokenType.TENSOR,
            TokenType.GPU,
            TokenType.MODEL,
            TokenType.LAYER,
            TokenType.OPTIMIZER,
            TokenType.EOF
        ]
        assert [t.type for t in tokens] == expected
    
    def test_integers(self):
        """Test integer literal parsing."""
        test_cases = [
            ("42", 42),
            ("0", 0),
            ("123456", 123456),
            ("007", 7)  # Leading zeros should be handled
        ]
        
        for source, expected_value in test_cases:
            tokens = self.lexer.tokenize(source)
            assert len(tokens) == 2  # INTEGER + EOF
            assert tokens[0].type == TokenType.INTEGER
            assert tokens[0].value == source
            assert int(tokens[0].value) == expected_value
    
    def test_floats(self):
        """Test float literal parsing."""
        test_cases = [
            "3.14",
            "0.5",
            "123.456",
            "1.0",
            "0.0"
        ]
        
        for source in test_cases:
            tokens = self.lexer.tokenize(source)
            assert len(tokens) == 2  # FLOAT + EOF
            assert tokens[0].type == TokenType.FLOAT
            assert tokens[0].value == source
    
    def test_scientific_notation(self):
        """Test scientific notation parsing."""
        test_cases = [
            "1e10",
            "2.5e-3",
            "1.23E+4",
            "5e0"
        ]
        
        for source in test_cases:
            tokens = self.lexer.tokenize(source)
            assert len(tokens) == 2  # FLOAT + EOF
            assert tokens[0].type == TokenType.FLOAT
            assert tokens[0].value == source
    
    def test_strings(self):
        """Test string literal parsing."""
        test_cases = [
            ('\"hello\"', "hello"),
            ("'world'", "world"),
            ('\"Hello, World!\"', "Hello, World!"),
            ('\"\"', ""),
            ("''", "")
        ]
        
        for source, expected in test_cases:
            tokens = self.lexer.tokenize(source)
            assert len(tokens) == 2  # STRING + EOF
            assert tokens[0].type == TokenType.STRING
            assert tokens[0].value == expected
    
    def test_string_escapes(self):
        """Test string escape sequences."""
        test_cases = [
            ('\"hello\\nworld\"', "hello\nworld"),
            ('\"tab\\there\"', "tab\there"),
            ('\"quote\\\"here\"', "quote\"here"),
            ('\"backslash\\\\here\"', "backslash\\here"),
            ('\"carriage\\rreturn\"', "carriage\rreturn")
        ]
        
        for source, expected in test_cases:
            tokens = self.lexer.tokenize(source)
            assert len(tokens) == 2  # STRING + EOF
            assert tokens[0].type == TokenType.STRING
            assert tokens[0].value == expected
    
    def test_multiline_strings(self):
        """Test multi-line string literals."""
        source = '\"hello\\nworld\\nfrom\\nneuro\"'
        tokens = self.lexer.tokenize(source)
        assert len(tokens) == 2
        assert tokens[0].type == TokenType.STRING
        assert "\\n" in tokens[0].value or "\n" in tokens[0].value
    
    def test_operators(self):
        """Test operator tokenization."""
        operators = [
            ("+", TokenType.PLUS),
            ("-", TokenType.MINUS),
            ("*", TokenType.MULTIPLY),
            ("/", TokenType.DIVIDE),
            ("%", TokenType.MODULO),
            ("**", TokenType.POWER),
            ("@", TokenType.MATRIX_MUL),
            ("==", TokenType.EQUAL),
            ("!=", TokenType.NOT_EQUAL),
            ("<", TokenType.LESS),
            ("<=", TokenType.LESS_EQUAL),
            (">", TokenType.GREATER),
            (">=", TokenType.GREATER_EQUAL),
            ("&&", TokenType.AND),
            ("||", TokenType.OR),
            ("!", TokenType.NOT),
            ("=", TokenType.ASSIGN),
            ("+=", TokenType.PLUS_ASSIGN),
            ("-=", TokenType.MINUS_ASSIGN),
            ("->", TokenType.ARROW),
            ("=>", TokenType.FAT_ARROW),
            ("::", TokenType.DOUBLE_COLON),
            ("..", TokenType.RANGE),
            ("..=", TokenType.RANGE_INCLUSIVE)
        ]
        
        for op_text, expected_type in operators:
            tokens = self.lexer.tokenize(op_text)
            assert len(tokens) == 2  # OPERATOR + EOF
            assert tokens[0].type == expected_type
            assert tokens[0].value == op_text
    
    def test_punctuation(self):
        """Test punctuation tokenization."""
        punctuation = [
            ("(", TokenType.LPAREN),
            (")", TokenType.RPAREN),
            ("[", TokenType.LBRACKET),
            ("]", TokenType.RBRACKET),
            ("{", TokenType.LBRACE),
            ("}", TokenType.RBRACE),
            (";", TokenType.SEMICOLON),
            (",", TokenType.COMMA),
            (":", TokenType.COLON),
            (".", TokenType.DOT),
            ("?", TokenType.QUESTION)
        ]
        
        for punct_text, expected_type in punctuation:
            tokens = self.lexer.tokenize(punct_text)
            assert len(tokens) == 2  # PUNCTUATION + EOF
            assert tokens[0].type == expected_type
            assert tokens[0].value == punct_text
    
    def test_line_comments(self):
        """Test line comment handling."""
        source = """
        let x = 42  // This is a comment
        let y = 3.14
        // Another comment
        """
        tokens = self.lexer.tokenize(source)
        
        # Comments should be skipped
        token_types = [t.type for t in tokens if t.type != TokenType.NEWLINE and t.type != TokenType.EOF]
        expected = [TokenType.LET, TokenType.IDENTIFIER, TokenType.ASSIGN, TokenType.INTEGER,
                   TokenType.LET, TokenType.IDENTIFIER, TokenType.ASSIGN, TokenType.FLOAT]
        assert token_types == expected
    
    def test_block_comments(self):
        """Test block comment handling."""
        source = """
        let x = 42  /* This is a
                      multi-line comment */
        let y = 3.14
        """
        tokens = self.lexer.tokenize(source)
        
        # Comments should be skipped
        token_types = [t.type for t in tokens if t.type != TokenType.NEWLINE and t.type != TokenType.EOF]
        expected = [TokenType.LET, TokenType.IDENTIFIER, TokenType.ASSIGN, TokenType.INTEGER,
                   TokenType.LET, TokenType.IDENTIFIER, TokenType.ASSIGN, TokenType.FLOAT]
        assert token_types == expected
    
    def test_complex_expression(self):
        """Test tokenization of a complex expression."""
        source = "let result = (a + b) * c.field[index]"
        tokens = self.lexer.tokenize(source)
        
        expected_types = [
            TokenType.LET, TokenType.IDENTIFIER,  # let result
            TokenType.ASSIGN,  # =
            TokenType.LPAREN, TokenType.IDENTIFIER, TokenType.PLUS, TokenType.IDENTIFIER, TokenType.RPAREN,  # (a + b)
            TokenType.MULTIPLY,  # *
            TokenType.IDENTIFIER, TokenType.DOT, TokenType.IDENTIFIER,  # c.field
            TokenType.LBRACKET, TokenType.IDENTIFIER, TokenType.RBRACKET,  # [index]
            TokenType.EOF
        ]
        
        assert [t.type for t in tokens] == expected_types
    
    def test_function_definition(self):
        """Test tokenization of a function definition."""
        source = """
        func multiply(a: int, b: int) -> int {
            return a * b
        }
        """
        tokens = self.lexer.tokenize(source)
        
        # Should contain all expected tokens for function definition
        token_types = [t.type for t in tokens if t.type != TokenType.NEWLINE]
        assert TokenType.FUNC in token_types
        assert TokenType.IDENTIFIER in token_types
        assert TokenType.LPAREN in token_types
        assert TokenType.COLON in token_types
        assert TokenType.INT in token_types
        assert TokenType.ARROW in token_types
        assert TokenType.RETURN in token_types
    
    def test_tensor_syntax(self):
        """Test tokenization of tensor syntax."""
        source = "let tensor: Tensor<float, (28, 28)> = [[1.0, 2.0], [3.0, 4.0]]"
        tokens = self.lexer.tokenize(source)
        
        token_types = [t.type for t in tokens]
        assert TokenType.TENSOR in token_types
        assert TokenType.LESS in token_types
        assert TokenType.FLOAT_TYPE in token_types
        assert TokenType.COMMA in token_types
        assert TokenType.GREATER in token_types
        assert TokenType.LBRACKET in token_types
        assert TokenType.RBRACKET in token_types
    
    def test_error_cases(self):
        """Test various error conditions."""
        error_cases = [
            ('\"unterminated string', "Unterminated string literal"),
            ('1.2.3', "Invalid number format"),
            ('1e', "Invalid scientific notation"),
            ('/*unterminated comment', "Unterminated block comment"),
            ('@#$', "Unexpected character")
        ]
        
        for source, expected_error in error_cases:
            with pytest.raises(LexerError) as exc_info:
                self.lexer.tokenize(source)
            assert expected_error.lower() in str(exc_info.value).lower()
    
    def test_location_tracking(self):
        """Test that source locations are properly tracked."""
        source = """line 1
        line 2 with content
        line 3"""
        
        tokens = self.lexer.tokenize(source, "test.nr")
        
        # Check that locations are properly set
        for token in tokens:
            assert token.line >= 1
            assert token.column >= 1
            assert token.filename == "test.nr"
        
        # Check specific locations
        identifier_tokens = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        assert len(identifier_tokens) >= 3
        
        # First identifier should be on line 1
        assert identifier_tokens[0].line == 1
        
        # Second identifier should be on line 2
        assert identifier_tokens[1].line == 2
    
    def test_neural_network_example(self):
        """Test tokenization of a neural network example."""
        source = """
        let model = NeuralNetwork<float, (784, 128, 10)> {
            dense_layer(units=128, activation=relu),
            dropout(rate=0.2),
            dense_layer(units=10, activation=softmax)
        }
        """
        
        tokens = self.lexer.tokenize(source)
        
        # Should successfully tokenize without errors
        assert tokens[-1].type == TokenType.EOF
        
        # Should contain AI-specific tokens
        token_types = [t.type for t in tokens]
        assert TokenType.NEURALNETWORK in token_types
        assert TokenType.IDENTIFIER in token_types  # dense_layer, relu, etc.
        assert TokenType.LBRACE in token_types
        assert TokenType.RBRACE in token_types


if __name__ == "__main__":
    pytest.main([__file__])