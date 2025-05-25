"""
Tests for the NEURO Parser

Comprehensive test suite covering AST generation and parsing edge cases.
"""

import pytest
from neuro.lexer import NeuroLexer
from neuro.parser import NeuroParser
from neuro.ast_nodes import *
from neuro.errors import ParseError


class TestNeuroParser:
    """Test suite for the NEURO parser."""
    
    def setup_method(self):
        """Set up fresh lexer and parser for each test."""
        self.lexer = NeuroLexer()
        self.parser = NeuroParser()
    
    def parse_source(self, source: str) -> Program:
        """Helper to tokenize and parse source code."""
        tokens = self.lexer.tokenize(source, "test.nr")
        return self.parser.parse(tokens)
    
    def test_empty_program(self):
        """Test parsing an empty program."""
        ast = self.parse_source("")
        assert isinstance(ast, Program)
        assert len(ast.statements) == 0
    
    def test_simple_variable_declaration(self):
        """Test parsing variable declarations."""
        source = "let x = 42"
        ast = self.parse_source(source)
        
        assert len(ast.statements) == 1
        stmt = ast.statements[0]
        assert isinstance(stmt, VariableDeclaration)
        assert stmt.name == "x"
        assert isinstance(stmt.initializer, Literal)
        assert stmt.initializer.value == 42
    
    def test_typed_variable_declaration(self):
        """Test parsing typed variable declarations."""
        source = "let x: int = 42"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, VariableDeclaration)
        assert stmt.name == "x"
        assert isinstance(stmt.type_annotation, PrimitiveType)
        assert stmt.type_annotation.name == "int"
        assert isinstance(stmt.initializer, Literal)
        assert stmt.initializer.value == 42
    
    def test_mutable_variable_declaration(self):
        """Test parsing mutable variable declarations."""
        source = "let mut x = 42"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, VariableDeclaration)
        assert stmt.name == "x"
        assert stmt.is_mutable == True
    
    def test_function_declaration(self):
        """Test parsing function declarations."""
        source = """
        func add(a: int, b: int) -> int {
            return a + b
        }
        """
        ast = self.parse_source(source)
        
        assert len(ast.statements) == 1
        func = ast.statements[0]
        assert isinstance(func, FunctionDeclaration)
        assert func.name == "add"
        assert len(func.parameters) == 2
        
        # Check parameters
        assert func.parameters[0].name == "a"
        assert isinstance(func.parameters[0].type_annotation, PrimitiveType)
        assert func.parameters[0].type_annotation.name == "int"
        
        assert func.parameters[1].name == "b"
        assert isinstance(func.parameters[1].type_annotation, PrimitiveType)
        assert func.parameters[1].type_annotation.name == "int"
        
        # Check return type
        assert isinstance(func.return_type, PrimitiveType)
        assert func.return_type.name == "int"
        
        # Check body
        assert isinstance(func.body, Block)
        assert len(func.body.statements) == 1
        assert isinstance(func.body.statements[0], ReturnStatement)
    
    def test_function_with_default_parameters(self):
        """Test parsing functions with default parameters."""
        source = """
        func greet(name: string, greeting: string = "Hello") {
            print(greeting + " " + name)
        }
        """
        ast = self.parse_source(source)
        
        func = ast.statements[0]
        assert isinstance(func, FunctionDeclaration)
        assert len(func.parameters) == 2
        
        # Second parameter should have default value
        assert func.parameters[1].name == "greeting"
        assert func.parameters[1].default_value is not None
        assert isinstance(func.parameters[1].default_value, Literal)
        assert func.parameters[1].default_value.value == "Hello"
    
    def test_struct_declaration(self):
        """Test parsing struct declarations."""
        source = """
        struct Point {
            x: float
            y: float
        }
        """
        ast = self.parse_source(source)
        
        assert len(ast.statements) == 1
        struct = ast.statements[0]
        assert isinstance(struct, StructDeclaration)
        assert struct.name == "Point"
        assert len(struct.fields) == 2
        
        assert struct.fields[0].name == "x"
        assert isinstance(struct.fields[0].type_annotation, PrimitiveType)
        assert struct.fields[0].type_annotation.name == "float"
        
        assert struct.fields[1].name == "y"
        assert isinstance(struct.fields[1].type_annotation, PrimitiveType)
        assert struct.fields[1].type_annotation.name == "float"
    
    def test_tensor_type_parsing(self):
        """Test parsing tensor types."""
        source = "let weights: Tensor<float, (128, 784)> = create_weights()"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, VariableDeclaration)
        assert isinstance(stmt.type_annotation, TensorType)
        
        tensor_type = stmt.type_annotation
        assert isinstance(tensor_type.element_type, PrimitiveType)
        assert tensor_type.element_type.name == "float"
        assert tensor_type.shape == [128, 784]
    
    def test_binary_expressions(self):
        """Test parsing binary expressions."""
        test_cases = [
            ("1 + 2", "+"),
            ("x - y", "-"),
            ("a * b", "*"),
            ("p / q", "/"),
            ("m % n", "%"),
            ("x @ y", "@"),  # Matrix multiplication
            ("a == b", "=="),
            ("x != y", "!="),
            ("p < q", "<"),
            ("m <= n", "<="),
            ("a > b", ">"),
            ("x >= y", ">="),
            ("p && q", "&&"),
            ("m || n", "||")
        ]
        
        for source, expected_op in test_cases:
            ast = self.parse_source(source)
            stmt = ast.statements[0]
            assert isinstance(stmt, ExpressionStatement)
            expr = stmt.expression
            assert isinstance(expr, BinaryOp)
            assert expr.operator == expected_op
    
    def test_unary_expressions(self):
        """Test parsing unary expressions."""
        test_cases = [
            ("-x", "-"),
            ("!flag", "!"),
            ("~bits", "~")
        ]
        
        for source, expected_op in test_cases:
            ast = self.parse_source(source)
            stmt = ast.statements[0]
            assert isinstance(stmt, ExpressionStatement)
            expr = stmt.expression
            assert isinstance(expr, UnaryOp)
            assert expr.operator == expected_op
    
    def test_function_calls(self):
        """Test parsing function calls."""
        source = "print(\"Hello, World!\")"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, ExpressionStatement)
        expr = stmt.expression
        assert isinstance(expr, FunctionCall)
        assert isinstance(expr.function, Identifier)
        assert expr.function.name == "print"
        assert len(expr.arguments) == 1
        assert isinstance(expr.arguments[0], Literal)
        assert expr.arguments[0].value == "Hello, World!"
    
    def test_member_access(self):
        """Test parsing member access expressions."""
        source = "point.x"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        expr = stmt.expression
        assert isinstance(expr, MemberAccess)
        assert isinstance(expr.object, Identifier)
        assert expr.object.name == "point"
        assert expr.member == "x"
    
    def test_index_access(self):
        """Test parsing index access expressions."""
        source = "array[index]"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        expr = stmt.expression
        assert isinstance(expr, IndexAccess)
        assert isinstance(expr.object, Identifier)
        assert expr.object.name == "array"
        assert isinstance(expr.index, Identifier)
        assert expr.index.name == "index"
    
    def test_tensor_literals(self):
        """Test parsing tensor literals."""
        source = "let matrix = [[1, 2], [3, 4]]"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt.initializer, TensorLiteral)
        tensor = stmt.initializer
        assert len(tensor.elements) == 2
        
        # Check nested structure
        assert isinstance(tensor.elements[0], TensorLiteral)
        assert isinstance(tensor.elements[1], TensorLiteral)
        
        row1 = tensor.elements[0]
        assert len(row1.elements) == 2
        assert isinstance(row1.elements[0], Literal)
        assert row1.elements[0].value == 1
    
    def test_if_statement(self):
        """Test parsing if statements."""
        source = """
        if x > 0 {
            print("positive")
        } else {
            print("not positive")
        }
        """
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, IfStatement)
        assert isinstance(stmt.condition, BinaryOp)
        assert stmt.condition.operator == ">"
        assert isinstance(stmt.then_branch, Block)
        assert isinstance(stmt.else_branch, Block)
    
    def test_while_statement(self):
        """Test parsing while statements."""
        source = """
        while i < 10 {
            i = i + 1
        }
        """
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, WhileStatement)
        assert isinstance(stmt.condition, BinaryOp)
        assert isinstance(stmt.body, Block)
    
    def test_for_statement(self):
        """Test parsing for statements."""
        source = """
        for item in collection {
            print(item)
        }
        """
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt, ForStatement)
        assert stmt.variable == "item"
        assert isinstance(stmt.iterable, Identifier)
        assert stmt.iterable.name == "collection"
        assert isinstance(stmt.body, Block)
    
    def test_assignment_expressions(self):
        """Test parsing assignment expressions."""
        source = "x = 42"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        expr = stmt.expression
        assert isinstance(expr, Assignment)
        assert isinstance(expr.target, Identifier)
        assert expr.target.name == "x"
        assert isinstance(expr.value, Literal)
        assert expr.value.value == 42
    
    def test_compound_assignment(self):
        """Test parsing compound assignment operators."""
        test_cases = ["+=", "-=", "*=", "/="]
        
        for op in test_cases:
            source = f"x {op} 5"
            ast = self.parse_source(source)
            
            stmt = ast.statements[0]
            expr = stmt.expression
            assert isinstance(expr, Assignment)
            # Compound assignment should be transformed to binary op
            assert isinstance(expr.value, BinaryOp)
            assert expr.value.operator == op[:-1]  # Remove '='
    
    def test_precedence_and_associativity(self):
        """Test operator precedence and associativity."""
        source = "a + b * c"
        ast = self.parse_source(source)
        
        # Should parse as a + (b * c), not (a + b) * c
        stmt = ast.statements[0]
        expr = stmt.expression
        assert isinstance(expr, BinaryOp)
        assert expr.operator == "+"
        assert isinstance(expr.left, Identifier)
        assert expr.left.name == "a"
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.operator == "*"
    
    def test_parenthesized_expressions(self):
        """Test parsing parenthesized expressions."""
        source = "(a + b) * c"
        ast = self.parse_source(source)
        
        # Should parse as (a + b) * c
        stmt = ast.statements[0]
        expr = stmt.expression
        assert isinstance(expr, BinaryOp)
        assert expr.operator == "*"
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.operator == "+"
        assert isinstance(expr.right, Identifier)
        assert expr.right.name == "c"
    
    def test_complex_expression(self):
        """Test parsing complex nested expressions."""
        source = "result = obj.method(x + y, z[i].field)"
        ast = self.parse_source(source)
        
        stmt = ast.statements[0]
        assert isinstance(stmt.expression, Assignment)
        
        # Right side should be a function call
        call = stmt.expression.value
        assert isinstance(call, FunctionCall)
        
        # Function should be member access
        assert isinstance(call.function, MemberAccess)
        assert call.function.member == "method"
        
        # Should have two arguments
        assert len(call.arguments) == 2
        
        # First argument is binary operation
        assert isinstance(call.arguments[0], BinaryOp)
        
        # Second argument is member access on index access
        assert isinstance(call.arguments[1], MemberAccess)
        assert isinstance(call.arguments[1].object, IndexAccess)
    
    def test_generic_types(self):
        """Test parsing generic type parameters."""
        source = """
        func identity<T>(value: T) -> T {
            return value
        }
        """
        ast = self.parse_source(source)
        
        func = ast.statements[0]
        assert isinstance(func, FunctionDeclaration)
        assert func.generic_params is not None
        assert len(func.generic_params) == 1
        assert func.generic_params[0].name == "T"
    
    def test_gpu_function_attribute(self):
        """Test parsing @gpu function attribute."""
        source = """
        @gpu func matrix_multiply(a: Tensor<float>, b: Tensor<float>) -> Tensor<float> {
            return a @ b
        }
        """
        # This test might need adjustment based on how @gpu is tokenized
        # For now, skip implementation details
        
    def test_error_recovery(self):
        """Test parser error recovery."""
        # Test various syntax errors
        error_cases = [
            "let x =",  # Missing value
            "func () {",  # Missing function name
            "if {",  # Missing condition
            "let x: = 42",  # Missing type
            "func f(a:) {}",  # Missing parameter type
        ]
        
        for source in error_cases:
            with pytest.raises(ParseError):
                self.parse_source(source)
    
    def test_complete_program(self):
        """Test parsing a complete program with multiple constructs."""
        source = """
        // A complete NEURO program
        
        struct Point {
            x: float
            y: float
        }
        
        func distance(p1: Point, p2: Point) -> float {
            let dx = p1.x - p2.x
            let dy = p1.y - p2.y
            return sqrt(dx * dx + dy * dy)
        }
        
        func main() {
            let x = 3.0
            let y = 4.0
            
            let dist = sqrt(x * x + y * y)
            print("Distance calculated")
            
            if dist > 5.0 {
                print("Far away")
            } else {
                print("Close")
            }
        }
        """
        
        ast = self.parse_source(source)
        
        # Should parse without errors
        assert isinstance(ast, Program)
        assert len(ast.statements) >= 3  # struct, distance function, main function
        
        # Check that we have the expected constructs
        types = [type(stmt) for stmt in ast.statements]
        assert StructDeclaration in types
        assert FunctionDeclaration in types
    
    def test_neural_network_syntax(self):
        """Test parsing neural network specific syntax."""
        source = """
        let model = NeuralNetwork<float, (784, 128, 10)> {
            dense_layer(units=128, activation=relu),
            batch_norm(),
            dropout(rate=0.2),
            dense_layer(units=10, activation=softmax)
        }
        """
        
        # This test depends on how neural network syntax is implemented
        # For now, this serves as a placeholder for future implementation
        
    def test_ast_pretty_printing(self):
        """Test that AST nodes can be pretty-printed."""
        source = "let x = 42 + 3.14"
        ast = self.parse_source(source)
        
        # Should be able to pretty print without errors
        pretty = ast.pretty_print()
        assert isinstance(pretty, str)
        assert len(pretty) > 0
        assert "let" in pretty
        assert "x" in pretty


if __name__ == "__main__":
    pytest.main([__file__]) 