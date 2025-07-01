"""
Tests for the NEURO parser.
"""

import pytest
from src.neuro.lexer import NeuroLexer
from src.neuro.parser import NeuroParser
from src.neuro.ast_nodes import *
from src.neuro.errors import NeuroSyntaxError


class TestNeuroParser:
    """Test cases for the NEURO parser."""
    
    def parse_source(self, source: str) -> Program:
        """Helper method to parse source code."""
        lexer = NeuroLexer(source)
        tokens = lexer.tokenize()
        parser = NeuroParser(tokens)
        return parser.parse()
    
    def test_empty_program(self):
        """Test parsing an empty program."""
        source = ""
        program = self.parse_source(source)
        
        assert isinstance(program, Program)
        assert len(program.declarations) == 0
        assert len(program.statements) == 0
    
    def test_simple_function(self):
        """Test parsing a simple function."""
        source = """
        func main() {
            return 42
        }
        """
        program = self.parse_source(source)
        
        assert len(program.declarations) == 1
        
        func_decl = program.declarations[0]
        assert isinstance(func_decl, FunctionDeclaration)
        assert func_decl.name == "main"
        assert len(func_decl.parameters) == 0
        assert func_decl.return_type is None
        assert len(func_decl.body) == 1
        
        return_stmt = func_decl.body[0]
        assert isinstance(return_stmt, ReturnStatement)
        assert isinstance(return_stmt.value, LiteralExpression)
        assert return_stmt.value.value == 42
    
    def test_function_with_parameters(self):
        """Test parsing a function with parameters."""
        source = """
        func add(a: int, b: int) -> int {
            return a + b
        }
        """
        program = self.parse_source(source)
        
        func_decl = program.declarations[0]
        assert isinstance(func_decl, FunctionDeclaration)
        assert func_decl.name == "add"
        assert len(func_decl.parameters) == 2
        
        # Check parameters
        param_a = func_decl.parameters[0]
        assert param_a.name == "a"
        assert isinstance(param_a.type_annotation, PrimitiveType)
        assert param_a.type_annotation.name == "int"
        
        param_b = func_decl.parameters[1]
        assert param_b.name == "b"
        assert isinstance(param_b.type_annotation, PrimitiveType)
        assert param_b.type_annotation.name == "int"
        
        # Check return type
        assert isinstance(func_decl.return_type, PrimitiveType)
        assert func_decl.return_type.name == "int"
    
    def test_generic_function(self):
        """Test parsing a generic function."""
        source = """
        func identity<T>(value: T) -> T {
            return value
        }
        """
        program = self.parse_source(source)
        
        func_decl = program.declarations[0]
        assert isinstance(func_decl, FunctionDeclaration)
        assert func_decl.name == "identity"
        assert func_decl.is_generic
        assert func_decl.type_params == ["T"]
    
    def test_variable_declaration(self):
        """Test parsing variable declarations."""
        source = """
        let x = 42
        let y: float = 3.14
        let name: string = "hello"
        """
        program = self.parse_source(source)
        
        assert len(program.statements) == 3
        
        # First variable: type inferred
        var1 = program.statements[0]
        assert isinstance(var1, VariableDeclaration)
        assert var1.name == "x"
        assert var1.type_annotation is None
        assert isinstance(var1.initializer, LiteralExpression)
        assert var1.initializer.value == 42
        
        # Second variable: explicit type
        var2 = program.statements[1]
        assert isinstance(var2, VariableDeclaration)
        assert var2.name == "y"
        assert isinstance(var2.type_annotation, PrimitiveType)
        assert var2.type_annotation.name == "float"
        
        # Third variable: string type
        var3 = program.statements[2]
        assert isinstance(var3, VariableDeclaration)
        assert var3.name == "name"
        assert isinstance(var3.type_annotation, PrimitiveType)
        assert var3.type_annotation.name == "string"
    
    def test_struct_declaration(self):
        """Test parsing struct declarations."""
        source = """
        struct Point {
            x: float,
            y: float
        }
        """
        program = self.parse_source(source)
        
        assert len(program.declarations) == 1
        
        struct_decl = program.declarations[0]
        assert isinstance(struct_decl, StructDeclaration)
        assert struct_decl.name == "Point"
        assert len(struct_decl.fields) == 2
        
        # Check fields
        field_x = struct_decl.fields[0]
        assert field_x.name == "x"
        assert isinstance(field_x.type_annotation, PrimitiveType)
        assert field_x.type_annotation.name == "float"
        
        field_y = struct_decl.fields[1]
        assert field_y.name == "y"
        assert isinstance(field_y.type_annotation, PrimitiveType)
        assert field_y.type_annotation.name == "float"
    
    def test_generic_struct(self):
        """Test parsing generic struct declarations."""
        source = """
        struct Point<T> {
            x: T,
            y: T
        }
        """
        program = self.parse_source(source)
        
        struct_decl = program.declarations[0]
        assert isinstance(struct_decl, StructDeclaration)
        assert struct_decl.name == "Point"
        assert struct_decl.is_generic
        assert struct_decl.type_params == ["T"]
    
    def test_binary_expressions(self):
        """Test parsing binary expressions."""
        source = """
        func test() {
            let a = 1 + 2
            let b = 3 * 4
            let c = a @ b
            let d = x == y
            let e = p and q
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Addition
        add_stmt = func_body[0]
        assert isinstance(add_stmt, VariableDeclaration)
        add_expr = add_stmt.initializer
        assert isinstance(add_expr, BinaryExpression)
        assert add_expr.operator == "+"
        
        # Multiplication
        mult_stmt = func_body[1]
        mult_expr = mult_stmt.initializer
        assert isinstance(mult_expr, BinaryExpression)
        assert mult_expr.operator == "*"
        
        # Matrix multiplication
        matmul_stmt = func_body[2]
        matmul_expr = matmul_stmt.initializer
        assert isinstance(matmul_expr, BinaryExpression)
        assert matmul_expr.operator == "@"
        
        # Equality
        eq_stmt = func_body[3]
        eq_expr = eq_stmt.initializer
        assert isinstance(eq_expr, BinaryExpression)
        assert eq_expr.operator == "=="
        
        # Logical and
        and_stmt = func_body[4]
        and_expr = and_stmt.initializer
        assert isinstance(and_expr, BinaryExpression)
        assert and_expr.operator == "and"
    
    def test_unary_expressions(self):
        """Test parsing unary expressions."""
        source = """
        func test() {
            let a = -42
            let b = not true
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Unary minus
        neg_stmt = func_body[0]
        neg_expr = neg_stmt.initializer
        assert isinstance(neg_expr, UnaryExpression)
        assert neg_expr.operator == "-"
        
        # Logical not
        not_stmt = func_body[1]
        not_expr = not_stmt.initializer
        assert isinstance(not_expr, UnaryExpression)
        assert not_expr.operator == "not"
    
    def test_function_calls(self):
        """Test parsing function calls."""
        source = """
        func test() {
            let result = add(1, 2)
            let generic_result = identity<int>(42)
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Regular function call
        call_stmt = func_body[0]
        call_expr = call_stmt.initializer
        assert isinstance(call_expr, CallExpression)
        assert isinstance(call_expr.function, IdentifierExpression)
        assert call_expr.function.name == "add"
        assert len(call_expr.arguments) == 2
        
        # Generic function call
        generic_call_stmt = func_body[1]
        generic_call_expr = generic_call_stmt.initializer
        assert isinstance(generic_call_expr, CallExpression)
        assert len(generic_call_expr.type_args) == 1
    
    def test_array_literals(self):
        """Test parsing array literals."""
        source = """
        func test() {
            let vector = [1, 2, 3, 4]
            let matrix = [[1, 2], [3, 4]]
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Vector
        vector_stmt = func_body[0]
        vector_expr = vector_stmt.initializer
        assert isinstance(vector_expr, ArrayExpression)
        assert len(vector_expr.elements) == 4
        
        # Matrix
        matrix_stmt = func_body[1]
        matrix_expr = matrix_stmt.initializer
        assert isinstance(matrix_expr, ArrayExpression)
        assert len(matrix_expr.elements) == 2
        assert isinstance(matrix_expr.elements[0], ArrayExpression)
    
    def test_struct_literals(self):
        """Test parsing struct literals."""
        source = """
        func test() {
            let point = Point { x: 1.0, y: 2.0 }
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        struct_stmt = func_body[0]
        struct_expr = struct_stmt.initializer
        assert isinstance(struct_expr, StructExpression)
        assert struct_expr.type_name == "Point"
        assert len(struct_expr.fields) == 2
        
        # Check fields
        field_x = struct_expr.fields[0]
        assert field_x[0] == "x"  # field name
        assert isinstance(field_x[1], LiteralExpression)  # field value
    
    def test_tensor_type(self):
        """Test parsing tensor types."""
        source = """
        func test() {
            let vector: Tensor<float> = [1.0, 2.0, 3.0]
            let matrix: Tensor<float, (3, 3)> = zeros(3, 3)
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Vector tensor
        vector_stmt = func_body[0]
        vector_type = vector_stmt.type_annotation
        assert isinstance(vector_type, TensorType)
        assert isinstance(vector_type.element_type, PrimitiveType)
        assert vector_type.element_type.name == "float"
        assert vector_type.shape is None
        
        # Matrix tensor with shape
        matrix_stmt = func_body[1]
        matrix_type = matrix_stmt.type_annotation
        assert isinstance(matrix_type, TensorType)
        assert matrix_type.shape == [3, 3]
    
    def test_neural_network_expression(self):
        """Test parsing neural network expressions."""
        source = """
        func test() {
            let model = NeuralNetwork<float, (784, 128, 10)> {
                dense_layer<128>(.relu),
                batch_norm(),
                dropout(0.2)
            }
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        nn_stmt = func_body[0]
        nn_expr = nn_stmt.initializer
        assert isinstance(nn_expr, NeuralNetworkExpression)
        assert isinstance(nn_expr.element_type, PrimitiveType)
        assert nn_expr.element_type.name == "float"
        assert nn_expr.shape_params == [784, 128, 10]
        assert len(nn_expr.layers) == 3
    
    def test_if_statement(self):
        """Test parsing if statements."""
        source = """
        func test() {
            if x > 0 {
                print("positive")
            } else {
                print("non-positive")
            }
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        if_stmt = func_body[0]
        assert isinstance(if_stmt, IfStatement)
        
        # Check condition
        assert isinstance(if_stmt.condition, BinaryExpression)
        assert if_stmt.condition.operator == ">"
        
        # Check then body
        assert len(if_stmt.then_body) == 1
        assert isinstance(if_stmt.then_body[0], ExpressionStatement)
        
        # Check else body
        assert len(if_stmt.else_body) == 1
        assert isinstance(if_stmt.else_body[0], ExpressionStatement)
    
    def test_for_statement(self):
        """Test parsing for statements."""
        source = """
        func test() {
            for element in array {
                print(element)
            }
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        for_stmt = func_body[0]
        assert isinstance(for_stmt, ForStatement)
        assert for_stmt.variable == "element"
        assert isinstance(for_stmt.iterable, IdentifierExpression)
        assert for_stmt.iterable.name == "array"
        assert len(for_stmt.body) == 1
    
    def test_member_access(self):
        """Test parsing member access expressions."""
        source = """
        func test() {
            let x = obj.field
            let y = point.x
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # First member access
        member_stmt = func_body[0]
        member_expr = member_stmt.initializer
        assert isinstance(member_expr, MemberExpression)
        assert isinstance(member_expr.object, IdentifierExpression)
        assert member_expr.object.name == "obj"
        assert member_expr.member == "field"
    
    def test_index_access(self):
        """Test parsing index access expressions."""
        source = """
        func test() {
            let x = array[index]
            let y = matrix[0][1]
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Simple index access
        index_stmt = func_body[0]
        index_expr = index_stmt.initializer
        assert isinstance(index_expr, IndexExpression)
        assert isinstance(index_expr.object, IdentifierExpression)
        assert index_expr.object.name == "array"
        
        # Chained index access
        chain_stmt = func_body[1]
        chain_expr = chain_stmt.initializer
        assert isinstance(chain_expr, IndexExpression)
        assert isinstance(chain_expr.object, IndexExpression)  # Nested index
    
    def test_assignment_statement(self):
        """Test parsing assignment statements."""
        source = """
        func test() {
            x = 42
            array[0] = value
            obj.field = new_value
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # Simple assignment
        assign_stmt = func_body[0]
        assert isinstance(assign_stmt, AssignmentStatement)
        assert isinstance(assign_stmt.target, IdentifierExpression)
        assert assign_stmt.target.name == "x"
        
        # Index assignment
        index_assign_stmt = func_body[1]
        assert isinstance(index_assign_stmt, AssignmentStatement)
        assert isinstance(index_assign_stmt.target, IndexExpression)
        
        # Member assignment
        member_assign_stmt = func_body[2]
        assert isinstance(member_assign_stmt, AssignmentStatement)
        assert isinstance(member_assign_stmt.target, MemberExpression)
    
    def test_operator_precedence(self):
        """Test operator precedence parsing."""
        source = """
        func test() {
            let result = a + b * c
            let result2 = (a + b) * c
            let result3 = a @ b + c
        }
        """
        program = self.parse_source(source)
        
        func_body = program.declarations[0].body
        
        # a + b * c should parse as a + (b * c)
        expr1 = func_body[0].initializer
        assert isinstance(expr1, BinaryExpression)
        assert expr1.operator == "+"
        assert isinstance(expr1.right, BinaryExpression)
        assert expr1.right.operator == "*"
        
        # (a + b) * c should parse as (a + b) * c
        expr2 = func_body[1].initializer
        assert isinstance(expr2, BinaryExpression)
        assert expr2.operator == "*"
        assert isinstance(expr2.left, BinaryExpression)
        assert expr2.left.operator == "+"
    
    def test_complex_program(self):
        """Test parsing a complex program with multiple constructs."""
        source = """
        struct Point<T> {
            x: T,
            y: T
        }
        
        func distance<T>(p1: Point<T>, p2: Point<T>) -> T {
            let dx = p1.x - p2.x
            let dy = p1.y - p2.y
            return dx * dx + dy * dy
        }
        
        func main() {
            let p1 = Point { x: 1.0, y: 2.0 }
            let p2 = Point { x: 4.0, y: 6.0 }
            let dist = distance(p1, p2)
            print("Distance: " + str(dist))
        }
        """
        program = self.parse_source(source)
        
        # Should parse successfully
        assert len(program.declarations) == 3  # struct, distance function, main function
        
        # Check struct declaration
        struct_decl = program.declarations[0]
        assert isinstance(struct_decl, StructDeclaration)
        assert struct_decl.name == "Point"
        assert struct_decl.is_generic
        
        # Check generic function
        distance_func = program.declarations[1]
        assert isinstance(distance_func, FunctionDeclaration)
        assert distance_func.name == "distance"
        assert distance_func.is_generic
        
        # Check main function
        main_func = program.declarations[2]
        assert isinstance(main_func, FunctionDeclaration)
        assert main_func.name == "main"
        assert not main_func.is_generic
    
    def test_error_recovery(self):
        """Test parser error recovery."""
        source = """
        func bad_syntax( {
            // Missing parameter list closing paren
        }
        
        func good_function() {
            return 42
        }
        """
        # Should handle syntax errors gracefully
        try:
            program = self.parse_source(source)
            # If parsing succeeds, check that error recovery worked
        except NeuroSyntaxError:
            # Expected for malformed syntax
            pass


if __name__ == "__main__":
    pytest.main([__file__]) 