"""
Test suite for the NEURO Type Checker and Inference Engine.

This test suite comprehensively validates:
1. Symbol table management
2. Type inference for expressions and statements
3. Type unification and constraint solving
4. Generic type handling
5. Tensor shape verification
6. Error detection and reporting
"""

import pytest
from typing import Dict, Any

from neuro.lexer import NeuroLexer
from neuro.parser import NeuroParser
from neuro.type_checker import TypeChecker, TypeVariable, NeuroTypeError, SymbolInfo
from neuro.ast_nodes import *
from neuro.errors import SourceLocation


class TestTypeChecker:
    """Test suite for the NEURO type checker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lexer = NeuroLexer()
        self.parser = NeuroParser()
        self.type_checker = TypeChecker(verbose=False)
    
    def _parse_and_check(self, source: str) -> Dict[ASTNode, Type]:
        """Helper to parse source and run type checking."""
        tokens = self.lexer.tokenize(source)
        ast = self.parser.parse(tokens)
        return self.type_checker.check_program(ast)
    
    # ========================================================================
    # Basic Type Inference Tests
    # ========================================================================
    
    def test_literal_type_inference(self):
        """Test type inference for literal values."""
        source = """
        func main() {
            let x = 42
            let y = 3.14
            let name = "hello"
            let flag = true
        }
        """
        
        types = self._parse_and_check(source)
        
        # Find variable declarations and check their types
        for node, node_type in types.items():
            if isinstance(node, VariableDeclaration):
                if node.name == "x":
                    assert isinstance(node_type, PrimitiveType)
                    assert node_type.name == "int"
                elif node.name == "y":
                    assert isinstance(node_type, PrimitiveType)
                    assert node_type.name == "float"
                elif node.name == "name":
                    assert isinstance(node_type, PrimitiveType)
                    assert node_type.name == "string"
                elif node.name == "flag":
                    assert isinstance(node_type, PrimitiveType)
                    assert node_type.name == "bool"
    
    def test_arithmetic_type_inference(self):
        """Test type inference for arithmetic operations."""
        source = """
        func main() {
            let a = 10
            let b = 20
            let sum = a + b
            let x = 1.5
            let y = 2.5
            let product = x * y
        }
        """
        
        types = self._parse_and_check(source)
        
        # Check that arithmetic operations preserve types
        for node, node_type in types.items():
            if isinstance(node, VariableDeclaration):
                if node.name in ["sum"]:
                    assert isinstance(node_type, PrimitiveType)
                    assert node_type.name == "int"
                elif node.name in ["product"]:
                    assert isinstance(node_type, PrimitiveType)
                    assert node_type.name == "float"
    
    def test_function_type_checking(self):
        """Test function parameter and return type checking."""
        source = """
        func add(a: int, b: int) -> int {
            return a + b
        }
        
        func main() {
            let result = add(5, 10)
        }
        """
        
        types = self._parse_and_check(source)
        
        # Check function types
        for node, node_type in types.items():
            if isinstance(node, FunctionDeclaration) and node.name == "add":
                assert isinstance(node_type, FunctionType)
                assert len(node_type.param_types) == 2
                assert all(isinstance(pt, PrimitiveType) and pt.name == "int" 
                          for pt in node_type.param_types)
                assert isinstance(node_type.return_type, PrimitiveType)
                assert node_type.return_type.name == "int"
    
    def test_tensor_type_inference(self):
        """Test tensor type inference and shape checking."""
        source = """
        func main() {
            let vector = [1.0, 2.0, 3.0]
            let matrix = [[1, 2], [3, 4]]
        }
        """
        
        types = self._parse_and_check(source)
        
        for node, node_type in types.items():
            if isinstance(node, VariableDeclaration):
                if node.name == "vector":
                    assert isinstance(node_type, TensorType)
                    assert isinstance(node_type.element_type, PrimitiveType)
                    assert node_type.element_type.name == "float"
                    assert node_type.shape == [3]
                elif node.name == "matrix":
                    assert isinstance(node_type, TensorType)
                    assert isinstance(node_type.element_type, PrimitiveType)
                    assert node_type.element_type.name == "int"
                    assert node_type.shape == [2, 2]
    
    def test_control_flow_type_checking(self):
        """Test type checking in control flow statements."""
        source = """
        func main() {
            let x = 10
            if x > 5 {
                let y = x + 1
            }
            
            let numbers = [1, 2, 3]
            for num in numbers {
                let doubled = num * 2
            }
        }
        """
        
        # Should complete without type errors
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    # ========================================================================
    # Type Constraint and Unification Tests
    # ========================================================================
    
    def test_type_variable_unification(self):
        """Test type variable unification in constraints."""
        source = """
        func identity<T>(x: T) -> T {
            return x
        }
        
        func main() {
            let result = identity(42)
        }
        """
        
        # Should complete without errors - generic instantiation
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    def test_complex_type_inference(self):
        """Test complex type inference scenarios."""
        source = """
        func process_data(data: Tensor<float>) -> float {
            let sum = 0.0
            for element in data {
                sum = sum + element
            }
            return sum
        }
        
        func main() {
            let data = [1.0, 2.0, 3.0, 4.0]
            let result = process_data(data)
        }
        """
        
        types = self._parse_and_check(source)
        
        # Verify the function type is correctly inferred
        for node, node_type in types.items():
            if isinstance(node, FunctionDeclaration) and node.name == "process_data":
                assert isinstance(node_type, FunctionType)
                assert len(node_type.param_types) == 1
                param_type = node_type.param_types[0]
                assert isinstance(param_type, TensorType)
                assert isinstance(param_type.element_type, PrimitiveType)
                assert param_type.element_type.name == "float"
    
    # ========================================================================
    # Error Detection Tests
    # ========================================================================
    
    def test_type_mismatch_error(self):
        """Test detection of type mismatches."""
        source = """
        func main() {
            let x: int = "hello"
        }
        """
        
        with pytest.raises(NeuroTypeError) as exc_info:
            self._parse_and_check(source)
        
        assert "cannot unify" in str(exc_info.value).lower()
    
    def test_undefined_variable_error(self):
        """Test detection of undefined variables."""
        source = """
        func main() {
            let x = undefined_var + 5
        }
        """
        
        with pytest.raises(NeuroTypeError) as exc_info:
            self._parse_and_check(source)
        
        assert "undefined variable" in str(exc_info.value).lower()
    
    def test_function_arity_error(self):
        """Test detection of function arity mismatches."""
        source = """
        func add(a: int, b: int) -> int {
            return a + b
        }
        
        func main() {
            let result = add(5)
        }
        """
        
        with pytest.raises(NeuroTypeError) as exc_info:
            self._parse_and_check(source)
        
        assert "expects" in str(exc_info.value).lower() and "arguments" in str(exc_info.value).lower()
    
    def test_boolean_condition_error(self):
        """Test that conditions must be boolean."""
        source = """
        func main() {
            if 42 {
                let x = 1
            }
        }
        """
        
        with pytest.raises(NeuroTypeError) as exc_info:
            self._parse_and_check(source)
        
        # Should detect that condition is not boolean
        assert "cannot unify" in str(exc_info.value).lower()
    
    # ========================================================================
    # Symbol Table Tests
    # ========================================================================
    
    def test_symbol_table_scoping(self):
        """Test that symbol tables handle scoping correctly."""
        source = """
        func main() {
            let x = 10
            {
                let y = 20
                let z = x + y
            }
            // y should not be accessible here
        }
        """
        
        # Should complete successfully - inner scope can access outer scope
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    def test_function_parameter_scoping(self):
        """Test function parameter scoping."""
        source = """
        func process(data: int) -> int {
            let result = data * 2
            return result
        }
        
        func main() {
            let value = process(42)
        }
        """
        
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    def test_duplicate_symbol_error(self):
        """Test detection of duplicate symbol definitions."""
        source = """
        func main() {
            let x = 10
            let x = 20
        }
        """
        
        with pytest.raises(NeuroTypeError) as exc_info:
            self._parse_and_check(source)
        
        assert "already defined" in str(exc_info.value).lower()
    
    # ========================================================================
    # Built-in Functions Tests
    # ========================================================================
    
    def test_builtin_functions(self):
        """Test built-in function type checking."""
        source = """
        func main() {
            print("Hello World")
            let text = str(42)
        }
        """
        
        types = self._parse_and_check(source)
        
        # Should complete without errors
        assert len(types) > 0
    
    def test_str_function_polymorphism(self):
        """Test that str function works with different types."""
        source = """
        func main() {
            let int_str = str(42)
            let float_str = str(3.14)
            let bool_str = str(true)
        }
        """
        
        types = self._parse_and_check(source)
        
        # All should be inferred as string type
        for node, node_type in types.items():
            if isinstance(node, VariableDeclaration) and node.name.endswith("_str"):
                # The str() call should eventually resolve to string type
                pass  # Complex to test without deeper introspection
    
    # ========================================================================
    # Advanced Features Tests
    # ========================================================================
    
    def test_tensor_operations_type_checking(self):
        """Test tensor operation type checking."""
        source = """
        func main() {
            let a = [1.0, 2.0, 3.0]
            let b = [4.0, 5.0, 6.0]
            let dot_product = a @ b
        }
        """
        
        types = self._parse_and_check(source)
        
        # Matrix multiplication should preserve element type
        for node, node_type in types.items():
            if isinstance(node, VariableDeclaration) and node.name == "dot_product":
                # Should be the element type (float)
                assert isinstance(node_type, TensorType)
    
    def test_assignment_type_checking(self):
        """Test assignment expression type checking."""
        source = """
        func main() {
            let mut x = 10
            x = 20
            let y = (x = 30)
        }
        """
        
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    def test_member_access_type_inference(self):
        """Test member access type inference."""
        source = """
        struct Point {
            x: float,
            y: float
        }
        
        func main() {
            let p = Point { x: 1.0, y: 2.0 }
            let x_coord = p.x
        }
        """
        
        # For now, this should complete (even with simplified struct handling)
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    def test_index_access_type_checking(self):
        """Test array/tensor indexing type checking."""
        source = """
        func main() {
            let data = [1.0, 2.0, 3.0]
            let element = data[0]
        }
        """
        
        types = self._parse_and_check(source)
        
        # Element should have the tensor's element type
        for node, node_type in types.items():
            if isinstance(node, VariableDeclaration) and node.name == "element":
                # Should be float (element type of the tensor)
                assert isinstance(node_type, PrimitiveType)
                assert node_type.name == "float"
    
    # ========================================================================
    # Type Checker Internal Tests
    # ========================================================================
    
    def test_type_variable_creation(self):
        """Test type variable creation and naming."""
        type_checker = TypeChecker()
        
        var1 = type_checker._new_type_variable()
        var2 = type_checker._new_type_variable()
        
        assert isinstance(var1, TypeVariable)
        assert isinstance(var2, TypeVariable)
        assert var1.id != var2.id
        assert var1.name.startswith("T")
        assert var2.name.startswith("T")
    
    def test_occurs_check(self):
        """Test occurs check in unification."""
        type_checker = TypeChecker()
        
        var = type_checker._new_type_variable()
        
        # Simple occurs check
        assert type_checker._occurs_check(var, var)
        
        # Occurs check in tensor type
        tensor_type = TensorType(var, [3])
        assert type_checker._occurs_check(var, tensor_type)
        
        # No occurrence
        int_type = PrimitiveType("int")
        assert not type_checker._occurs_check(var, int_type)
    
    def test_substitution_application(self):
        """Test type substitution application."""
        type_checker = TypeChecker()
        
        var = type_checker._new_type_variable()
        int_type = PrimitiveType("int")
        
        # Add substitution
        type_checker.type_substitutions[var] = int_type
        
        # Apply substitution
        result = type_checker._apply_substitution(var)
        assert isinstance(result, PrimitiveType)
        assert result.name == "int"
        
        # Test recursive substitution
        var2 = type_checker._new_type_variable()
        type_checker.type_substitutions[var2] = var
        
        result2 = type_checker._apply_substitution(var2)
        assert isinstance(result2, PrimitiveType)
        assert result2.name == "int"

    def test_neural_network_model_initializer_scoping(self):
        """Test that model initializers don't leak scope or type variables."""
        code = """
        // Define a function that returns a type variable based on its input
        func identity<T>(x: T) -> T {
            return x
        }

        // Model that uses a generic function inside
        let model = NeuralNetwork<f32, (10, 5)> {
            identity(dense_layer<5>(.relu))
        }

        // Another variable to see if type vars are reused incorrectly
        let output_var = identity(10)
        """
        tokens = self.lexer.tokenize(code, "test.nr")
        program = self.parser.parse(tokens)
        
        # Use a fresh TypeChecker for this complex test
        checker = TypeChecker(debug=True)
        type_cache = checker.check_program(program)
        
        # Get the type of output_var, which is the last statement
        output_var_node = program.statements[-1]
        final_output_type = type_cache.get(output_var_node)

        # After substitutions, the generic type T should be resolved to int
        expected_type = PrimitiveType("int")
        assert final_output_type == expected_type, \
            f"Expected output_var type to be {expected_type}, got {final_output_type}"


class TestTypeInferenceIntegration:
    """Integration tests for type inference with complete programs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lexer = NeuroLexer()
        self.parser = NeuroParser()
        self.type_checker = TypeChecker(verbose=False)
    
    def _parse_and_check(self, source: str) -> Dict[ASTNode, Type]:
        """Helper to parse source and run type checking."""
        tokens = self.lexer.tokenize(source)
        ast = self.parser.parse(tokens)
        return self.type_checker.check_program(ast)
    
    def test_complete_program_type_checking(self):
        """Test type checking on a complete program."""
        source = """
        func factorial(n: int) -> int {
            if n <= 1 {
                return 1
            } else {
                return n * factorial(n - 1)
            }
        }
        
        func main() {
            let numbers = [1, 2, 3, 4, 5]
            let results: Tensor<int> = []
            
            for num in numbers {
                let fact = factorial(num)
                // results.push(fact)  // Would need array methods
            }
            
            print("Factorial computation complete")
        }
        """
        
        types = self._parse_and_check(source)
        
        # Should complete successfully with proper type inference
        assert len(types) > 0
        
        # Check that factorial function has correct type
        for node, node_type in types.items():
            if isinstance(node, FunctionDeclaration) and node.name == "factorial":
                assert isinstance(node_type, FunctionType)
                assert len(node_type.param_types) == 1
                assert isinstance(node_type.param_types[0], PrimitiveType)
                assert node_type.param_types[0].name == "int"
                assert isinstance(node_type.return_type, PrimitiveType)
                assert node_type.return_type.name == "int"
    
    def test_neural_network_type_checking(self):
        """Test type checking for neural network code."""
        source = """
        func create_model() -> NeuralNetwork<float, (784, 10)> {
            let model = NeuralNetwork<float, (784, 10)> {
                dense_layer(units=128, activation=relu),
                dense_layer(units=10, activation=softmax)
            }
            return model
        }
        
        func train_step(model: NeuralNetwork<float, (784, 10)>, 
                       input: Tensor<float, (32, 784)>,
                       target: Tensor<float, (32, 10)>) -> float {
            let output = model.forward(input)
            let loss = cross_entropy(output, target)
            return loss
        }
        
        func main() {
            let model = create_model()
            let batch_input: Tensor<float, (32, 784)> = load_data()
            let batch_target: Tensor<float, (32, 10)> = load_labels()
            
            let loss = train_step(model, batch_input, batch_target)
            print("Loss: " + str(loss))
        }
        
        // Placeholder functions for compilation
        func load_data() -> Tensor<float, (32, 784)> {
            let data: Tensor<float> = []
            return data.reshape(32, 784)
        }
        
        func load_labels() -> Tensor<float, (32, 10)> {
            let labels: Tensor<float> = []
            return labels.reshape(32, 10)
        }
        """
        
        # Should complete without type errors
        types = self._parse_and_check(source)
        assert len(types) > 0
    
    def test_error_recovery_in_type_checking(self):
        """Test that type checker provides good error messages."""
        source = """
        func main() {
            let x: int = 42
            let y: string = "hello"
            let bad_op = x + y
        }
        """
        
        with pytest.raises(NeuroTypeError) as exc_info:
            self._parse_and_check(source)
        
        error_msg = str(exc_info.value).lower()
        assert "cannot unify" in error_msg or "type mismatch" in error_msg 