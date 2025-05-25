"""
Test suite for all NEURO example programs.

This module tests that all example programs in the examples/ directory
can be successfully parsed and compiled through the NEURO compiler pipeline.
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro.lexer import NeuroLexer
from neuro.parser import NeuroParser
from neuro.compiler import NeuroCompiler
from neuro.ast_nodes import *


class TestAllExamples:
    """Test suite for NEURO example programs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lexer = NeuroLexer()
        self.parser = NeuroParser()
        self.compiler = NeuroCompiler()
        self.examples_dir = Path(__file__).parent.parent / "examples"
    
    def parse_file(self, filepath: Path) -> Program:
        """Parse a NEURO source file and return the AST."""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tokens = self.lexer.tokenize(source, str(filepath))
        ast = self.parser.parse(tokens)
        return ast
    
    def test_hello_world_parsing(self):
        """Test that hello_world.nr parses correctly."""
        hello_world_path = self.examples_dir / "hello_world.nr"
        
        # Ensure the file exists
        assert hello_world_path.exists(), f"hello_world.nr not found at {hello_world_path}"
        
        # Parse the file
        ast = self.parse_file(hello_world_path)
        
        # Verify basic structure
        assert isinstance(ast, Program)
        assert len(ast.statements) == 3  # main, add, create_simple_tensor functions
        
        # Check main function
        main_func = ast.statements[0]
        assert isinstance(main_func, FunctionDeclaration)
        assert main_func.name == "main"
        assert len(main_func.parameters) == 0
        assert main_func.return_type is None
        assert isinstance(main_func.body, Block)
        
        # Check add function
        add_func = ast.statements[1]
        assert isinstance(add_func, FunctionDeclaration)
        assert add_func.name == "add"
        assert len(add_func.parameters) == 2
        assert add_func.parameters[0].name == "a"
        assert add_func.parameters[1].name == "b"
        assert isinstance(add_func.return_type, PrimitiveType)
        assert add_func.return_type.name == "int"
        
        # Check tensor function
        tensor_func = ast.statements[2]
        assert isinstance(tensor_func, FunctionDeclaration)
        assert tensor_func.name == "create_simple_tensor"
        assert len(tensor_func.parameters) == 0
        assert isinstance(tensor_func.return_type, TensorType)
    
    def test_hello_world_compilation(self):
        """Test that hello_world.nr compiles without errors."""
        hello_world_path = self.examples_dir / "hello_world.nr"
        ast = self.parse_file(hello_world_path)
        
        # Test compilation pipeline through the compiler
        try:
            # Use the compile method directly with AST
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
                output_path = tmp.name
            
            self.compiler.compile(ast, output_path)
            # Should complete without raising exceptions
            
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)
            ll_file = Path(output_path).with_suffix('.ll')
            if ll_file.exists():
                ll_file.unlink()
                
        except Exception as e:
            pytest.fail(f"Compilation failed: {e}")
    
    def test_hello_world_language_features(self):
        """Test specific language features used in hello_world.nr."""
        hello_world_path = self.examples_dir / "hello_world.nr"
        ast = self.parse_file(hello_world_path)
        
        main_func = ast.statements[0]
        main_body = main_func.body.statements
        
        # Check variable declarations
        message_decl = main_body[0]
        assert isinstance(message_decl, VariableDeclaration)
        assert message_decl.name == "message"
        assert isinstance(message_decl.initializer, Literal)
        assert message_decl.initializer.value == "Hello, NEURO World!"
        
        # Check function calls
        print_call = main_body[1]
        assert isinstance(print_call, ExpressionStatement)
        assert isinstance(print_call.expression, FunctionCall)
        print_func = print_call.expression
        assert isinstance(print_func.function, Identifier)
        assert print_func.function.name == "print"
        
        # Check arithmetic expressions
        x_decl = main_body[2]
        y_decl = main_body[3]
        z_decl = main_body[4]
        
        assert isinstance(x_decl, VariableDeclaration)
        assert x_decl.name == "x"
        assert x_decl.initializer.value == 42
        
        assert isinstance(y_decl, VariableDeclaration)
        assert y_decl.name == "y"
        assert y_decl.initializer.value == 3.14
        
        assert isinstance(z_decl, VariableDeclaration)
        assert z_decl.name == "z"
        assert isinstance(z_decl.initializer, BinaryOp)
        assert z_decl.initializer.operator == "+"
    
    def test_all_example_files_exist(self):
        """Test that all expected example files exist."""
        expected_examples = [
            "hello_world.nr"
        ]
        
        for example in expected_examples:
            example_path = self.examples_dir / example
            assert example_path.exists(), f"Example file {example} not found"
    
    def test_all_example_files_parse(self):
        """Test that all .nr files in examples/ directory parse successfully."""
        if not self.examples_dir.exists():
            pytest.skip("Examples directory not found")
        
        nr_files = list(self.examples_dir.glob("*.nr"))
        assert len(nr_files) > 0, "No .nr files found in examples directory"
        
        for nr_file in nr_files:
            print(f"Testing parsing of {nr_file.name}")
            try:
                ast = self.parse_file(nr_file)
                assert isinstance(ast, Program)
                assert len(ast.statements) > 0, f"Empty program in {nr_file.name}"
            except Exception as e:
                pytest.fail(f"Failed to parse {nr_file.name}: {e}")
    
    def test_syntax_highlighting_example(self):
        """Test that the hello_world.nr demonstrates key syntax features."""
        hello_world_path = self.examples_dir / "hello_world.nr"
        
        with open(hello_world_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Check that key language features are demonstrated
        assert "func main()" in source, "Missing main function"
        assert "let message = " in source, "Missing variable declaration"
        assert "func add(a: int, b: int) -> int" in source, "Missing typed function"
        assert "Tensor<float>" in source, "Missing tensor type"
        assert "return" in source, "Missing return statement"
        assert "//" in source, "Missing comments"
    
    def test_compiler_pipeline_integration(self):
        """Test the complete compiler pipeline with hello_world.nr."""
        hello_world_path = self.examples_dir / "hello_world.nr"
        
        with open(hello_world_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Test lexing
        tokens = self.lexer.tokenize(source, str(hello_world_path))
        assert len(tokens) > 0
        assert tokens[-1].type.name == "EOF"
        
        # Test parsing
        ast = self.parser.parse(tokens)
        assert isinstance(ast, Program)
        
        # Test AST pretty printing
        pretty = ast.pretty_print()
        assert isinstance(pretty, str)
        assert len(pretty) > 0
        assert "main" in pretty
        
        # Test that AST contains expected node types with proper traversal
        def count_node_types(node, counts=None):
            if counts is None:
                counts = {}
            
            # Count the current node
            node_type = type(node).__name__
            counts[node_type] = counts.get(node_type, 0) + 1
            
            # Recursively count child nodes
            if hasattr(node, 'statements') and isinstance(node.statements, list):
                for stmt in node.statements:
                    count_node_types(stmt, counts)
            
            if hasattr(node, 'body') and node.body is not None:
                count_node_types(node.body, counts)
            
            if hasattr(node, 'parameters') and isinstance(node.parameters, list):
                for param in node.parameters:
                    count_node_types(param, counts)
            
            if hasattr(node, 'return_type') and node.return_type is not None:
                count_node_types(node.return_type, counts)
            
            if hasattr(node, 'initializer') and node.initializer is not None:
                count_node_types(node.initializer, counts)
            
            if hasattr(node, 'expression') and node.expression is not None:
                count_node_types(node.expression, counts)
            
            if hasattr(node, 'left') and node.left is not None:
                count_node_types(node.left, counts)
            
            if hasattr(node, 'right') and node.right is not None:
                count_node_types(node.right, counts)
            
            if hasattr(node, 'function') and node.function is not None:
                count_node_types(node.function, counts)
            
            if hasattr(node, 'arguments') and isinstance(node.arguments, list):
                for arg in node.arguments:
                    count_node_types(arg, counts)
            
            return counts
        
        node_counts = count_node_types(ast)
        
        # Verify we have the expected AST node types
        assert node_counts.get('FunctionDeclaration', 0) >= 3, f"Should have at least 3 functions, got {node_counts.get('FunctionDeclaration', 0)}"
        assert node_counts.get('VariableDeclaration', 0) >= 4, f"Should have at least 4 variables, got {node_counts.get('VariableDeclaration', 0)}"
        assert node_counts.get('Literal', 0) >= 3, f"Should have several literals, got {node_counts.get('Literal', 0)}"
        assert node_counts.get('BinaryOp', 0) >= 1, f"Should have at least one binary operation, got {node_counts.get('BinaryOp', 0)}"


if __name__ == "__main__":
    pytest.main([__file__]) 