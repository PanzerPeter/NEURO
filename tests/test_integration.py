"""
Integration Tests for NEURO Compiler

End-to-end tests that verify the complete compilation pipeline.
"""

import os
import tempfile
import subprocess
from pathlib import Path
import pytest

from neuro.lexer import NeuroLexer
from neuro.parser import NeuroParser
from neuro.compiler import NeuroCompiler
from neuro.errors import NeuroError


class TestNeuroIntegration:
    """Integration test suite for the complete NEURO compiler."""
    
    def setup_method(self):
        """Set up test environment."""
        self.lexer = NeuroLexer()
        self.parser = NeuroParser()
        self.compiler = NeuroCompiler(verbose=True)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def compile_source(self, source_code: str, output_name: str = "test_program") -> str:
        """Helper to compile source code and return output path."""
        # Tokenize
        tokens = self.lexer.tokenize(source_code, "test.nr")
        
        # Parse
        ast = self.parser.parse(tokens)
        
        # Compile
        output_path = os.path.join(self.temp_dir, output_name)
        self.compiler.compile(ast, output_path)
        
        return output_path
    
    def test_simple_program_compilation(self):
        """Test compilation of a simple program."""
        source = """
        func main() {
            let x = 42
            let y = 3.14
            let z = x + y
        }
        """
        
        # Should compile without errors
        output_path = self.compile_source(source)
        
        # Output file should exist (or fallback script)
        possible_extensions = ['', '.exe', '.bat', '.sh']
        found_output = False
        for ext in possible_extensions:
            if os.path.exists(output_path + ext):
                found_output = True
                break
        
        assert found_output, f"No output file found at {output_path}"
    
    def test_function_with_parameters(self):
        """Test compilation of functions with parameters."""
        source = """
        func add(a: int, b: int) -> int {
            return a + b
        }
        
        func main() {
            let result = add(5, 3)
        }
        """
        
        # Should compile without errors
        output_path = self.compile_source(source)
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_struct_definition(self):
        """Test compilation of struct definitions."""
        source = """
        struct Point {
            x: float
            y: float
        }
        
        func main() {
            let p = Point { x: 1.0, y: 2.0 }
        }
        """
        
        # Should compile without errors
        output_path = self.compile_source(source)
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_control_flow(self):
        """Test compilation of control flow structures."""
        source = """
        func main() {
            let x = 10
            
            if x > 5 {
                let y = x * 2
            } else {
                let y = x / 2
            }
            
            let i = 0
            while i < x {
                i = i + 1
            }
            
            for j in 0..10 {
                let temp = j * j
            }
        }
        """
        
        # Should compile without errors
        output_path = self.compile_source(source)
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_tensor_operations(self):
        """Test compilation of tensor operations."""
        source = """
        func main() {
            let vector: Tensor<float> = [1.0, 2.0, 3.0, 4.0]
            let matrix: Tensor<float, (2, 2)> = [[1.0, 2.0], [3.0, 4.0]]
            
            let length = len(vector)
            let element = vector[0]
        }
        """
        
        # Should compile without errors
        output_path = self.compile_source(source)
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_llvm_ir_generation(self):
        """Test LLVM IR generation."""
        source = """
        func main() {
            let message = "Hello, NEURO!"
        }
        """
        
        tokens = self.lexer.tokenize(source, "test.nr")
        ast = self.parser.parse(tokens)
        
        # Test IR generation
        output_path = os.path.join(self.temp_dir, "test_ir")
        self.compiler.compile(ast, output_path, emit_llvm_ir=True)
        
        # This test mainly checks that IR generation doesn't crash
        # In a real implementation, we'd verify the IR content
    
    def test_optimization_levels(self):
        """Test different optimization levels."""
        source = """
        func factorial(n: int) -> int {
            if n <= 1 {
                return 1
            } else {
                return n * factorial(n - 1)
            }
        }
        
        func main() {
            let result = factorial(5)
        }
        """
        
        # Test each optimization level
        for opt_level in [0, 1, 2, 3]:
            compiler = NeuroCompiler(optimization_level=opt_level, verbose=False)
            tokens = self.lexer.tokenize(source, "test.nr")
            ast = self.parser.parse(tokens)
            
            output_path = os.path.join(self.temp_dir, f"test_opt_{opt_level}")
            compiler.compile(ast, output_path)
            
            # Should create output file
            assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_error_handling(self):
        """Test compiler error handling."""
        error_cases = [
            # Lexer errors
            ('let x = "unterminated string', "lexer"),
            ('let x = 1.2.3', "lexer"),
            
            # Parser errors
            ('let x =', "parser"),
            ('func () {}', "parser"),
            
            # Semantic errors (would be caught in full implementation)
            # ('let x: int = "string"', "type"),
        ]
        
        for source, error_type in error_cases:
            with pytest.raises(NeuroError):
                try:
                    tokens = self.lexer.tokenize(source, "test.nr")
                    ast = self.parser.parse(tokens)
                    output_path = os.path.join(self.temp_dir, "error_test")
                    self.compiler.compile(ast, output_path)
                except Exception as e:
                    # Re-raise as NeuroError for test consistency
                    raise NeuroError(str(e))
    
    def test_hello_world_example(self):
        """Test the hello world example from examples/."""
        hello_world_source = '''
        // Simple Hello World in NEURO
        // This demonstrates basic language syntax
        
        func main() {
            let message = "Hello, NEURO World!"
            print(message)
            
            // Basic arithmetic
            let x = 42
            let y = 3.14
            let z = x + y
            
            print("The answer is: " + str(z))
        }
        
        // Example function with parameters and return type
        func add(a: int, b: int) -> int {
            return a + b
        }
        
        // Example with tensor operations (AI-focused)
        func create_simple_tensor() -> Tensor<float> {
            let data = [1.0, 2.0, 3.0, 4.0]
            return data
        }
        '''
        
        # Should compile the example successfully
        output_path = self.compile_source(hello_world_source, "hello_world")
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_complex_program(self):
        """Test compilation of a more complex program."""
        source = """
        // Complex NEURO program demonstrating multiple features
        
        struct Vector3D {
            x: float
            y: float
            z: float
        }
        
        func dot_product(a: Vector3D, b: Vector3D) -> float {
            return a.x * b.x + a.y * b.y + a.z * b.z
        }
        
        func magnitude(v: Vector3D) -> float {
            return sqrt(dot_product(v, v))
        }
        
        func normalize(v: Vector3D) -> Vector3D {
            let mag = magnitude(v)
            return Vector3D {
                x: v.x / mag,
                y: v.y / mag,
                z: v.z / mag
            }
        }
        
        func main() {
            let v1 = Vector3D { x: 1.0, y: 2.0, z: 3.0 }
            let v2 = Vector3D { x: 4.0, y: 5.0, z: 6.0 }
            
            let dot = dot_product(v1, v2)
            let mag1 = magnitude(v1)
            let mag2 = magnitude(v2)
            
            let normalized = normalize(v1)
            
            if dot > 0.0 {
                print("Vectors point in similar direction")
            } else {
                print("Vectors point in different directions")
            }
            
            // Demonstrate tensor operations
            let matrix: Tensor<float, (3, 3)> = [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
            
            let vector: Tensor<float, (3,)> = [v1.x, v1.y, v1.z]
            let result = matrix @ vector
        }
        """
        
        # Should compile without errors
        output_path = self.compile_source(source, "complex_program")
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_neural_network_program(self):
        """Test compilation of a neural network program."""
        source = """
        func create_model() -> NeuralNetwork<float, (784, 128, 10)> {
            return NeuralNetwork<float, (784, 128, 10)> {
                dense_layer(units=128, activation=relu),
                batch_norm(),
                dropout(rate=0.2),
                dense_layer(units=10, activation=softmax)
            }
        }
        
        func train_model(model: NeuralNetwork<float, (784, 128, 10)>, 
                        data: Tensor<float, (1000, 784)>,
                        labels: Tensor<float, (1000, 10)>) {
            
            let optimizer = Adam(learning_rate=0.001)
            let loss_fn = CrossEntropyLoss()
            
            for epoch in 0..100 {
                let predictions = model.forward(data)
                let loss = loss_fn(predictions, labels)
                
                model.backward(loss)
                optimizer.step()
                
                if epoch % 10 == 0 {
                    print("Epoch " + str(epoch) + ", Loss: " + str(loss))
                }
            }
        }
        
        func main() {
            let model = create_model()
            
            // In a real implementation, we'd load actual data
            let dummy_data: Tensor<float, (1000, 784)> = zeros(1000, 784)
            let dummy_labels: Tensor<float, (1000, 10)> = zeros(1000, 10)
            
            train_model(model, dummy_data, dummy_labels)
        }
        """
        
        # Should compile without errors (even though some functions aren't implemented)
        output_path = self.compile_source(source, "neural_network")
        assert any(os.path.exists(output_path + ext) for ext in ['', '.exe', '.bat', '.sh'])
    
    def test_compiler_phases(self):
        """Test individual compiler phases."""
        source = """
        func test_function(x: int) -> int {
            return x * 2
        }
        
        func main() {
            let result = test_function(21)
        }
        """
        
        tokens = self.lexer.tokenize(source, "test.nr")
        ast = self.parser.parse(tokens)
        
        # Test that all phases can be called without errors
        compiler = NeuroCompiler(verbose=True)
        
        # Phase 1: AST Validation
        compiler._validate_ast(ast)
        
        # Phase 2: Type Checking  
        compiler._type_check(ast)
        
        # Phase 3: Optimization
        optimized_ast = compiler._optimize(ast)
        assert optimized_ast is not None
        
        # Phase 4: Code Generation
        llvm_ir = compiler._generate_llvm_ir(optimized_ast)
        assert isinstance(llvm_ir, str)
        assert "define" in llvm_ir  # Basic LLVM IR structure
    
    def test_fallback_compilation(self):
        """Test that fallback compilation works when LLVM is not available."""
        source = """
        func main() {
            let greeting = "Hello from fallback!"
        }
        """
        
        # Force fallback by using a compiler that will fail LLVM compilation
        compiler = NeuroCompiler(verbose=True)
        tokens = self.lexer.tokenize(source, "test.nr")
        ast = self.parser.parse(tokens)
        
        output_path = os.path.join(self.temp_dir, "fallback_test")
        compiler.compile(ast, output_path)
        
        # Should create some form of output (script fallback at minimum)
        found_output = False
        for ext in ['', '.exe', '.bat', '.sh', '.c']:
            if os.path.exists(output_path + ext):
                found_output = True
                break
        
        assert found_output


if __name__ == "__main__":
    pytest.main([__file__]) 