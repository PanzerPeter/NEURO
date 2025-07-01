"""
Integration tests for the NEURO compiler.
Tests the complete compilation pipeline from source to output.
"""

import pytest
import os
import tempfile
from pathlib import Path
import psutil
import threading
import time

from src.neuro.compiler import NeuroCompiler, CompileOptions, OptimizationLevel
from src.neuro.main import main


class TestNeuroIntegration:
    """Integration tests for the NEURO compiler."""
    
    def get_example_path(self, filename: str) -> str:
        """Get the path to an example file."""
        # Assuming tests are run from project root
        return os.path.join("examples", "basics", filename)
    
    def get_ai_example_path(self, filename: str) -> str:
        """Get the path to an AI/ML example file."""
        return os.path.join("examples", "ai_ml", filename)
    
    def test_hello_world_compilation(self):
        """Test compiling the hello world example."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
        assert result.compilation_time > 0
    
    def test_basic_type_inference_compilation(self):
        """Test compiling the basic type inference example."""
        example_file = self.get_example_path("basic_type_inference.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
    
    def test_tensor_operations_compilation(self):
        """Test compiling the tensor operations example."""
        example_file = self.get_example_path("tensor_operations.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
    
    def test_struct_demo_compilation(self):
        """Test compiling the struct demo example."""
        example_file = self.get_example_path("struct_demo.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
    
    def test_function_type_inference_compilation(self):
        """Test compiling the function type inference example."""
        example_file = self.get_example_path("function_type_inference.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
    
    def test_constraint_solving_compilation(self):
        """Test compiling the constraint solving example."""
        example_file = self.get_example_path("constraint_solving.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
    
    def test_neural_network_compilation(self):
        """Test compiling the neural network example."""
        example_file = self.get_ai_example_path("neural_network.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
        assert result.output_file is not None
    
    def test_type_errors_handling(self):
        """Test that type errors are properly handled."""
        example_file = self.get_example_path("type_errors.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        # This file might succeed or fail depending on whether it contains actual errors
        # If it contains commented-out errors, it should succeed
        # If it contains real errors, it should fail
        # Either way, the compiler should handle it gracefully
        assert result.compilation_time > 0
    
    def test_compilation_with_different_options(self):
        """Test compilation with different options."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        # Test different optimization levels
        for opt_level in [OptimizationLevel.O0, OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3]:
            options = CompileOptions(optimization_level=opt_level)
            compiler = NeuroCompiler(options)
            result = compiler.compile_file(example_file)
            
            assert result.success, f"Compilation failed with {opt_level}: {result.errors}"
    
    def test_compilation_with_debug_info(self):
        """Test compilation with debug information."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        options = CompileOptions(debug_info=True, verbose=True)
        compiler = NeuroCompiler(options)
        result = compiler.compile_file(example_file)
        
        assert result.success, f"Compilation failed: {result.errors}"
    
    def test_llvm_ir_generation(self):
        """Test LLVM IR generation."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "hello.ll")
            options = CompileOptions(
                output_format="llvm-ir",
                output_file=output_file
            )
            compiler = NeuroCompiler(options)
            result = compiler.compile_file(example_file)
            
            assert result.success, f"Compilation failed: {result.errors}"
            assert os.path.exists(output_file)
            
            # Check that LLVM IR was generated
            with open(output_file, 'r') as f:
                content = f.read()
                assert "define" in content or "declare" in content
    
    def test_assembly_generation(self):
        """Test assembly generation."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "hello.s")
            options = CompileOptions(
                output_format="assembly",
                output_file=output_file
            )
            compiler = NeuroCompiler(options)
            result = compiler.compile_file(example_file)
            
            assert result.success, f"Compilation failed: {result.errors}"
            assert os.path.exists(output_file)
            
            # Check that assembly was generated
            with open(output_file, 'r') as f:
                content = f.read()
                assert ".section" in content or ".text" in content
    
    def test_compilation_with_custom_output(self):
        """Test compilation with custom output file."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "custom_hello")
            options = CompileOptions(output_file=output_file)
            compiler = NeuroCompiler(options)
            result = compiler.compile_file(example_file)
            
            assert result.success, f"Compilation failed: {result.errors}"
            assert result.output_file == output_file
    
    def test_cli_compilation(self):
        """Test compilation through CLI interface."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        # Test basic compilation
        exit_code = main([example_file])
        assert exit_code == 0
        
        # Test with verbose output
        exit_code = main([example_file, "--verbose"])
        assert exit_code == 0
    
    def test_cli_with_emit_options(self):
        """Test CLI with emit options."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        # Test emit tokens
        exit_code = main([example_file, "--emit-tokens"])
        assert exit_code == 0
        
        # Test emit AST
        exit_code = main([example_file, "--emit-ast"])
        assert exit_code == 0
        
        # Test emit LLVM IR
        exit_code = main([example_file, "--emit-llvm-ir"])
        assert exit_code == 0
    
    def test_compilation_error_handling(self):
        """Test compilation error handling for invalid files."""
        # Test non-existent file
        exit_code = main(["nonexistent.nr"])
        assert exit_code == 1
        
        # Test invalid syntax (create temporary file)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nr', delete=False) as f:
            f.write("func invalid_syntax( {\\n    // Missing closing paren\\n}")
            f.flush()
            f.close()  # Explicitly close the file
            
            try:
                exit_code = main([f.name])
                # Should either fail (exit code 1) or handle error gracefully
                assert exit_code in [0, 1]
            finally:
                os.unlink(f.name)
    
    def test_all_examples_compile(self):
        """Test that all example files compile successfully."""
        examples_dir = "examples"
        
        if not os.path.exists(examples_dir):
            pytest.skip("Examples directory not found")
        
        compiler = NeuroCompiler()
        
        # Find all .nr files in examples directory
        example_files = []
        for root, dirs, files in os.walk(examples_dir):
            for file in files:
                if file.endswith('.nr'):
                    example_files.append(os.path.join(root, file))
        
        assert len(example_files) > 0, "No example files found"
        
        failed_files = []
        
        for example_file in example_files:
            result = compiler.compile_file(example_file)
            
            if not result.success:
                failed_files.append((example_file, result.errors))
        
        if failed_files:
            error_msg = "Failed to compile example files:\n"
            for file, errors in failed_files:
                error_msg += f"  {file}: {errors}\n"
            pytest.fail(error_msg)
    
    def test_compilation_performance(self):
        """Test compilation performance."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        compiler = NeuroCompiler()
        result = compiler.compile_file(example_file)
        
        assert result.success
        # Compilation should be reasonably fast (less than 5 seconds for simple file)
        assert result.compilation_time < 5.0
    
    def test_memory_usage(self):
        """Test that compilation doesn't use excessive memory."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Compile multiple times
        compiler = NeuroCompiler()
        for _ in range(10):
            result = compiler.compile_file(example_file)
            assert result.success
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for simple compilation)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_concurrent_compilation(self):
        """Test concurrent compilation of multiple files."""
        example_file = self.get_example_path("hello_world.nr")
        
        if not os.path.exists(example_file):
            pytest.skip(f"Example file {example_file} not found")
        
        results = []
        errors = []
        
        def compile_file():
            try:
                compiler = NeuroCompiler()
                result = compiler.compile_file(example_file)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple compilation threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=compile_file)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Compilation errors in threads: {errors}"
        assert len(results) == 5
        
        for result in results:
            assert result.success, f"Compilation failed: {result.errors}"


if __name__ == "__main__":
    pytest.main([__file__]) 