"""
NEURO Compiler
Orchestrates the compilation pipeline: lexing -> parsing -> type checking -> code generation.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

from .lexer import NeuroLexer, Token
from .parser import NeuroParser
from .ast_nodes import Program
from .type_checker import NeuroTypeChecker
from .errors import ErrorReporter, NeuroError


class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    O0 = auto()  # No optimization
    O1 = auto()  # Basic optimization
    O2 = auto()  # Standard optimization
    O3 = auto()  # Aggressive optimization


@dataclass
class CompileOptions:
    """Configuration options for compilation."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    target_platform: str = "cpu"  # cpu, cuda, opencl, etc.
    output_format: str = "executable"  # executable, llvm-ir, assembly
    debug_info: bool = False
    emit_tokens: bool = False
    emit_ast: bool = False
    emit_llvm_ir: bool = False
    verbose: bool = False
    output_file: Optional[str] = None


@dataclass
class CompileResult:
    """Result of compilation process."""
    success: bool
    program: Optional[Program] = None
    tokens: Optional[List[Token]] = None
    output_file: Optional[str] = None
    compilation_time: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class NeuroCompiler:
    """Main compiler class for the NEURO programming language."""
    
    def __init__(self, options: Optional[CompileOptions] = None):
        self.options = options or CompileOptions()
        self.error_reporter = ErrorReporter()
    
    def compile_file(self, filename: str) -> CompileResult:
        """Compile a NEURO source file."""
        start_time = time.time()
        
        try:
            # Read source file
            with open(filename, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            return self.compile_source(source_code, filename)
            
        except FileNotFoundError:
            self.error_reporter.error(f"File not found: {filename}")
            return CompileResult(
                success=False,
                compilation_time=time.time() - start_time,
                errors=[f"File not found: {filename}"]
            )
        except Exception as e:
            self.error_reporter.error(f"Error reading file {filename}: {e}")
            return CompileResult(
                success=False,
                compilation_time=time.time() - start_time,
                errors=[f"Error reading file {filename}: {e}"]
            )
    
    def compile_source(self, source_code: str, filename: str = "<input>") -> CompileResult:
        """Compile NEURO source code."""
        start_time = time.time()
        result = CompileResult(success=False)
        
        try:
            # Step 1: Lexical Analysis
            if self.options.verbose:
                print(f"Lexing {filename}...")
            
            lexer = NeuroLexer(source_code, filename)
            tokens = lexer.tokenize()
            
            if lexer.get_error_reporter().has_errors():
                result.errors.extend([str(e) for e in lexer.get_error_reporter().errors])
                return result
            
            if self.options.emit_tokens:
                self._emit_tokens(tokens)
            
            result.tokens = tokens
            
            # Step 2: Parsing
            if self.options.verbose:
                print(f"Parsing {filename}...")
            
            parser = NeuroParser(tokens)
            program = parser.parse()
            
            if parser.get_error_reporter().has_errors():
                result.errors.extend([str(e) for e in parser.get_error_reporter().errors])
                return result
            
            if self.options.emit_ast:
                self._emit_ast(program)
            
            result.program = program
            
            # Step 3: Type Checking
            if self.options.verbose:
                print(f"Type checking {filename}...")
            
            type_checker = NeuroTypeChecker()
            typed_program = type_checker.check_program(program)
            
            if type_checker.get_error_reporter().has_errors():
                result.errors.extend([str(e) for e in type_checker.get_error_reporter().errors])
                return result
            
            # Collect warnings
            if type_checker.get_error_reporter().has_warnings():
                result.warnings.extend(type_checker.get_error_reporter().warnings)
            
            # Step 4: Code Generation
            if self.options.verbose:
                print(f"Generating code for {filename}...")
            
            output_file = self._generate_code(typed_program, filename)
            result.output_file = output_file
            
            result.success = True
            
        except NeuroError as e:
            result.errors.append(str(e))
        except Exception as e:
            result.errors.append(f"Internal compiler error: {e}")
            # Print traceback for debugging
            if self.options.verbose:
                import traceback
                traceback.print_exc()
        
        result.compilation_time = time.time() - start_time
        
        if self.options.verbose:
            self._print_compilation_summary(result)
        
        return result
    
    def _emit_tokens(self, tokens: List[Token]) -> None:
        """Emit tokens for debugging."""
        print("=== TOKENS ===")
        for i, token in enumerate(tokens):
            print(f"{i:3d}: {token}")
        print()
    
    def _emit_ast(self, program: Program) -> None:
        """Emit AST for debugging."""
        print("=== AST ===")
        print(f"Program with {len(program.declarations)} declarations and {len(program.statements)} statements")
        
        for i, decl in enumerate(program.declarations):
            print(f"Declaration {i}: {decl}")
        
        for i, stmt in enumerate(program.statements):
            print(f"Statement {i}: {stmt}")
        print()
    
    def _generate_code(self, program: Program, source_filename: str) -> str:
        """Generate code from the typed AST."""
        # For now, generate a simple interpretable format
        # In a full implementation, this would generate LLVM IR or machine code
        
        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        
        if self.options.output_file:
            output_file = self.options.output_file
        else:
            if self.options.output_format == "llvm-ir":
                output_file = f"{base_name}.ll"
            elif self.options.output_format == "assembly":
                output_file = f"{base_name}.s"
            else:
                # Executable - use .bat for Windows since we generate batch wrappers
                output_file = f"{base_name}.bat" if sys.platform == "win32" else base_name
        
        if self.options.output_format == "llvm-ir":
            self._generate_llvm_ir(program, output_file)
        elif self.options.output_format == "assembly":
            self._generate_assembly(program, output_file)
        else:
            self._generate_executable(program, output_file)
        
        return output_file
    
    def _generate_llvm_ir(self, program: Program, output_file: str) -> None:
        """Generate LLVM IR code."""
        # Simplified LLVM IR generation
        llvm_code = self._generate_basic_llvm(program)
        
        with open(output_file, 'w') as f:
            f.write(llvm_code)
        
        if self.options.emit_llvm_ir:
            print("=== LLVM IR ===")
            print(llvm_code)
    
    def _generate_assembly(self, program: Program, output_file: str) -> None:
        """Generate assembly code."""
        # For now, just create a stub assembly file
        asm_code = f"""
; Generated by NEURO compiler
; Source: {program}

.section .text
.globl _start

_start:
    ; Program entry point
    mov $60, %rax    ; sys_exit
    mov $0, %rdi     ; exit status
    syscall          ; invoke system call
"""
        
        with open(output_file, 'w') as f:
            f.write(asm_code)
    
    def _generate_executable(self, program: Program, output_file: str) -> None:
        """Generate executable code."""
        # For now, generate a Python script that can interpret the AST
        # In a full implementation, this would compile to machine code
        
        python_code = self._generate_python_interpreter(program)
        
        # Write Python file
        py_file = output_file + ".py"
        with open(py_file, 'w') as f:
            f.write(python_code)
        
        # Create executable wrapper script
        if sys.platform == "win32":
            batch_content = f"@echo off\npython {py_file} %*\n"
            with open(output_file, 'w') as f:
                f.write(batch_content)
        else:
            script_content = f"#!/bin/bash\npython3 {py_file} \"$@\"\n"
            with open(output_file, 'w') as f:
                f.write(script_content)
            os.chmod(output_file, 0o755)  # Make executable
    
    def _generate_basic_llvm(self, program: Program) -> str:
        """Generate basic LLVM IR."""
        # Simplified LLVM IR generation
        llvm_code = """
; Generated by NEURO compiler
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"Hello World!\\00", align 1

declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0))
  ret i32 0
}
"""
        return llvm_code
    
    def _generate_python_interpreter(self, program: Program) -> str:
        """Generate Python code that interprets the NEURO program."""
        # Create a simple Python interpreter for the AST
        code = f'''#!/usr/bin/env python3
"""
Generated NEURO program interpreter
"""

import sys
import math

# Built-in functions
def neuro_print(value):
    print(str(value))
    return str(value)

def neuro_str(value):
    return str(value)

def neuro_float(value):
    return float(value)

# Global variables
globals_dict = {{}}

# Tensor operations (simplified)
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, list):
            self.data = data
            if shape is None:
                self.shape = self._infer_shape(data)
            else:
                self.shape = shape
        else:
            self.data = data
            self.shape = shape or []
    
    def _infer_shape(self, data):
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                return [len(data), len(data[0])]
            else:
                return [len(data)]
        return []
    
    def __matmul__(self, other):
        # Simple dot product for vectors
        if isinstance(other, Tensor) and len(self.shape) == 1 and len(other.shape) == 1:
            return sum(a * b for a, b in zip(self.data, other.data))
        return Tensor(self.data)  # Simplified
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor([a + b for a, b in zip(self.data, other.data)])
        return Tensor([a + other for a in self.data])
    
    def __str__(self):
        return f"Tensor({{self.data}})"

def zeros(rows, cols):
    return Tensor([[0.0 for _ in range(cols)] for _ in range(rows)])

# Neural Network (simplified)
class NeuralNetwork:
    def __init__(self, element_type, shape_params, layers):
        self.element_type = element_type
        self.shape_params = shape_params
        self.layers = layers
    
    def forward(self, input_data):
        # Simplified forward pass
        return Tensor([0.0] * self.shape_params[-1])

def dense_layer(units, activation=None):
    return {{"type": "dense", "units": units, "activation": activation}}

def batch_norm():
    return {{"type": "batch_norm"}}

def dropout(rate):
    return {{"type": "dropout", "rate": rate}}

# Main program execution
def main():
    try:
{self._generate_program_body(program)}
        
        # Execute main function if it exists
        if 'main' in globals_dict:
            globals_dict['main']()
        
    except Exception as e:
        print(f"Runtime error: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return code
    
    def _generate_program_body(self, program: Program) -> str:
        """Generate the main body of the Python interpreter."""
        # Simplified code generation for declarations and statements
        lines = []
        
        # Generate function declarations
        for decl in program.declarations:
            if hasattr(decl, 'name'):
                lines.append(f"        # Function: {decl.name}")
                lines.append(f"        def {decl.name}():")
                lines.append(f"            pass  # Function body would be generated here")
                lines.append(f"        globals_dict['{decl.name}'] = {decl.name}")
                lines.append("")
        
        # Generate top-level statements
        for stmt in program.statements:
            lines.append(f"        # Statement: {type(stmt).__name__}")
            if hasattr(stmt, 'name'):
                lines.append(f"        # Variable: {stmt.name}")
            lines.append(f"        pass  # Statement would be generated here")
        
        return "\n".join(lines)
    
    def _print_compilation_summary(self, result: CompileResult) -> None:
        """Print compilation summary."""
        print(f"\n=== COMPILATION SUMMARY ===")
        print(f"Success: {result.success}")
        print(f"Time: {result.compilation_time:.3f}s")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print(f"Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.output_file:
            print(f"Output: {result.output_file}")
        
        print()
    
    def get_error_reporter(self) -> ErrorReporter:
        """Get the error reporter."""
        return self.error_reporter 