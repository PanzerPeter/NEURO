"""
NEURO Compiler

Main compiler class that orchestrates all compilation phases:
1. AST Processing and Validation
2. Type Checking and Inference
3. Optimization Passes
4. Code Generation (LLVM IR)
5. Linking and Output

This is the core of the NEURO compiler that transforms AST into executable code.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from .ast_nodes import Program, ASTNode
from .errors import CompilerError, SourceLocation
from .type_checker import TypeChecker


class NeuroCompiler:
    """
    Main compiler class for the NEURO programming language.
    
    Handles the complete compilation pipeline from AST to executable,
    including type checking, optimization, and code generation.
    """
    
    def __init__(self, optimization_level: int = 1, verbose: bool = False, debug: bool = False):
        """
        Initialize the NEURO compiler.
        
        Args:
            optimization_level: Optimization level (0-3)
            verbose: Enable verbose output
            debug: Enable debug mode output
        """
        self.optimization_level = optimization_level
        self.verbose = verbose
        self.debug = debug
        
        # Compilation state
        self.symbol_table: Dict[str, Any] = {}
        self.type_cache: Dict[ASTNode, str] = {}
        self.current_module = ""
        
        # Standard library and built-ins
        self._init_builtin_types()
        self._init_standard_library()
    
    def compile(self, ast: Program, output_path: str, emit_llvm_ir: bool = False) -> None:
        """
        Compile an AST to executable code.
        
        Args:
            ast: The AST to compile
            output_path: Path for the output executable
            emit_llvm_ir: Whether to emit LLVM IR instead of compiling
            
        Raises:
            CompilerError: If compilation fails
        """
        try:
            if self.verbose:
                print("Starting NEURO compilation...")
            
            # Phase 1: AST Validation and Preprocessing
            if self.verbose:
                print("Phase 1: AST Validation")
            self._validate_ast(ast)
            
            # Phase 2: Type Checking and Inference
            if self.verbose:
                print("Phase 2: Type Checking")
            self._type_check(ast)
            
            # Phase 3: Optimization
            if self.verbose:
                print("Phase 3: Optimization")
            optimized_ast = self._optimize(ast)
            
            # Phase 4: Code Generation
            if self.verbose:
                print("Phase 4: Code Generation")
            
            if emit_llvm_ir:
                llvm_ir = self._generate_llvm_ir(optimized_ast)
                print("=== LLVM IR ===")
                print(llvm_ir)
                return
            
            # Generate and compile to executable
            self._compile_to_executable(optimized_ast, output_path)
            
            if self.verbose:
                print(f"Compilation successful: {output_path}")
                
        except Exception as e:
            raise CompilerError(f"Compilation failed: {e}")
    
    # ========================================================================
    # Phase 1: AST Validation
    # ========================================================================
    
    def _validate_ast(self, ast: Program) -> None:
        """Validate the AST for basic correctness."""
        # Basic AST validation
        if not ast.statements:
            raise CompilerError("Empty program")
        
        # Check for duplicate function/struct names
        names = set()
        for stmt in ast.statements:
            if hasattr(stmt, 'name'):
                if stmt.name in names:
                    raise CompilerError(
                        f"Duplicate definition: {stmt.name}",
                        stmt.location
                    )
                names.add(stmt.name)
        
        if self.verbose:
            print(f"  ✓ AST validation passed ({len(ast.statements)} statements)")
    
    # ========================================================================
    # Phase 2: Type Checking
    # ========================================================================
    
    def _type_check(self, ast: Program) -> None:
        """Perform type checking and inference on the AST."""
        # Create type checker instance
        type_checker = TypeChecker(verbose=self.verbose, debug=self.debug)
        
        # Perform comprehensive type checking
        self.type_cache = type_checker.check_program(ast)
        
        if self.verbose:
            print(f"  ✓ Type checking completed - {len(self.type_cache)} types inferred")
    
    # ========================================================================
    # Phase 3: Optimization
    # ========================================================================
    
    def _optimize(self, ast: Program) -> Program:
        """Apply optimization passes to the AST."""
        # Placeholder for optimization passes
        # In a full implementation, this would include:
        # 1. Dead code elimination
        # 2. Constant folding
        # 3. Tensor operation fusion
        # 4. Memory layout optimization
        # 5. Loop optimization
        
        if self.verbose:
            level_name = ["none", "basic", "standard", "aggressive"][self.optimization_level]
            print(f"  ✓ Optimization level {self.optimization_level} ({level_name})")
        
        return ast
    
    # ========================================================================
    # Phase 4: Code Generation
    # ========================================================================
    
    def _generate_llvm_ir(self, ast: Program) -> str:
        """Generate LLVM IR from the optimized AST."""
        # Placeholder LLVM IR generation
        # This would be a full LLVM IR generator in the real implementation
        
        ir_lines = [
            "; NEURO Compiled Program",
            f"; Optimization Level: {self.optimization_level}",
            "",
            "target triple = \"x86_64-pc-windows-msvc\"",
            "",
            "declare i32 @printf(i8*, ...)",
            "",
            "@hello_str = private unnamed_addr constant [13 x i8] c\"Hello NEURO!\\00\"",
            "",
            "define i32 @main() {",
            "entry:",
            "  %0 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @hello_str, i32 0, i32 0))",
            "  ret i32 0",
            "}",
            ""
        ]
        
        return "\n".join(ir_lines)
    
    def _compile_to_executable(self, ast: Program, output_path: str) -> None:
        """Compile AST to executable via LLVM."""
        # Generate LLVM IR
        llvm_ir = self._generate_llvm_ir(ast)
        
        # Write IR to temporary file
        ir_file = Path(output_path).with_suffix('.ll')
        with open(ir_file, 'w') as f:
            f.write(llvm_ir)
        
        try:
            # Use clang to compile LLVM IR to executable
            # This is a simplified approach - real implementation would use LLVM APIs
            clang_cmd = [
                'clang',
                str(ir_file),
                '-O' + str(self.optimization_level),
                '-o', output_path
            ]
            
            if self.verbose:
                print(f"  Running: {' '.join(clang_cmd)}")
            
            result = subprocess.run(clang_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_message = f"""LLVM compilation failed.
Command: {' '.join(clang_cmd)}
Exit Code: {result.returncode}
--- Clang Stdout: ---
{result.stdout}
--- Clang Stderr: ---
{result.stderr}
"""
                raise CompilerError(error_message)
            
        except FileNotFoundError:
            # LLVM/clang not available
            raise CompilerError(
                "Compiler command 'clang' not found. "
                "Please install LLVM and ensure 'clang' is in your system's PATH."
            )
        
        # Clean up IR file
        if ir_file.exists():
            ir_file.unlink()
    
    def _create_c_fallback(self, output_path: str) -> None:
        """Create a simple C program as fallback when LLVM is not available."""
        c_code = '''
#include <stdio.h>

int main() {
    printf("Hello from NEURO! (compiled fallback)\\n");
    printf("This is a placeholder executable - full NEURO compilation requires LLVM\\n");
    return 0;
}
'''
        
        c_file = Path(output_path).with_suffix('.c')
        
        try:
            # Write C code
            with open(c_file, 'w') as f:
                f.write(c_code)
            
            # Compile with gcc/cl
            if os.name == 'nt':  # Windows
                compile_cmd = ['cl', str(c_file), '/Fe:' + output_path]
            else:  # Unix-like
                compile_cmd = ['gcc', str(c_file), '-o', output_path]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Create batch/shell script as final fallback
                self._create_script_fallback(output_path)
            
        except (FileNotFoundError, subprocess.SubprocessError):
            # Create script as final fallback
            self._create_script_fallback(output_path)
        
        # Clean up C file
        if c_file.exists():
            c_file.unlink()
    
    def _create_script_fallback(self, output_path: str) -> None:
        """Create a script file as final fallback."""
        if os.name == 'nt':  # Windows
            script_content = '@echo off\necho Hello from NEURO! (script fallback)\necho Full compilation requires LLVM and clang\npause\n'
            script_path = Path(output_path).with_suffix('.bat')
        else:  # Unix-like
            script_content = '#!/bin/bash\necho "Hello from NEURO! (script fallback)"\necho "Full compilation requires LLVM and clang"\n'
            script_path = Path(output_path).with_suffix('.sh')
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix-like systems
        if os.name != 'nt':
            os.chmod(script_path, 0o755)
        
        if self.verbose:
            print(f"  Created script fallback: {script_path}")
    
    # ========================================================================
    # Built-in Types and Standard Library
    # ========================================================================
    
    def _init_builtin_types(self) -> None:
        """Initialize built-in types in the compiler."""
        self.builtin_types = {
            'int': {'size': 32, 'signed': True},
            'float': {'size': 32, 'format': 'ieee754'},
            'bool': {'size': 1},
            'string': {'type': 'utf8'},
        }
        
        # Tensor types are handled dynamically
        
    def _init_standard_library(self) -> None:
        """Initialize standard library functions."""
        self.stdlib_functions = {
            'print': {
                'params': ['string'],
                'return': 'void',
                'builtin': True
            },
            'len': {
                'params': ['tensor'],
                'return': 'int',
                'builtin': True
            },
            # Neural network functions would be added here
            'relu': {
                'params': ['tensor'],
                'return': 'tensor',
                'gpu_available': True
            },
            'softmax': {
                'params': ['tensor'],
                'return': 'tensor',
                'gpu_available': True
            }
        } 