"""
NEURO Type Checker and Inference Engine

Implements static type checking and type inference for the NEURO programming language.
This module provides:
1. Symbol table management
2. Type inference for expressions and statements
3. Type unification and constraint solving
4. Generic type instantiation
5. Tensor shape verification
"""

from typing import Dict, List, Optional, Set, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .ast_nodes import *
from .errors import CompilerError, TypeError as NeuroTypeError, SourceLocation


class TypeConstraint:
    """Represents a type constraint for inference."""
    
    def __init__(self, left: Type, right: Type, location: Optional[SourceLocation] = None):
        self.left = left
        self.right = right
        self.location = location
    
    def __repr__(self) -> str:
        return f"{self.left} ~ {self.right}"


@dataclass
class TypeVariable:
    """Type variable for inference."""
    name: str
    id: int
    constraints: List[Type] = field(default_factory=list)
    location: Optional[SourceLocation] = None
    
    def __repr__(self) -> str:
        return f"?{self.name}{self.id}"
    
    def __hash__(self) -> int:
        """Make type variables hashable using their unique ID."""
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        """Type variables are equal if they have the same ID."""
        return isinstance(other, TypeVariable) and self.id == other.id


@dataclass
class SymbolInfo:
    """Information about a symbol in the symbol table."""
    name: str
    symbol_type: Type
    is_mutable: bool = False
    is_function: bool = False
    location: Optional[SourceLocation] = None
    generic_params: Optional[List[GenericType]] = None
    generic_param_map: Optional[Dict[str, TypeVariable]] = None


class Scope:
    """Represents a lexical scope."""
    
    _scope_counter = 0 # Class variable to generate unique scope IDs

    def __init__(self, parent: Optional['Scope'] = None, name_prefix: str = "scope"):
        self.parent = parent
        self.symbols: Dict[str, SymbolInfo] = {}
        self.children: List['Scope'] = []
        Scope._scope_counter += 1
        self.id = Scope._scope_counter
        self.name = f"{name_prefix}_{self.id}"
        if parent:
            parent.children.append(self)
    
    def define(self, name: str, symbol_info: SymbolInfo) -> None:
        """Define a symbol in this scope."""
        if name in self.symbols:
            raise NeuroTypeError(
                f"Symbol '{name}' already defined in this scope",
                symbol_info.location
            )
        self.symbols[name] = symbol_info
    
    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """Look up a symbol in this scope or parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def lookup_local(self, name: str) -> Optional[SymbolInfo]:
        """Look up a symbol only in this scope."""
        return self.symbols.get(name)


class TypeChecker:
    """
    Main type checker and inference engine for NEURO.
    
    Implements Hindley-Milner style type inference with extensions for:
    - Tensor types with shape checking
    - Generic functions and structs
    - Neural network model validation
    """
    
    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose
        self.debug = debug # Store debug flag
        
        # Symbol table management
        Scope._scope_counter = 0 # Reset counter for fresh run
        self.global_scope = Scope(name_prefix="global")
        self.current_scope = self.global_scope
        
        # Type inference state
        self.type_var_counter = 0
        self.type_substitutions: Dict[TypeVariable, Type] = {}
        self.constraints: List[TypeConstraint] = []
        self.current_function_return_type: Optional[Type] = None
        self.current_generic_context: Dict[str, TypeVariable] = {} # For generic parameter resolution
        
        # Built-in types and functions
        self._init_builtin_types()
        self._init_builtin_functions()
        self._init_builtin_structs() # Initialize built-in structs
        
        # Type cache for expressions
        self.type_cache: Dict[ASTNode, Type] = {}
    
    def check_program(self, program: Program) -> Dict[ASTNode, Type]:
        """
        Perform type checking on a complete program.
        
        Returns:
            Dictionary mapping AST nodes to their inferred types
            
        Raises:
            NeuroTypeError: If type checking fails
        """
        if self.verbose: # General verbose message
            print("Starting type checking...")
        
        try:
            # Phase 1: Collect all top-level declarations
            self._collect_declarations(program)
            
            # Phase 2: Type check all statements
            for stmt in program.statements:
                self._check_statement(stmt)
            
            # Phase 3: Solve type constraints
            self._solve_constraints()
            
            # Phase 4: Apply substitutions and finalize types
            self._apply_substitutions()
            
            if self.verbose: # General verbose message
                print(f"✓ Type checking completed successfully")
            if self.debug: # Debug-specific message
                print(f"  - {len(self.type_cache)} types inferred")
                print(f"  - {len(self.global_scope.symbols)} global symbols")
            
            return self.type_cache.copy()
            
        except Exception as e:
            if isinstance(e, NeuroTypeError):
                raise
            raise NeuroTypeError(f"Type checking failed: {e}")
    
    # ========================================================================
    # Phase 1: Declaration Collection
    # ========================================================================
    
    def _collect_declarations(self, program: Program) -> None:
        """Collect all function, struct, and global variable declarations first."""
        for stmt in program.statements:
            if isinstance(stmt, FunctionDeclaration):
                self._collect_function_declaration(stmt)
            elif isinstance(stmt, StructDeclaration):
                self._collect_struct_declaration(stmt)
            elif isinstance(stmt, VariableDeclaration):
                # Assuming top-level VariableDeclarations are global.
                # self.current_scope is global_scope here.
                self._collect_global_variable_stub(stmt)
    
    def _collect_global_variable_stub(self, var_decl: VariableDeclaration) -> None:
        """Create a stub for a global variable in the symbol table."""
        # Global variables don't have their own generic params in the same way functions do.
        # Their type will be fully resolved in the _check_variable_declaration phase.
        # For now, register with a fresh type variable.
        tv = self._new_type_variable(location=var_decl.location)
        symbol_info = SymbolInfo(
            name=var_decl.name,
            symbol_type=tv,
            is_mutable=var_decl.is_mutable,
            is_function=False, # Explicitly false for variables
            location=var_decl.location,
            # generic_params and generic_param_map are None by default
        )
        if self.debug: # Log stub creation
            print(f"DEBUG: Collecting global variable stub for '{var_decl.name}' in scope '{self.global_scope.name}' (ID: {self.global_scope.id}).")
        self.global_scope.define(var_decl.name, symbol_info)

    def _collect_function_declaration(self, func: FunctionDeclaration) -> None:
        """Collect a function declaration into the symbol table."""
        
        original_generic_context = self.current_generic_context
        # Create a fresh generic context for this function's type variables
        current_function_generic_map: Dict[str, TypeVariable] = {}

        # Populate context for this function's generic parameters
        # and set it as the current generic context for resolving types within this function signature
        if func.generic_params:
            for gp_node in func.generic_params:
                if hasattr(gp_node, 'name'):
                    tv_location = gp_node.location if hasattr(gp_node, 'location') else func.location
                    current_function_generic_map[gp_node.name] = self._new_type_variable(location=tv_location)
        
        self.current_generic_context = current_function_generic_map

        try:
            param_types = []
            for param in func.parameters:
                # Resolve param_type using the current_function_generic_map
                param_type = self._resolve_type_annotation(param.type_annotation, defer_on_lookup_failure=True)
                param_types.append(param_type)
            
            return_type_location = func.return_type.location if func.return_type and hasattr(func.return_type, 'location') else func.location
            if func.return_type:
                return_type = self._resolve_type_annotation(func.return_type, defer_on_lookup_failure=True)
            else:
                return_type = self._new_type_variable(location=return_type_location)
            
            func_type = FunctionType(param_types, return_type)
            
            symbol_info = SymbolInfo(
                name=func.name,
                symbol_type=func_type,
                is_function=True,
                location=func.location,
                generic_params=func.generic_params, 
                generic_param_map=current_function_generic_map
            )
            
            if self.debug: # DIAGNOSTIC: Log collection attempt, now tied to debug flag
                print(f"DEBUG: Collecting function declaration for '{func.name}' in scope '{self.global_scope.name}' (ID: {self.global_scope.id}). Attempting to define/overwrite.") 
            # Handle potential overwrite of a variable stub
            existing_symbol = self.global_scope.lookup_local(func.name)
            if existing_symbol and \
               not existing_symbol.is_function and \
               existing_symbol.generic_params is None and \
               isinstance(existing_symbol.symbol_type, TypeVariable):
                # This is likely a variable stub, overwrite it with the function definition
                self.global_scope.symbols[func.name] = symbol_info
                if self.debug: # DIAGNOSTIC, now tied to debug flag
                    print(f"DEBUG: Function '{func.name}' (re)defined, overwriting stub in scope '{self.global_scope.name}' (ID: {self.global_scope.id}).") 
            else:
                # Not a stub or not defined yet, define normally (will error if it's a non-stub conflict)
                try:
                    self.global_scope.define(func.name, symbol_info)
                    if self.debug: # DIAGNOSTIC, now tied to debug flag
                         print(f"DEBUG: Function '{func.name}' defined successfully in scope '{self.global_scope.name}' (ID: {self.global_scope.id}).") 
                except NeuroTypeError as e:
                    # This means there was a non-stub conflict (e.g. func already defined as func)
                    if self.debug: # DIAGNOSTIC, now tied to debug flag
                        print(f"ERROR: Failed to define function '{func.name}' in scope '{self.global_scope.name}' (ID: {self.global_scope.id}) due to conflict: {e}") 
                    raise # Re-raise the legitimate conflict error

            self.type_cache[func] = func_type
            
        finally:
            self.current_generic_context = original_generic_context # Restore context
    
    def _collect_struct_declaration(self, struct: StructDeclaration) -> None:
        """Collect a struct declaration into the symbol table."""
        
        original_generic_context = self.current_generic_context
        current_struct_generic_map: Dict[str, TypeVariable] = {}

        if struct.generic_params:
            for gp_node in struct.generic_params:
                 if hasattr(gp_node, 'name'):
                    tv_location = gp_node.location if hasattr(gp_node, 'location') else struct.location
                    current_struct_generic_map[gp_node.name] = self._new_type_variable(location=tv_location)

        self.current_generic_context = current_struct_generic_map

        try:
            struct_type_repr = PrimitiveType(struct.name) 
            
            field_types_resolved = {}
            if not hasattr(self, 'struct_fields'): 
                 self.struct_fields = {}

            for field_def in struct.fields:
                field_types_resolved[field_def.name] = self._resolve_type_annotation(field_def.type_annotation, defer_on_lookup_failure=True)
            self.struct_fields[struct.name] = field_types_resolved

            symbol_info = SymbolInfo(
                name=struct.name,
                symbol_type=struct_type_repr, 
                is_function=False, 
                location=struct.location,
                generic_params=struct.generic_params,
                generic_param_map=current_struct_generic_map # Store the map for this struct
            )
            if self.debug: # Log struct collection
                print(f"DEBUG: Collecting struct declaration for '{struct.name}' in scope '{self.global_scope.name}' (ID: {self.global_scope.id}). Attempting to define/overwrite.")
            
            # Handle potential overwrite of a variable stub
            existing_symbol = self.global_scope.lookup_local(struct.name)
            if existing_symbol and \
               not existing_symbol.is_function and \
               existing_symbol.generic_params is None and \
               isinstance(existing_symbol.symbol_type, TypeVariable):
                # This is likely a variable stub, overwrite it with the struct definition
                if self.debug:
                    print(f"DEBUG: Struct '{struct.name}' (re)defined, overwriting stub in scope '{self.global_scope.name}' (ID: {self.global_scope.id}).")
                self.global_scope.symbols[struct.name] = symbol_info
            else:
                # Not a stub or not defined yet, define normally (will error if it's a non-stub conflict)
                try:
                    self.global_scope.define(struct.name, symbol_info)
                    if self.debug:
                        print(f"DEBUG: Struct '{struct.name}' defined successfully in scope '{self.global_scope.name}' (ID: {self.global_scope.id}).")
                except NeuroTypeError as e:
                    if self.debug:
                        print(f"ERROR: Failed to define struct '{struct.name}' in scope '{self.global_scope.name}' (ID: {self.global_scope.id}) due to conflict: {e}")
                    raise
            
            self.type_cache[struct] = struct_type_repr 
            
        finally:
            self.current_generic_context = original_generic_context
    
    # ========================================================================
    # Phase 2: Statement Type Checking
    # ========================================================================
    
    def _check_statement(self, stmt: Statement) -> None:
        """Type check a statement."""
        if isinstance(stmt, VariableDeclaration):
            self._check_variable_declaration(stmt)
        elif isinstance(stmt, FunctionDeclaration):
            self._check_function_declaration(stmt)
        elif isinstance(stmt, StructDeclaration):
            # Struct declarations are already processed in declaration collection phase
            # No additional type checking needed for the struct itself
            pass
        elif isinstance(stmt, ExpressionStatement):
            self._check_expression(stmt.expression)
        elif isinstance(stmt, IfStatement):
            self._check_if_statement(stmt)
        elif isinstance(stmt, WhileStatement):
            self._check_while_statement(stmt)
        elif isinstance(stmt, ForStatement):
            self._check_for_statement(stmt)
        elif isinstance(stmt, ReturnStatement):
            self._check_return_statement(stmt)
        elif isinstance(stmt, Block):
            self._check_block(stmt)
        # Add other statement types as needed
    
    def _check_variable_declaration(self, var_decl: VariableDeclaration) -> None:
        """Type check a variable declaration."""
        # Infer type from initializer if present
        if self.debug:
            print(f"DEBUG: Checking variable declaration for '{var_decl.name}' in scope '{self.current_scope.name}' (ID: {self.current_scope.id}). Type annotation: {var_decl.type_annotation}, Initializer: {var_decl.initializer is not None}")

        if var_decl.initializer:
            init_type = self._check_expression(var_decl.initializer)
        else:
            init_type = None
        
        # Get declared type if present
        declared_type = None
        if var_decl.type_annotation:
            # When checking a variable declaration (not collecting), generic context should be active
            # if inside a generic function. _resolve_type_annotation uses self.current_generic_context.
            declared_type = self._resolve_type_annotation(var_decl.type_annotation)
        
        # Determine final type
        final_type: Type
        if declared_type and init_type:
            # Both declared and inferred - must be compatible
            self._add_constraint(declared_type, init_type, var_decl.location)
            final_type = declared_type # Or apply_substitution(declared_type) after solving
        elif declared_type:
            # Only declared type
            final_type = declared_type
        elif init_type:
            # Only inferred type
            final_type = init_type
        else:
            # Neither - error
            raise NeuroTypeError(
                f"Variable '{var_decl.name}' must have either a type annotation or initializer",
                var_decl.location
            )
        
        # Add/Update symbol in symbol table
        if self.current_scope is self.global_scope:
            # Global variable: it was stubbed by _collect_global_variable_stub.
            # Update its type. is_mutable and location were set during stubbing.
            existing_symbol_info = self.global_scope.symbols.get(var_decl.name)
            if existing_symbol_info:
                # Ensure it's not a function/struct masquerading with same name
                if existing_symbol_info.is_function:
                     raise NeuroTypeError(f"Cannot redefine function '{var_decl.name}' as a variable.", var_decl.location)
                
                existing_symbol_info.symbol_type = final_type
                existing_symbol_info.is_mutable = var_decl.is_mutable # Update mutability if needed
            else:
                # This case implies a bug in collection logic if stubs are expected.
                # Define it fresh, but this is a fallback.
                new_symbol_info = SymbolInfo(
                    name=var_decl.name,
                    symbol_type=final_type,
                    is_mutable=var_decl.is_mutable,
                    location=var_decl.location
                )
                self.global_scope.define(var_decl.name, new_symbol_info)
                if self.debug:
                    print(f"DEBUG: Defined new global variable '{var_decl.name}' with type {final_type} in scope '{self.global_scope.name}' (ID: {self.global_scope.id}). Fallback path.")
        else:
            # Local variable: define it in the current (local) scope.
            symbol_info = SymbolInfo(
                name=var_decl.name,
                symbol_type=final_type,
                is_mutable=var_decl.is_mutable,
                location=var_decl.location
            )
            self.current_scope.define(var_decl.name, symbol_info)
            if self.debug:
                print(f"DEBUG: Defined local variable '{var_decl.name}' with type {final_type} in scope '{self.current_scope.name}' (ID: {self.current_scope.id}).")
        
        self.type_cache[var_decl] = final_type
    
    def _check_function_declaration(self, func: FunctionDeclaration) -> None:
        """Type check a function declaration body."""
        func_symbol_info = self.global_scope.lookup(func.name)
        if not func_symbol_info:
            raise NeuroTypeError(f"Function '{func.name}' not found in global scope.", func.location)

        old_function_return_type = self.current_function_return_type
        old_generic_context = self.current_generic_context

        # Set the specific generic context for this function from its SymbolInfo
        # This ensures TypeVariables for generic parameters are correctly reused.
        self.current_generic_context = func_symbol_info.generic_param_map or {}
        
        # Determine expected return type
        if isinstance(func_symbol_info.symbol_type, FunctionType):
            self.current_function_return_type = func_symbol_info.symbol_type.return_type
        # Fallback for safety, though func_symbol_info.symbol_type should be FunctionType
        elif func.return_type:
             # Re-resolve with current_generic_context (now set for this function)
             self.current_function_return_type = self._resolve_type_annotation(func.return_type, defer_on_lookup_failure=True)
        else:
            return_type_loc = func.location # Default location for inferred return type TV
            self.current_function_return_type = self._new_type_variable(location=return_type_loc)

        # Create new scope for function body
        func_scope_name = f"function_{func.name}"
        func_scope = Scope(self.current_scope, name_prefix=func_scope_name)
        old_scope = self.current_scope
        self.current_scope = func_scope
        
        if self.debug:
            print(f"DEBUG: Entered scope '{self.current_scope.name}' (ID: {self.current_scope.id}) for function '{func.name}'. Parent: '{old_scope.name}' (ID: {old_scope.id}).")

        try:
            # Add parameters to the function's scope
            # Their types are taken from the FunctionType stored in SymbolInfo, which were resolved with the correct generic context
            if isinstance(func_symbol_info.symbol_type, FunctionType):
                if len(func.parameters) == len(func_symbol_info.symbol_type.param_types):
                    for param_ast_node, param_type_from_func_sig in zip(func.parameters, func_symbol_info.symbol_type.param_types):
                        param_info = SymbolInfo(
                            name=param_ast_node.name,
                            symbol_type=param_type_from_func_sig, # This type is from the collected signature
                            is_mutable=False, 
                            location=param_ast_node.location # Use location from AST Parameter node
                        )
                        self.current_scope.define(param_ast_node.name, param_info)
                else:
                     # Arity mismatch, should have been caught earlier or is an internal error
                    pass # Or raise internal error
            else:
                # Fallback: if symbol_type is not FunctionType (shouldn't happen)
                # This path indicates a potential issue in earlier phases.
                for param_ast_node in func.parameters:
                    param_type = self._resolve_type_annotation(param_ast_node.type_annotation, defer_on_lookup_failure=True)
                    symbol_info = SymbolInfo(
                        name=param_ast_node.name, symbol_type=param_type, 
                        is_mutable=False, location=param_ast_node.location
                    )
                    self.current_scope.define(param_ast_node.name, symbol_info)

            if isinstance(func.body, Block):
                self._check_function_body_block(func.body)
            else:
                self._check_statement(func.body)
            
        finally:
            self.current_scope = old_scope
            if self.debug:
                print(f"DEBUG: Exited to scope '{self.current_scope.name}' (ID: {self.current_scope.id}) from function '{func.name}'.")
            self.current_function_return_type = old_function_return_type
            self.current_generic_context = old_generic_context # Restore old generic context
    
    def _check_if_statement(self, if_stmt: IfStatement) -> None:
        """Type check an if statement."""
        # Condition must be boolean
        condition_type = self._check_expression(if_stmt.condition)
        bool_type = PrimitiveType("bool")
        self._add_constraint(condition_type, bool_type, if_stmt.condition.location)
        
        # Check branches
        self._check_statement(if_stmt.then_branch)
        if if_stmt.else_branch:
            self._check_statement(if_stmt.else_branch)
    
    def _check_while_statement(self, while_stmt: WhileStatement) -> None:
        """Type check a while statement."""
        # Condition must be boolean
        condition_type = self._check_expression(while_stmt.condition)
        bool_type = PrimitiveType("bool")
        self._add_constraint(condition_type, bool_type, while_stmt.condition.location)
        
        # Check body
        self._check_statement(while_stmt.body)
    
    def _check_for_statement(self, for_stmt: ForStatement) -> None:
        """Type check a for statement."""
        # Check iterable first to get its type
        iterable_type = self._check_expression(for_stmt.iterable)
        
        # Create new scope for loop variable
        loop_scope_name = f"for_loop_var_{for_stmt.variable}"
        loop_scope = Scope(self.current_scope, name_prefix=loop_scope_name)
        old_scope = self.current_scope
        self.current_scope = loop_scope

        if self.debug:
            print(f"DEBUG: Entered scope '{self.current_scope.name}' (ID: {self.current_scope.id}) for 'for' loop variable '{for_stmt.variable}'. Parent: '{old_scope.name}' (ID: {old_scope.id}).")
        
        try:
            # Infer loop variable type from iterable
            loop_var_type = self._infer_iterator_element_type(iterable_type)
            
            # Add loop variable to scope
            symbol_info = SymbolInfo(
                name=for_stmt.variable,
                symbol_type=loop_var_type,
                location=for_stmt.location
            )
            self.current_scope.define(for_stmt.variable, symbol_info)
            
            # Check body
            self._check_statement(for_stmt.body)
            
        finally:
            self.current_scope = old_scope
            if self.debug:
                print(f"DEBUG: Exited to scope '{self.current_scope.name}' (ID: {self.current_scope.id}) from 'for' loop variable '{for_stmt.variable}'.")
    
    def _infer_iterator_element_type(self, iterable_type: Type) -> Type:
        """Infer the element type from an iterable type."""
        iterable_type = self._apply_substitution(iterable_type)
        
        if isinstance(iterable_type, TensorType):
            return iterable_type.element_type
        elif isinstance(iterable_type, TypeVariable):
            # For type variables, create a new type variable for element type
            # This might need to be constrained further based on usage.
            element_tv = self._new_type_variable()
            # Example: iterable_type could be constrained to be Tensor<element_tv>
            return element_tv
        else:
            # For unknown types, assume int (for range-like constructs)
            # A more robust solution would be to require an __iter__ protocol or similar.
            return PrimitiveType("int")
    
    def _check_return_statement(self, ret_stmt: ReturnStatement) -> None:
        """Type check a return statement."""
        actual_return_type: Type
        if ret_stmt.value:
            actual_return_type = self._check_expression(ret_stmt.value)
        else:
            # Void return type; needs a representation if functions can be explicitly void.
            # For now, let's use a placeholder. If language has 'void' type, use that.
            # Assuming PrimitiveType("void") or similar might be defined if needed.
            # If not, this implies a function that implicitly returns, e.g. no return type annotation and no return statements with value.
            # This part might need alignment with how void functions are declared/inferred.
            # For now, if no value, we can assign a special void marker or infer based on context.
            # Let's assume a function returning nothing should have its return type constrained by this.
            # A common approach for void is a special unit type.
            # If Neuro has explicit void, use that. Otherwise, this constraint might be problematic if current_function_return_type expects a value.
            # For now, let's assume void is implicitly handled.
            # If current_function_return_type is set, an empty return must match it (e.g. if it's void or a TypeVar that unifies with void)
            actual_return_type = PrimitiveType("void") # Placeholder for void/unit type

        if self.current_function_return_type:
            self._add_constraint(actual_return_type, self.current_function_return_type, ret_stmt.location)
        
        self.type_cache[ret_stmt] = actual_return_type
    
    def _check_block(self, block: Block) -> None:
        """Type check a block statement."""
        # Create new scope for block
        block_scope_name = f"block"
        block_scope = Scope(self.current_scope, name_prefix=block_scope_name)
        old_scope = self.current_scope
        self.current_scope = block_scope

        if self.debug:
            print(f"DEBUG: Entered scope '{self.current_scope.name}' (ID: {self.current_scope.id}) for block. Parent: '{old_scope.name}' (ID: {old_scope.id}).")
        
        try:
            for i, stmt in enumerate(block.statements):
                try:
                    self._check_statement(stmt)
                except NeuroTypeError as e:
                    if self.debug:
                        print(f"DEBUG: NeuroTypeError caught while checking statement {i} ({type(stmt)}) in block '{self.current_scope.name}': {e.message} at {e.location}")
                    raise # Re-raise to be caught by higher-level handlers or terminate block checking
                except Exception as e_generic:
                    if self.debug:
                        stmt_repr = str(stmt)[:100] # Basic representation of the statement
                        print(f"DEBUG: Generic Exception caught while checking statement {i} ({type(stmt)}: {stmt_repr}) in block '{self.current_scope.name}': {e_generic}")
                    location = stmt.location if hasattr(stmt, 'location') else None
                    raise NeuroTypeError(f"Internal error checking statement in block: {e_generic}", location=location) from e_generic
        finally:
            self.current_scope = old_scope
            if self.debug:
                print(f"DEBUG: Exited to scope '{self.current_scope.name}' (ID: {self.current_scope.id}) from block.")
    
    def _check_function_body_block(self, block: Block) -> None:
        """Type check a function body block without creating a new scope."""
        # Process statements directly in current scope (function scope)
        if self.debug:
            print(f"DEBUG: Checking function body block for function within scope '{self.current_scope.name}' (ID: {self.current_scope.id}). Statements count: {len(block.statements)}")
        for i, stmt in enumerate(block.statements):
            if self.debug:
                stmt_type_name = type(stmt).__name__
                stmt_details = stmt.name if hasattr(stmt, 'name') else stmt_type_name
                print(f"DEBUG: Checking statement {i+1}/{len(block.statements)} in function body '{self.current_scope.name}': {stmt_type_name} ({stmt_details}) at {stmt.location}")
            try:
                self._check_statement(stmt)
            except NeuroTypeError as e:
                if self.debug:
                    print(f"DEBUG: NeuroTypeError caught while checking statement {i+1} ({type(stmt)}) in function body '{self.current_scope.name}': {e.message} at {e.location}")
                raise # Re-raise to be caught by _check_function_declaration's try-finally
            except Exception as e_generic:
                if self.debug:
                    stmt_repr = str(stmt)[:100] # Basic representation of the statement
                    print(f"DEBUG: Generic Exception caught while checking statement {i+1} ({type(stmt)}: {stmt_repr}) in function body '{self.current_scope.name}': {e_generic}")
                location = stmt.location if hasattr(stmt, 'location') else None
                raise NeuroTypeError(f"Internal error checking statement in function body: {e_generic}", location=location) from e_generic
    
    # ========================================================================
    # Expression Type Checking
    # ========================================================================
    
    def _check_expression(self, expr: Expression) -> Type:
        """Type check an expression and return its type."""
        if expr in self.type_cache:
            return self.type_cache[expr]
        
        if isinstance(expr, Literal):
            expr_type = self._check_literal(expr)
        elif isinstance(expr, Identifier):
            expr_type = self._check_identifier(expr)
        elif isinstance(expr, BinaryOp):
            expr_type = self._check_binary_op(expr)
        elif isinstance(expr, UnaryOp):
            expr_type = self._check_unary_op(expr)
        elif isinstance(expr, FunctionCall):
            expr_type = self._check_function_call(expr)
        elif isinstance(expr, TensorLiteral):
            expr_type = self._check_tensor_literal(expr)
        elif isinstance(expr, Assignment):
            expr_type = self._check_assignment(expr)
        elif isinstance(expr, MemberAccess):
            expr_type = self._check_member_access(expr)
        elif isinstance(expr, IndexAccess):
            expr_type = self._check_index_access(expr)
        elif isinstance(expr, StructInitializer):
            expr_type = self._check_struct_initializer(expr)
        elif isinstance(expr, RangeLiteral):
            expr_type = self._check_range_literal(expr)
        elif isinstance(expr, NamedArgument):
            expr_type = self._check_named_argument(expr)
        elif isinstance(expr, ModelDefinition):
            if self.debug:
                print(f"DEBUG: Checking ModelDefinition: {expr} in scope {self.current_scope.name}")

            # Create a new scope for the model initializer to contain its symbols
            model_scope_name = f"model_initializer"
            model_scope = Scope(self.current_scope, name_prefix=model_scope_name)
            old_scope = self.current_scope
            self.current_scope = model_scope

            if self.debug:
                print(f"DEBUG: Entered scope '{self.current_scope.name}' (ID: {self.current_scope.id}) for model definition. Parent: '{old_scope.name}' (ID: {old_scope.id}).")

            try:
                # Type check each layer expression in the new scope.
                for layer_expr in expr.layers:
                    self._check_expression(layer_expr)
            finally:
                # Restore the original scope
                self.current_scope = old_scope
                if self.debug:
                    print(f"DEBUG: Exited to scope '{self.current_scope.name}' (ID: {self.current_scope.id}) from model definition.")

            # This expression resolves to a NeuralNetwork type.
            # A more advanced implementation could create a specific type for each model.
            expr_type = PrimitiveType("NeuralNetwork", location=expr.location)
        else:
            # Default case - create type variable
            if self.debug:
                print(f"DEBUG: Expression {type(expr)} at {expr.location} falling to default new_type_variable in _check_expression. Scope: {self.current_scope.name}")
            expr_type = self._new_type_variable(location=expr.location)
        
        self.type_cache[expr] = expr_type
        return expr_type
    
    def _check_literal(self, literal: Literal) -> Type:
        """Type check a literal value."""
        if isinstance(literal.value, bool):
            return PrimitiveType("bool")
        elif isinstance(literal.value, int):
            return PrimitiveType("int")
        elif isinstance(literal.value, float):
            return PrimitiveType("float")
        elif isinstance(literal.value, str):
            return PrimitiveType("string")
        else:
            raise NeuroTypeError(f"Unknown literal type: {type(literal.value)}", literal.location)
    
    def _check_identifier(self, identifier: Identifier) -> Type:
        """Type check an identifier."""
        if self.debug: # More detailed logging for identifier lookup
            print(f"DEBUG: Checking identifier '{identifier.name}' at {identifier.location} in scope '{self.current_scope.name}' (ID: {self.current_scope.id}).")
            print(f"DEBUG: Current scope '{self.current_scope.name}' symbols: {list(self.current_scope.symbols.keys())}")

        symbol = self.current_scope.lookup(identifier.name)
        if not symbol:
            if self.debug: # DIAGNOSTIC output block, now tied to debug flag
                print(f"--- DIAGNOSTIC: Undefined variable '{identifier.name}' at {identifier.location} ---")
                scope_chain = []
                curr = self.current_scope
                idx = 0
                while curr:
                    scope_info = f"  Scope {idx}: ({type(curr).__name__}) Symbols: {list(curr.symbols.keys())}"
                    if curr is self.global_scope:
                        scope_info += " (Global Scope)"
                    scope_chain.append(scope_info)
                    curr = curr.parent
                    idx += 1
                print("Scope chain at point of failure:")
                for s_info in scope_chain:
                    print(s_info)
                print(f"Global scope ('self.global_scope') symbols: {list(self.global_scope.symbols.keys())}")
                print(f"Current generic context: {self.current_generic_context}")
                print(f"--- END DIAGNOSTIC for '{identifier.name}' ---")
            raise NeuroTypeError(f"Undefined variable: {identifier.name}", identifier.location)
        return symbol.symbol_type
    
    def _check_binary_op(self, binary_op: BinaryOp) -> Type:
        """Type check a binary operation."""
        left_type = self._check_expression(binary_op.left)
        right_type = self._check_expression(binary_op.right)
        
        # Arithmetic operations
        if binary_op.operator in ['+', '-', '*', '/', '%']:
            return self._handle_numeric_operation(left_type, right_type, binary_op.operator, binary_op.location)
        
        # Comparison operations
        elif binary_op.operator in ['==', '!=', '<', '>', '<=', '>=']:
            self._add_constraint(left_type, right_type, binary_op.location)
            return PrimitiveType("bool")
        
        # Logical operations
        elif binary_op.operator in ['&&', '||']:
            bool_type = PrimitiveType("bool")
            self._add_constraint(left_type, bool_type, binary_op.left.location)
            self._add_constraint(right_type, bool_type, binary_op.right.location)
            return bool_type
        
        # Matrix operations
        elif binary_op.operator == '@':
            # Matrix multiplication - for now, return left type
            return left_type
        
        else:
            raise NeuroTypeError(f"Unknown binary operator: {binary_op.operator}", binary_op.location)
    
    def _handle_numeric_operation(self, left_type: Type, right_type: Type, operator: str, location: Optional[SourceLocation]) -> Type:
        """Handle numeric operations with type coercion."""
        # Apply substitutions first
        left_type = self._apply_substitution(left_type)
        right_type = self._apply_substitution(right_type)
        
        # String concatenation with + (both operands must be strings)
        if operator == '+':
            if (isinstance(left_type, PrimitiveType) and left_type.name == "string") and \
               (isinstance(right_type, PrimitiveType) and right_type.name == "string"):
                return PrimitiveType("string")
        
        # Check if both types are numeric
        numeric_types = {"int", "float"}
        
        left_is_numeric = isinstance(left_type, PrimitiveType) and left_type.name in numeric_types
        right_is_numeric = isinstance(right_type, PrimitiveType) and right_type.name in numeric_types
        
        if left_is_numeric and right_is_numeric:
            # Numeric coercion: int + float = float, float + int = float
            if left_type.name == "float" or right_type.name == "float":
                return PrimitiveType("float")
            else:
                return PrimitiveType("int")
        
        # Handle type variables - try to unify
        if isinstance(left_type, TypeVariable) or isinstance(right_type, TypeVariable):
            self._add_constraint(left_type, right_type, location)
            # Return the non-variable type if one exists, otherwise left
            if isinstance(left_type, TypeVariable) and not isinstance(right_type, TypeVariable):
                return right_type
            return left_type
        
        # Default case: types must match exactly
        self._add_constraint(left_type, right_type, location)
        return left_type
    
    def _check_unary_op(self, unary_op: UnaryOp) -> Type:
        """Type check a unary operation."""
        operand_type = self._check_expression(unary_op.operand)
        
        if unary_op.operator == '-':
            # Negation - operand must be numeric
            return operand_type
        elif unary_op.operator == '!':
            # Logical not - operand must be boolean
            bool_type = PrimitiveType("bool")
            self._add_constraint(operand_type, bool_type, unary_op.operand.location)
            return bool_type
        else:
            raise NeuroTypeError(f"Unknown unary operator: {unary_op.operator}", unary_op.location)
    
    def _check_function_call(self, call: FunctionCall) -> Type:
        """Type check a function call."""
        func_type = self._check_expression(call.function)
        
        # Handle struct constructors (when struct name is used as function)
        if isinstance(call.function, Identifier) and hasattr(self, 'struct_fields'):
            struct_name = call.function.name
            if struct_name in self.struct_fields:
                # This is a struct constructor call - return the struct type
                return PrimitiveType(struct_name)
        
        if not isinstance(func_type, FunctionType):
            raise NeuroTypeError(
                f"Cannot call non-function type: {func_type}",
                call.function.location
            )
        
        # Check argument count
        if len(call.arguments) != len(func_type.param_types):
            raise NeuroTypeError(
                f"Function expects {len(func_type.param_types)} arguments, got {len(call.arguments)}",
                call.location
            )
        
        # Handle polymorphic functions (like str) by creating fresh type variables
        instantiated_func_type = self._instantiate_polymorphic_function(func_type, call.function)
        
        # Check argument types
        for i, (arg, param_type) in enumerate(zip(call.arguments, instantiated_func_type.param_types)):
            arg_type = self._check_expression(arg)
            self._add_constraint(arg_type, param_type, arg.location)
        
        return instantiated_func_type.return_type
    
    def _instantiate_polymorphic_function(self, func_type: FunctionType, func_expr: Expression) -> FunctionType:
        """Create a fresh instance of a polymorphic function type."""
        # Check if this is a polymorphic function (contains type variables)
        type_vars_in_func = self._collect_type_variables(func_type)
        
        if not type_vars_in_func:
            # Not polymorphic, return as-is
            return func_type
        
        # Create fresh type variables for each type variable in the function
        substitution_map = {}
        for type_var in type_vars_in_func:
            substitution_map[type_var] = self._new_type_variable()
        
        # Apply substitutions to create fresh instance
        new_param_types = []
        for param_type in func_type.param_types:
            new_param_types.append(self._substitute_type_variables(param_type, substitution_map))
        
        new_return_type = self._substitute_type_variables(func_type.return_type, substitution_map)
        
        return FunctionType(new_param_types, new_return_type)
    
    def _collect_type_variables(self, type_expr: Type) -> Set[TypeVariable]:
        """Collect all type variables in a type expression."""
        type_vars = set()
        
        if isinstance(type_expr, TypeVariable):
            type_vars.add(type_expr)
        elif isinstance(type_expr, TensorType):
            type_vars.update(self._collect_type_variables(type_expr.element_type))
        elif isinstance(type_expr, FunctionType):
            for param_type in type_expr.param_types:
                type_vars.update(self._collect_type_variables(param_type))
            type_vars.update(self._collect_type_variables(type_expr.return_type))
        
        return type_vars
    
    def _substitute_type_variables(self, type_expr: Type, substitution_map: Dict[TypeVariable, TypeVariable]) -> Type:
        """Substitute type variables in a type expression."""
        if isinstance(type_expr, TypeVariable):
            return substitution_map.get(type_expr, type_expr)
        elif isinstance(type_expr, TensorType):
            new_element_type = self._substitute_type_variables(type_expr.element_type, substitution_map)
            return TensorType(new_element_type, type_expr.shape)
        elif isinstance(type_expr, FunctionType):
            new_param_types = [
                self._substitute_type_variables(pt, substitution_map) 
                for pt in type_expr.param_types
            ]
            new_return_type = self._substitute_type_variables(type_expr.return_type, substitution_map)
            return FunctionType(new_param_types, new_return_type)
        else:
            return type_expr
    
    def _check_tensor_literal(self, tensor: TensorLiteral) -> Type:
        """Type check a tensor literal."""
        if not tensor.elements:
            # Empty tensor - create type variable for element type
            element_type = self._new_type_variable()
            return TensorType(element_type, shape=[0])
        
        # For nested tensor literals like [[1, 2], [3, 4]], we want to infer
        # the element type and full shape
        first_elem_type = self._check_expression(tensor.elements[0])
        
        if isinstance(first_elem_type, TensorType):
            # This is a tensor of tensors - build full shape
            element_type = first_elem_type.element_type
            
            # Ensure all elements are tensors of the same type and shape
            for elem in tensor.elements[1:]:
                elem_type = self._check_expression(elem)
                self._add_constraint(first_elem_type, elem_type, elem.location)
            
            # Build full shape: outer dimension + inner shape
            if first_elem_type.shape:
                shape = [len(tensor.elements)] + first_elem_type.shape
            else:
                shape = [len(tensor.elements)]
        else:
            # This is a tensor of primitives
            element_type = first_elem_type
            
            # Ensure all elements have the same type
            for elem in tensor.elements[1:]:
                elem_type = self._check_expression(elem)
                self._add_constraint(element_type, elem_type, elem.location)
            
            # Shape is just the outer dimension
            shape = [len(tensor.elements)]
        
        return TensorType(element_type, shape)
    
    def _check_assignment(self, assignment: Assignment) -> Type:
        """Type check an assignment expression."""
        target_type = self._check_expression(assignment.target)
        value_type = self._check_expression(assignment.value)
        
        # Assignment result type is the value type
        self._add_constraint(target_type, value_type, assignment.location)
        return value_type
    
    def _check_member_access(self, member_access: MemberAccess) -> Type:
        """Type check member access."""
        object_type = self._check_expression(member_access.object)
        
        # Handle struct member access
        if isinstance(object_type, PrimitiveType) and hasattr(self, 'struct_fields'):
            struct_name = object_type.name
            if struct_name in self.struct_fields:
                field_types = self.struct_fields[struct_name]
                if member_access.member in field_types:
                    member_type = field_types[member_access.member]
                    self.type_cache[member_access] = member_type
                    return member_type
                else:
                    raise NeuroTypeError(
                        f"Struct '{struct_name}' has no field '{member_access.member}'",
                        member_access.location
                    )
        
        # Handle built-in method access (forward, backward, etc.)
        if member_access.member in ['forward', 'backward', 'step']:
            # Return a function type for method calls
            method_type = FunctionType(
                [self._new_type_variable()],  # Accept arguments
                self._new_type_variable()  # Return type
            )
            self.type_cache[member_access] = method_type
            return method_type
        
        # Handle tensor methods
        if isinstance(object_type, TensorType) and member_access.member == 'reshape':
            # reshape method returns a function that takes dimensions and returns a tensor
            method_type = FunctionType(
                [PrimitiveType("int"), PrimitiveType("int")],  # dimensions
                TensorType(object_type.element_type, None)  # Return tensor with same element type
            )
            self.type_cache[member_access] = method_type
            return method_type
        
        # Default case - return type variable
        result_type = self._new_type_variable()
        self.type_cache[member_access] = result_type
        return result_type
    
    def _check_index_access(self, index_access: IndexAccess) -> Type:
        """Type check index access."""
        object_type = self._check_expression(index_access.object)
        index_type = self._check_expression(index_access.index)
        
        # Index must be integer
        int_type = PrimitiveType("int")
        self._add_constraint(index_type, int_type, index_access.index.location)
        
        # If object is tensor, return element type
        if isinstance(object_type, TensorType):
            return object_type.element_type
        
        # Otherwise, return type variable
        return self._new_type_variable()
    
    def _check_struct_initializer(self, struct_init: StructInitializer) -> Type:
        """Type check struct initialization."""
        # Check if the struct type exists
        struct_symbol = self.current_scope.lookup(struct_init.struct_type)
        if not struct_symbol:
            raise NeuroTypeError(f"Unknown struct type: {struct_init.struct_type}", struct_init.location)
        
        # Type check each field value (but not the field names themselves)
        for field_name, field_value in struct_init.fields.items():
            # Type check the field value expression
            field_type = self._check_expression(field_value)
            
            # In a full implementation, we'd validate the field type against the struct definition
            # For now, we just ensure the field values are properly typed
        
        # Return the struct type
        return PrimitiveType(struct_init.struct_type)
    
    def _check_range_literal(self, range_lit: RangeLiteral) -> Type:
        """Type check range literal."""
        start_type = self._check_expression(range_lit.start)
        end_type = self._check_expression(range_lit.end)
        
        # Both start and end should be integers
        int_type = PrimitiveType("int")
        self._add_constraint(start_type, int_type, range_lit.start.location)
        self._add_constraint(end_type, int_type, range_lit.end.location)
        
        # Range produces a tensor of integers
        return TensorType(int_type, None)  # Dynamic size
    
    def _check_named_argument(self, named_arg: NamedArgument) -> Type:
        """Type check a named argument."""
        # Type check the value expression
        value_type = self._check_expression(named_arg.value)
        # The named argument has the same type as its value
        return value_type
    
    # ========================================================================
    # Type Resolution and Utilities
    # ========================================================================
    
    def _resolve_type_annotation(self, type_annotation: Type, defer_on_lookup_failure: bool = False) -> Type:
        """Resolve a type annotation to a concrete type, using current generic context."""
        try:
            if isinstance(type_annotation, PrimitiveType):
                return type_annotation
            elif isinstance(type_annotation, TensorType):
                element_type = self._resolve_type_annotation(type_annotation.element_type, defer_on_lookup_failure)
                return TensorType(element_type, type_annotation.shape, location=type_annotation.location)
            elif isinstance(type_annotation, FunctionType):
                param_types = [self._resolve_type_annotation(pt, defer_on_lookup_failure) for pt in type_annotation.param_types]
                return_type = self._resolve_type_annotation(type_annotation.return_type, defer_on_lookup_failure)
                return FunctionType(param_types, return_type)
            elif isinstance(type_annotation, GenericType):
                # Case 1: Is it a formal generic parameter (e.g., T in func foo<T>(param: T))?
                if hasattr(type_annotation, 'name') and type_annotation.name in self.current_generic_context:
                    return self.current_generic_context[type_annotation.name]
                
                # Case 2: Is it a known concrete type name (e.g., a struct name like Vector3D)?
                if hasattr(type_annotation, 'name'):
                    loc = type_annotation.location if hasattr(type_annotation, 'location') else None
                    symbol_info = self.current_scope.lookup(type_annotation.name)
                    if symbol_info:
                        if isinstance(symbol_info.symbol_type, Type):
                            # TODO: If symbol_info.symbol_type is for a generic struct (e.g. List<T>),
                            # and type_annotation is used like List<int>, we'd need to handle instantiation.
                            # For now, returning the base type (e.g. PrimitiveType("List") for struct List).
                            return symbol_info.symbol_type
                        else:
                            # Found a symbol, but it's not a type (e.g., a variable used in a type context).
                            raise NeuroTypeError(f"Identifier '{type_annotation.name}' refers to a value, not a type.", loc)
                    else:
                        # Symbol not found, implies undefined type.
                        raise NeuroTypeError(f"Undefined type: '{type_annotation.name}'", loc)
            
            elif isinstance(type_annotation, TypeVariable):
                 return type_annotation # It's already a TypeVariable
            
            # If type_annotation is not any of the above known Type subclasses, it's an error.
            loc = type_annotation.location if hasattr(type_annotation, 'location') else None
            raise NeuroTypeError(f"Unknown or unresolvable type annotation node: {type(type_annotation)}", loc)

        except NeuroTypeError as e:
            if defer_on_lookup_failure and (e.message.startswith("Undefined type:") or "refers to a value, not a type" in e.message):
                # If deferring and it's a lookup-related error, return a placeholder TypeVariable.
                return self._new_type_variable(location=getattr(type_annotation, 'location', None))
            else:
                raise # Re-raise other NeuroTypeErrors or if not deferring
    
    def _new_type_variable(self, location: Optional[SourceLocation] = None) -> TypeVariable:
        """Create a new type variable."""
        var = TypeVariable(f"T{self.type_var_counter}", self.type_var_counter, location=location)
        self.type_var_counter += 1
        return var
    
    def _add_constraint(self, left: Type, right: Type, location: Optional[SourceLocation] = None) -> None:
        """Add a type constraint for unification."""
        constraint = TypeConstraint(left, right, location)
        self.constraints.append(constraint)
    
    # ========================================================================
    # Phase 3: Constraint Solving
    # ========================================================================
    
    def _solve_constraints(self) -> None:
        """Solve all type constraints using unification."""
        if self.verbose: # General verbose message for constraint solving start
            print(f"Solving {len(self.constraints)} type constraints...")
        
        for constraint in self.constraints:
            self._unify(constraint.left, constraint.right, constraint.location)
    
    def _unify(self, left: Type, right: Type, location: Optional[SourceLocation] = None) -> None:
        """Unify two types."""
        # Apply current substitutions
        left = self._apply_substitution(left)
        right = self._apply_substitution(right)
        
        # Same type
        if left == right:
            return
        
        # Type variable on left
        if isinstance(left, TypeVariable):
            if self._occurs_check(left, right):
                raise NeuroTypeError(f"Infinite type: {left} occurs in {right}", location)
            self.type_substitutions[left] = right
            return
        
        # Type variable on right
        if isinstance(right, TypeVariable):
            if self._occurs_check(right, left):
                raise NeuroTypeError(f"Infinite type: {right} occurs in {left}", location)
            self.type_substitutions[right] = left
            return
        
        # Primitive types
        if isinstance(left, PrimitiveType) and isinstance(right, PrimitiveType):
            if left.name != right.name:
                raise NeuroTypeError(
                    f"Type mismatch: cannot unify {left.name} with {right.name}",
                    location
                )
            return
        
        # Tensor types
        if isinstance(left, TensorType) and isinstance(right, TensorType):
            self._unify(left.element_type, right.element_type, location)
            # Allow None to unify with any shape (represents unknown shape)
            if left.shape is not None and right.shape is not None and left.shape != right.shape:
                raise NeuroTypeError(
                    f"Tensor shape mismatch: {left.shape} vs {right.shape}",
                    location
                )
            return
        
        # Function types
        if isinstance(left, FunctionType) and isinstance(right, FunctionType):
            if len(left.param_types) != len(right.param_types):
                raise NeuroTypeError(
                    f"Function arity mismatch: {len(left.param_types)} vs {len(right.param_types)}",
                    location
                )
            
            for lp, rp in zip(left.param_types, right.param_types):
                self._unify(lp, rp, location)
            
            self._unify(left.return_type, right.return_type, location)
            return
        
        # Default case - cannot unify
        raise NeuroTypeError(f"Cannot unify {left} with {right}", location)
    
    def _occurs_check(self, var: TypeVariable, type_expr: Type) -> bool:
        """Check if type variable occurs in type expression (prevents infinite types)."""
        if var == type_expr:
            return True
        
        if isinstance(type_expr, TensorType):
            return self._occurs_check(var, type_expr.element_type)
        elif isinstance(type_expr, FunctionType):
            return (
                any(self._occurs_check(var, pt) for pt in type_expr.param_types) or
                self._occurs_check(var, type_expr.return_type)
            )
        
        return False
    
    def _apply_substitution(self, type_expr: Type) -> Type:
        """Apply current substitutions to a type expression."""
        if isinstance(type_expr, TypeVariable):
            if type_expr in self.type_substitutions:
                # Recursively apply substitutions
                return self._apply_substitution(self.type_substitutions[type_expr])
            return type_expr
        elif isinstance(type_expr, TensorType):
            element_type = self._apply_substitution(type_expr.element_type)
            return TensorType(element_type, type_expr.shape)
        elif isinstance(type_expr, FunctionType):
            param_types = [self._apply_substitution(pt) for pt in type_expr.param_types]
            return_type = self._apply_substitution(type_expr.return_type)
            return FunctionType(param_types, return_type)
        else:
            return type_expr
    
    # ========================================================================
    # Phase 4: Finalization
    # ========================================================================
    
    def _apply_substitutions(self) -> None:
        """Apply all substitutions to the type cache."""
        for node, type_expr in self.type_cache.items():
            self.type_cache[node] = self._apply_substitution(type_expr)
    
    # ========================================================================
    # Built-in Types and Functions
    # ========================================================================
    
    def _init_builtin_types(self) -> None:
        """Initialize built-in types."""
        builtins = [
            ("int", PrimitiveType("int")),
            ("float", PrimitiveType("float")),
            ("bool", PrimitiveType("bool")),
            ("string", PrimitiveType("string")),
        ]
        
        for name, type_obj in builtins:
            symbol_info = SymbolInfo(name, type_obj)
            self.global_scope.define(name, symbol_info)
    
    def _init_builtin_functions(self) -> None:
        """Initialize built-in functions."""
        # print function
        print_type = FunctionType(
            [PrimitiveType("string")],
            PrimitiveType("string")  # Returns input for chaining
        )
        
        print_info = SymbolInfo(
            name="print",
            symbol_type=print_type,
            is_function=True
        )
        
        self.global_scope.define("print", print_info)
        
        # str function (type conversion)
        str_type = FunctionType(
            [self._new_type_variable()],  # Accept any type
            PrimitiveType("string")
        )
        
        str_info = SymbolInfo(
            name="str",
            symbol_type=str_type,
            is_function=True
        )
        
        self.global_scope.define("str", str_info)
        
        # len function
        len_type = FunctionType(
            [TensorType(self._new_type_variable(), None)],  # Accept any tensor
            PrimitiveType("int")
        )
        
        len_info = SymbolInfo(
            name="len",
            symbol_type=len_type,
            is_function=True
        )
        
        self.global_scope.define("len", len_info)
        
        # sqrt function
        sqrt_type = FunctionType(
            [PrimitiveType("float")],
            PrimitiveType("float")
        )
        
        sqrt_info = SymbolInfo(
            name="sqrt",
            symbol_type=sqrt_type,
            is_function=True
        )
        
        self.global_scope.define("sqrt", sqrt_info)
        
        # float type conversion function
        to_float_type = FunctionType(
            [self._new_type_variable()],  # Accept numeric types
            PrimitiveType("float")
        )
        
        to_float_info = SymbolInfo(
            name="to_float",
            symbol_type=to_float_type,
            is_function=True
        )
        
        self.global_scope.define("to_float", to_float_info)
        
        # Also add float as a function (type conversion)
        # Note: This creates a naming conflict with the float type, but we handle it
        # by checking if the symbol is used as a function call vs type annotation
        float_func_type = FunctionType(
            [self._new_type_variable()],  # Accept numeric types
            PrimitiveType("float")
        )
        
        float_func_info = SymbolInfo(
            name="float",
            symbol_type=float_func_type,
            is_function=True
        )
        
        # Override the type definition with the function
        # In a real implementation, we'd have separate namespaces for types and values
        self.global_scope.symbols["float"] = float_func_info
        
        # Neural network related functions
        batch_norm_type = FunctionType(
            [],  # No parameters for now
            self._new_type_variable()  # Return type variable
        )
        
        batch_norm_info = SymbolInfo(
            name="batch_norm",
            symbol_type=batch_norm_type,
            is_function=True
        )
        
        self.global_scope.define("batch_norm", batch_norm_info)
        
        # relu activation function
        relu_type = FunctionType(
            [TensorType(PrimitiveType("float"), None)],
            TensorType(PrimitiveType("float"), None)
        )
        
        relu_info = SymbolInfo(
            name="relu",
            symbol_type=relu_type,
            is_function=True
        )
        
        self.global_scope.define("relu", relu_info)
        
        # softmax activation function
        softmax_type = FunctionType(
            [TensorType(PrimitiveType("float"), None)],
            TensorType(PrimitiveType("float"), None)
        )
        
        softmax_info = SymbolInfo(
            name="softmax",
            symbol_type=softmax_type,
            is_function=True
        )
        
        self.global_scope.define("softmax", softmax_info)
        
        # dropout function
        dropout_type = FunctionType(
            [PrimitiveType("float")],  # rate parameter
            self._new_type_variable()  # Return type variable for layer
        )
        
        dropout_info = SymbolInfo(
            name="dropout",
            symbol_type=dropout_type,
            is_function=True
        )
        
        self.global_scope.define("dropout", dropout_info)
        
        # dense_layer function
        dense_layer_type = FunctionType(
            [PrimitiveType("int"), self._new_type_variable()],  # units and activation
            self._new_type_variable()  # Return type variable for layer
        )
        
        dense_layer_info = SymbolInfo(
            name="dense_layer",
            symbol_type=dense_layer_type,
            is_function=True
        )
        
        self.global_scope.define("dense_layer", dense_layer_info)
        
        # Adam optimizer
        adam_type = FunctionType(
            [PrimitiveType("float")],  # learning_rate parameter
            self._new_type_variable()  # Return optimizer type
        )
        
        adam_info = SymbolInfo(
            name="Adam",
            symbol_type=adam_type,
            is_function=True
        )
        
        self.global_scope.define("Adam", adam_info)
        
        # CrossEntropyLoss
        cross_entropy_type = FunctionType(
            [],  # No parameters for constructor
            self._new_type_variable()  # Return loss function type
        )
        
        cross_entropy_info = SymbolInfo(
            name="CrossEntropyLoss",
            symbol_type=cross_entropy_type,
            is_function=True
        )
        
        self.global_scope.define("CrossEntropyLoss", cross_entropy_info)
        
        # NeuralNetwork constructor (simplified)
        neural_network_type = FunctionType(
            [self._new_type_variable()],  # Generic layer configuration
            self._new_type_variable()  # Return network type
        )
        
        neural_network_info = SymbolInfo(
            name="NeuralNetwork",
            symbol_type=neural_network_type,
            is_function=True
        )
        
        self.global_scope.define("NeuralNetwork", neural_network_info)
        
        # zeros function for creating zero tensors
        zeros_type = FunctionType(
            [PrimitiveType("int"), PrimitiveType("int")],  # dimensions
            TensorType(PrimitiveType("float"), None)  # Return tensor
        )
        
        zeros_info = SymbolInfo(
            name="zeros",
            symbol_type=zeros_type,
            is_function=True
        )
        
        self.global_scope.define("zeros", zeros_info)
        
        # cross_entropy function
        cross_entropy_func_type = FunctionType(
            [TensorType(PrimitiveType("float"), None), TensorType(PrimitiveType("float"), None)],
            PrimitiveType("float")
        )
        
        cross_entropy_func_info = SymbolInfo(
            name="cross_entropy",
            symbol_type=cross_entropy_func_type,
            is_function=True
        )
        
        self.global_scope.define("cross_entropy", cross_entropy_func_info)
        
        # reshape method for tensors (as a standalone function for now)
        reshape_type = FunctionType(
            [TensorType(PrimitiveType("float"), None), PrimitiveType("int"), PrimitiveType("int")],
            TensorType(PrimitiveType("float"), None)
        )
        
        reshape_info = SymbolInfo(
            name="reshape",
            symbol_type=reshape_type,
            is_function=True
        )
        
        self.global_scope.define("reshape", reshape_info)

    def _init_builtin_structs(self) -> None:
        """Initialize built-in struct types and their fields."""
        # NeuralNetwork struct
        nn_struct_name = "NeuralNetwork"
        
        # Check if NeuralNetwork is already defined (likely as a function)
        if self.global_scope.lookup(nn_struct_name) is None:
            nn_type = PrimitiveType(nn_struct_name) # Structs are represented by PrimitiveType of their name
            # Define the NeuralNetwork type in the global scope
            # This allows 'NeuralNetwork' to be recognized as a type name.
            self.global_scope.define(nn_struct_name, SymbolInfo(name=nn_struct_name, symbol_type=nn_type, is_function=False))

        # Initialize struct_fields if it doesn't exist
        if not hasattr(self, 'struct_fields'):
            self.struct_fields = {}
        
        # Define fields for NeuralNetwork.
        # The 'layers' field is expected to be a list/tensor of layer-like objects.
        # The type of elements in 'layers' can be a TypeVariable or a more specific 'LayerType' if defined.
        layer_element_type = self._new_type_variable() # Placeholder for individual layer type
        layers_type = TensorType(layer_element_type, None) # Representing a list/collection of layers
        
        self.struct_fields[nn_struct_name] = {
            "layers": layers_type
            # Add other built-in fields if NeuralNetwork struct has them, e.g.:
            # "optimizer": self._new_type_variable(),
            # "loss_function": self._new_type_variable()
        }

        # If there was a function 'NeuralNetwork' that conflicted, decide precedence.
        # For struct initialization syntax `NeuralNetwork { ... }`, it must be known as a struct type.
        # If `NeuralNetwork(...)` is also a valid function call, they need to be distinguishable
        # or the language rules must clarify. For now, defining it as a struct type for the
        # initializer syntax is key.
        # The function definition for `NeuralNetwork` from `_init_builtin_functions` might still exist.
        # The parser and type checker need to correctly distinguish between `NeuralNetwork` as a type
        # (for `StructInitializer`) and as a function (for `FunctionCall`).
        # `_check_function_call` has a special check for struct names used as functions,
        # which might need review if `NeuralNetwork` is both a struct type and a globally defined function.

    def _check_model_definition(self, model_def: ModelDefinition) -> Type:
        """Handle type checking for neural network model definitions"""
        # Create a new scope for the model initializer
        with self._enter_scope(f"model_{model_def.name or 'anonymous'}"):
            # Check each layer in the model
            for layer in model_def.layers:
                self._check_expression(layer)
                
            # Create a new type variable for the model type
            model_type = self._new_type_variable(model_def.location)
            return model_type 