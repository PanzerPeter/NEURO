"""
Type Checking Slice
Performs type inference, constraint solving, and semantic analysis for NEURO.
"""

from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass
from enum import Enum, auto

from .ast_nodes import *
from .errors import ErrorReporter, NeuroTypeError, NeuroSemanticError, SourceLocation


class TypeKind(Enum):
    """Different kinds of types in the type system."""
    PRIMITIVE = auto()
    TENSOR = auto()
    FUNCTION = auto()
    STRUCT = auto()
    GENERIC = auto()
    TYPE_VARIABLE = auto()
    NEURAL_NETWORK = auto()


@dataclass
class TypeVariable:
    """Represents a type variable for inference."""
    id: int
    name: str
    constraints: List['TypeConstraint'] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
    
    def __str__(self) -> str:
        return f"?{self.name}{self.id}"


@dataclass 
class TypeConstraint:
    """Represents a constraint between types."""
    left: Union[Type, TypeVariable]
    right: Union[Type, TypeVariable]
    kind: str  # 'unify', 'subtype', 'has_method', etc.
    location: Optional[SourceLocation] = None


class TypeEnvironment:
    """Manages type bindings in different scopes."""
    
    def __init__(self, parent: Optional['TypeEnvironment'] = None):
        self.parent = parent
        self.bindings: Dict[str, Union[Type, TypeVariable]] = {}
        self.type_params: Set[str] = set()
    
    def bind(self, name: str, type_val: Union[Type, TypeVariable]) -> None:
        """Bind a name to a type in the current scope."""
        self.bindings[name] = type_val
    
    def lookup(self, name: str) -> Optional[Union[Type, TypeVariable]]:
        """Look up a type binding."""
        if name in self.bindings:
            return self.bindings[name]
        elif self.parent:
            return self.parent.lookup(name)
        return None
    
    def bind_type_param(self, name: str) -> None:
        """Bind a type parameter."""
        self.type_params.add(name)
    
    def is_type_param(self, name: str) -> bool:
        """Check if a name is a type parameter."""
        return (name in self.type_params or 
                (self.parent and self.parent.is_type_param(name)))
    
    def create_child(self) -> 'TypeEnvironment':
        """Create a child environment."""
        return TypeEnvironment(self)


class NeuroTypeChecker:
    """Type checker for the NEURO programming language."""
    
    def __init__(self):
        self.error_reporter = ErrorReporter()
        self.type_var_counter = 0
        self.constraints: List[TypeConstraint] = []
        self.substitutions: Dict[int, Union[Type, TypeVariable]] = {}
        
        # Create global environment with built-ins
        self.global_env = TypeEnvironment()
        self.current_env = self.global_env
        
        # Initialize built-in types and functions
        self._init_builtins()
    
    def _init_builtins(self) -> None:
        """Initialize built-in types and functions."""
        # Built-in types
        self.global_env.bind("int", PrimitiveType("int"))
        self.global_env.bind("float", PrimitiveType("float"))
        self.global_env.bind("string", PrimitiveType("string"))
        self.global_env.bind("bool", PrimitiveType("bool"))
        
        # Built-in functions
        print_type = FunctionType([PrimitiveType("string")], PrimitiveType("string"))
        self.global_env.bind("print", print_type)
        
        str_type = FunctionType([self._create_type_variable("T")], PrimitiveType("string"))
        self.global_env.bind("str", str_type)
        
        float_type = FunctionType([PrimitiveType("int")], PrimitiveType("float"))
        self.global_env.bind("float", float_type)
        
        # Tensor creation functions
        zeros_type = FunctionType([PrimitiveType("int"), PrimitiveType("int")], 
                                 TensorType(PrimitiveType("float"), [None, None]))
        self.global_env.bind("zeros", zeros_type)
    
    def _create_type_variable(self, name: str = "T") -> TypeVariable:
        """Create a fresh type variable."""
        var = TypeVariable(self.type_var_counter, name)
        self.type_var_counter += 1
        return var
    
    def _add_constraint(self, left: Union[Type, TypeVariable], 
                       right: Union[Type, TypeVariable], 
                       kind: str = "unify",
                       location: Optional[SourceLocation] = None) -> None:
        """Add a type constraint."""
        constraint = TypeConstraint(left, right, kind, location)
        self.constraints.append(constraint)
    
    def check_program(self, program: Program) -> Program:
        """Type check an entire program."""
        # First pass: collect all declarations
        for decl in program.declarations:
            if isinstance(decl, FunctionDeclaration):
                self._collect_function_declaration(decl)
            elif isinstance(decl, StructDeclaration):
                self._collect_struct_declaration(decl)
        
        # Second pass: type check declarations
        for decl in program.declarations:
            self.check_declaration(decl)
        
        # Type check top-level statements
        for stmt in program.statements:
            self.check_statement(stmt)
        
        # Solve all constraints
        self._solve_constraints()
        
        # Report errors if any
        if self.error_reporter.has_errors():
            self.error_reporter.print_errors()
            raise NeuroTypeError("Type checking failed")
        
        return program
    
    def _collect_function_declaration(self, decl: FunctionDeclaration) -> None:
        """Collect function declaration for forward reference."""
        # Build function type
        param_types = []
        for param in decl.parameters:
            if param.type_annotation:
                param_types.append(param.type_annotation)
            else:
                # Create type variable for unspecified parameter types
                param_types.append(self._create_type_variable(f"param_{param.name}"))
        
        return_type = decl.return_type or self._create_type_variable(f"ret_{decl.name}")
        
        func_type = FunctionType(param_types, return_type)
        self.global_env.bind(decl.name, func_type)
    
    def _collect_struct_declaration(self, decl: StructDeclaration) -> None:
        """Collect struct declaration for forward reference."""
        # For now, just bind the struct name
        struct_type = GenericType(decl.name, [])
        self.global_env.bind(decl.name, struct_type)
    
    def check_declaration(self, decl: Declaration) -> None:
        """Type check a declaration."""
        if isinstance(decl, FunctionDeclaration):
            self._check_function_declaration(decl)
        elif isinstance(decl, StructDeclaration):
            self._check_struct_declaration(decl)
    
    def _check_function_declaration(self, decl: FunctionDeclaration) -> None:
        """Type check a function declaration."""
        # Create new environment for function scope
        func_env = self.current_env.create_child()
        old_env = self.current_env
        self.current_env = func_env
        
        # Bind type parameters if generic
        if decl.type_params:
            for type_param in decl.type_params:
                func_env.bind_type_param(type_param)
        
        # Bind parameters
        for param in decl.parameters:
            param_type = param.type_annotation or self._create_type_variable(f"param_{param.name}")
            func_env.bind(param.name, param_type)
        
        # Type check function body
        return_type_var = self._create_type_variable(f"ret_{decl.name}")
        
        for stmt in decl.body:
            stmt_type = self.check_statement(stmt)
            
            # If it's a return statement, constrain with return type
            if isinstance(stmt, ReturnStatement) and stmt.value:
                expr_type = self.check_expression(stmt.value)
                if decl.return_type:
                    self._add_constraint(expr_type, decl.return_type, "unify", stmt.location)
                else:
                    self._add_constraint(expr_type, return_type_var, "unify", stmt.location)
        
        # Restore environment
        self.current_env = old_env
    
    def _check_struct_declaration(self, decl: StructDeclaration) -> None:
        """Type check a struct declaration."""
        # Validate field types
        for field in decl.fields:
            self._validate_type(field.type_annotation)
    
    def check_statement(self, stmt: Statement) -> Optional[Union[Type, TypeVariable]]:
        """Type check a statement."""
        if isinstance(stmt, ExpressionStatement):
            return self.check_expression(stmt.expression)
        elif isinstance(stmt, VariableDeclaration):
            return self._check_variable_declaration(stmt)
        elif isinstance(stmt, AssignmentStatement):
            return self._check_assignment_statement(stmt)
        elif isinstance(stmt, IfStatement):
            return self._check_if_statement(stmt)
        elif isinstance(stmt, ForStatement):
            return self._check_for_statement(stmt)
        elif isinstance(stmt, ReturnStatement):
            return self._check_return_statement(stmt)
        elif isinstance(stmt, BlockStatement):
            return self._check_block_statement(stmt)
        
        return None
    
    def _check_variable_declaration(self, stmt: VariableDeclaration) -> Union[Type, TypeVariable]:
        """Type check a variable declaration."""
        if stmt.initializer:
            init_type = self.check_expression(stmt.initializer)
            
            if stmt.type_annotation:
                # Check that initializer type matches annotation
                self._add_constraint(init_type, stmt.type_annotation, "unify", stmt.location)
                var_type = stmt.type_annotation
            else:
                # Infer type from initializer
                var_type = init_type
            
            self.current_env.bind(stmt.name, var_type)
            return var_type
        elif stmt.type_annotation:
            # Declaration without initializer
            self.current_env.bind(stmt.name, stmt.type_annotation)
            return stmt.type_annotation
        else:
            # Error: need either type annotation or initializer
            self.error_reporter.type_error(
                f"Variable '{stmt.name}' needs either type annotation or initializer",
                stmt.location
            )
            return self._create_type_variable(f"error_{stmt.name}")
    
    def _check_assignment_statement(self, stmt: AssignmentStatement) -> Union[Type, TypeVariable]:
        """Type check an assignment statement."""
        target_type = self.check_expression(stmt.target)
        value_type = self.check_expression(stmt.value)
        
        # Ensure assignment type compatibility
        self._add_constraint(target_type, value_type, "unify", stmt.location)
        
        return value_type
    
    def _check_if_statement(self, stmt: IfStatement) -> None:
        """Type check an if statement."""
        # Check condition is boolean
        cond_type = self.check_expression(stmt.condition)
        bool_type = PrimitiveType("bool")
        self._add_constraint(cond_type, bool_type, "unify", stmt.location)
        
        # Check then body
        for then_stmt in stmt.then_body:
            self.check_statement(then_stmt)
        
        # Check else body if present
        if stmt.else_body:
            for else_stmt in stmt.else_body:
                self.check_statement(else_stmt)
    
    def _check_for_statement(self, stmt: ForStatement) -> None:
        """Type check a for statement."""
        # Check iterable type
        iterable_type = self.check_expression(stmt.iterable)
        
        # Create new scope for loop variable
        loop_env = self.current_env.create_child()
        old_env = self.current_env
        self.current_env = loop_env
        
        # Infer loop variable type from iterable
        if isinstance(iterable_type, TensorType):
            element_type = iterable_type.element_type
        else:
            # For now, create a type variable
            element_type = self._create_type_variable(f"elem_{stmt.variable}")
        
        loop_env.bind(stmt.variable, element_type)
        
        # Check loop body
        for body_stmt in stmt.body:
            self.check_statement(body_stmt)
        
        # Restore environment
        self.current_env = old_env
    
    def _check_return_statement(self, stmt: ReturnStatement) -> Optional[Union[Type, TypeVariable]]:
        """Type check a return statement."""
        if stmt.value:
            return self.check_expression(stmt.value)
        return None
    
    def _check_block_statement(self, stmt: BlockStatement) -> None:
        """Type check a block statement."""
        # Create new scope
        block_env = self.current_env.create_child()
        old_env = self.current_env
        self.current_env = block_env
        
        # Check all statements in block
        for block_stmt in stmt.statements:
            self.check_statement(block_stmt)
        
        # Restore environment
        self.current_env = old_env
    
    def check_expression(self, expr: Expression) -> Union[Type, TypeVariable]:
        """Type check an expression."""
        if isinstance(expr, LiteralExpression):
            return self._check_literal_expression(expr)
        elif isinstance(expr, IdentifierExpression):
            return self._check_identifier_expression(expr)
        elif isinstance(expr, BinaryExpression):
            return self._check_binary_expression(expr)
        elif isinstance(expr, UnaryExpression):
            return self._check_unary_expression(expr)
        elif isinstance(expr, CallExpression):
            return self._check_call_expression(expr)
        elif isinstance(expr, MemberExpression):
            return self._check_member_expression(expr)
        elif isinstance(expr, IndexExpression):
            return self._check_index_expression(expr)
        elif isinstance(expr, ArrayExpression):
            return self._check_array_expression(expr)
        elif isinstance(expr, StructExpression):
            return self._check_struct_expression(expr)
        elif isinstance(expr, NeuralNetworkExpression):
            return self._check_neural_network_expression(expr)
        
        # Default case
        return self._create_type_variable("unknown")
    
    def _check_literal_expression(self, expr: LiteralExpression) -> Type:
        """Type check a literal expression."""
        if isinstance(expr.value, int):
            return PrimitiveType("int")
        elif isinstance(expr.value, float):
            return PrimitiveType("float")
        elif isinstance(expr.value, str):
            return PrimitiveType("string")
        elif isinstance(expr.value, bool):
            return PrimitiveType("bool")
        else:
            return self._create_type_variable("literal")
    
    def _check_identifier_expression(self, expr: IdentifierExpression) -> Union[Type, TypeVariable]:
        """Type check an identifier expression."""
        var_type = self.current_env.lookup(expr.name)
        if var_type is None:
            self.error_reporter.semantic_error(
                f"Undefined variable '{expr.name}'",
                expr.location
            )
            return self._create_type_variable(f"undefined_{expr.name}")
        return var_type
    
    def _check_binary_expression(self, expr: BinaryExpression) -> Union[Type, TypeVariable]:
        """Type check a binary expression."""
        left_type = self.check_expression(expr.left)
        right_type = self.check_expression(expr.right)
        
        # Handle different operators
        if expr.operator in ['+', '-', '*', '/', '%']:
            return self._check_arithmetic_binary(expr, left_type, right_type)
        elif expr.operator == '@':
            return self._check_matrix_multiply(expr, left_type, right_type)
        elif expr.operator in ['==', '!=', '<', '<=', '>', '>=']:
            return self._check_comparison_binary(expr, left_type, right_type)
        elif expr.operator in ['and', 'or']:
            return self._check_logical_binary(expr, left_type, right_type)
        else:
            # Default: return type variable
            result_type = self._create_type_variable(f"binop_{expr.operator}")
            return result_type
    
    def _check_arithmetic_binary(self, expr: BinaryExpression, 
                                left_type: Union[Type, TypeVariable], 
                                right_type: Union[Type, TypeVariable]) -> Union[Type, TypeVariable]:
        """Type check arithmetic binary operations."""
        # For arithmetic operations, both operands should be numeric
        # Result type depends on operand types
        
        if isinstance(left_type, PrimitiveType) and isinstance(right_type, PrimitiveType):
            if left_type.name == "float" or right_type.name == "float":
                return PrimitiveType("float")
            elif left_type.name == "int" and right_type.name == "int":
                return PrimitiveType("int")
            elif left_type.name == "string" and right_type.name == "string" and expr.operator == '+':
                return PrimitiveType("string")
        
        # For tensors, element-wise operations
        if isinstance(left_type, TensorType) and isinstance(right_type, TensorType):
            # Check shapes are compatible (for now, just use left type)
            return left_type
        
        # Create constraints for type variables
        result_type = self._create_type_variable(f"arith_{expr.operator}")
        return result_type
    
    def _check_matrix_multiply(self, expr: BinaryExpression,
                              left_type: Union[Type, TypeVariable],
                              right_type: Union[Type, TypeVariable]) -> Union[Type, TypeVariable]:
        """Type check matrix multiplication (@)."""
        # For matrix multiplication, we need compatible tensor shapes
        if isinstance(left_type, TensorType) and isinstance(right_type, TensorType):
            # For vector @ vector, result is scalar
            # For matrix @ matrix, result is matrix
            # For now, return the element type for dot product
            if (left_type.shape and len(left_type.shape) == 1 and
                right_type.shape and len(right_type.shape) == 1):
                return left_type.element_type  # Dot product returns scalar
            else:
                return left_type  # Matrix multiplication returns tensor
        
        result_type = self._create_type_variable("matmul")
        return result_type
    
    def _check_comparison_binary(self, expr: BinaryExpression,
                                left_type: Union[Type, TypeVariable],
                                right_type: Union[Type, TypeVariable]) -> Type:
        """Type check comparison operations."""
        # Comparison operations always return bool
        # But we should check that operands are comparable
        return PrimitiveType("bool")
    
    def _check_logical_binary(self, expr: BinaryExpression,
                             left_type: Union[Type, TypeVariable],
                             right_type: Union[Type, TypeVariable]) -> Type:
        """Type check logical operations."""
        # Logical operations require bool operands and return bool
        bool_type = PrimitiveType("bool")
        self._add_constraint(left_type, bool_type, "unify", expr.location)
        self._add_constraint(right_type, bool_type, "unify", expr.location)
        return bool_type
    
    def _check_unary_expression(self, expr: UnaryExpression) -> Union[Type, TypeVariable]:
        """Type check a unary expression."""
        operand_type = self.check_expression(expr.operand)
        
        if expr.operator == '-':
            # Unary minus preserves numeric type
            return operand_type
        elif expr.operator == 'not':
            # Logical not requires bool operand and returns bool
            bool_type = PrimitiveType("bool")
            self._add_constraint(operand_type, bool_type, "unify", expr.location)
            return bool_type
        
        return self._create_type_variable(f"unary_{expr.operator}")
    
    def _check_call_expression(self, expr: CallExpression) -> Union[Type, TypeVariable]:
        """Type check a function call expression."""
        func_type = self.check_expression(expr.function)
        
        # Check argument types
        arg_types = [self.check_expression(arg) for arg in expr.arguments]
        
        if isinstance(func_type, FunctionType):
            # Check argument count
            if len(arg_types) != len(func_type.param_types):
                self.error_reporter.type_error(
                    f"Function expects {len(func_type.param_types)} arguments, got {len(arg_types)}",
                    expr.location
                )
            
            # Check argument types
            for i, (arg_type, param_type) in enumerate(zip(arg_types, func_type.param_types)):
                self._add_constraint(arg_type, param_type, "unify", expr.location)
            
            return func_type.return_type
        
        # For type variables or unknown functions, create result type variable
        result_type = self._create_type_variable("call_result")
        return result_type
    
    def _check_member_expression(self, expr: MemberExpression) -> Union[Type, TypeVariable]:
        """Type check a member access expression."""
        obj_type = self.check_expression(expr.object)
        
        # For structs, look up field type
        # For tensors, handle special methods
        # For now, create type variable
        return self._create_type_variable(f"member_{expr.member}")
    
    def _check_index_expression(self, expr: IndexExpression) -> Union[Type, TypeVariable]:
        """Type check an index access expression."""
        obj_type = self.check_expression(expr.object)
        index_type = self.check_expression(expr.index)
        
        # Index should be integer
        int_type = PrimitiveType("int")
        self._add_constraint(index_type, int_type, "unify", expr.location)
        
        # For tensors, return element type
        if isinstance(obj_type, TensorType):
            return obj_type.element_type
        
        # For arrays, create element type variable
        return self._create_type_variable("index_result")
    
    def _check_array_expression(self, expr: ArrayExpression) -> Union[Type, TypeVariable]:
        """Type check an array literal expression."""
        if not expr.elements:
            # Empty array - create type variable for element type
            element_type = self._create_type_variable("empty_array_elem")
            return TensorType(element_type)
        
        # Check all elements have the same type
        element_types = [self.check_expression(elem) for elem in expr.elements]
        first_type = element_types[0]
        
        for i, elem_type in enumerate(element_types[1:], 1):
            self._add_constraint(first_type, elem_type, "unify", expr.location)
        
        # Determine if it's a nested array (matrix)
        if (expr.elements and isinstance(expr.elements[0], ArrayExpression)):
            # Matrix case
            rows = len(expr.elements)
            cols = len(expr.elements[0].elements) if expr.elements[0].elements else 0
            return TensorType(first_type, [rows, cols])
        else:
            # Vector case
            return TensorType(first_type, [len(expr.elements)])
    
    def _check_struct_expression(self, expr: StructExpression) -> Union[Type, TypeVariable]:
        """Type check a struct literal expression."""
        # Look up struct type
        struct_type = self.current_env.lookup(expr.type_name)
        
        if struct_type is None:
            self.error_reporter.semantic_error(
                f"Undefined struct type '{expr.type_name}'",
                expr.location
            )
            return self._create_type_variable(f"undefined_struct_{expr.type_name}")
        
        # For now, just return the struct type
        return struct_type
    
    def _check_neural_network_expression(self, expr: NeuralNetworkExpression) -> Union[Type, TypeVariable]:
        """Type check a neural network expression."""
        # Create neural network type
        element_type = expr.element_type
        shape_params = expr.shape_params
        
        # For now, create a generic type representing the neural network
        return GenericType("NeuralNetwork", [element_type])
    
    def _validate_type(self, type_val: Type) -> bool:
        """Validate that a type is well-formed."""
        if isinstance(type_val, PrimitiveType):
            return type_val.name in ["int", "float", "string", "bool"]
        elif isinstance(type_val, TensorType):
            return self._validate_type(type_val.element_type)
        elif isinstance(type_val, FunctionType):
            return (all(self._validate_type(pt) for pt in type_val.param_types) and
                   self._validate_type(type_val.return_type))
        elif isinstance(type_val, GenericType):
            return all(self._validate_type(tp) for tp in type_val.type_params)
        
        return True
    
    def _solve_constraints(self) -> None:
        """Solve all type constraints using unification."""
        # Simple constraint solver - unify constraints
        for constraint in self.constraints:
            try:
                self._unify(constraint.left, constraint.right)
            except Exception as e:
                if constraint.location:
                    self.error_reporter.type_error(
                        f"Type constraint failed: {e}",
                        constraint.location
                    )
                else:
                    self.error_reporter.type_error(f"Type constraint failed: {e}")
    
    def _unify(self, left: Union[Type, TypeVariable], right: Union[Type, TypeVariable]) -> None:
        """Unify two types."""
        # Apply substitutions first
        left = self._apply_substitution(left)
        right = self._apply_substitution(right)
        
        # Same type
        if left == right:
            return
        
        # Type variable cases
        if isinstance(left, TypeVariable):
            self.substitutions[left.id] = right
            return
        elif isinstance(right, TypeVariable):
            self.substitutions[right.id] = left
            return
        
        # Primitive type cases
        if isinstance(left, PrimitiveType) and isinstance(right, PrimitiveType):
            if left.name != right.name:
                raise NeuroTypeError(f"Cannot unify {left.name} with {right.name}")
            return
        
        # Tensor type cases
        if isinstance(left, TensorType) and isinstance(right, TensorType):
            self._unify(left.element_type, right.element_type)
            # For now, don't check shape compatibility strictly
            return
        
        # Function type cases
        if isinstance(left, FunctionType) and isinstance(right, FunctionType):
            if len(left.param_types) != len(right.param_types):
                raise NeuroTypeError("Function arity mismatch")
            
            for lp, rp in zip(left.param_types, right.param_types):
                self._unify(lp, rp)
            
            self._unify(left.return_type, right.return_type)
            return
        
        # Generic type cases  
        if isinstance(left, GenericType) and isinstance(right, GenericType):
            if left.name != right.name:
                raise NeuroTypeError(f"Cannot unify {left.name} with {right.name}")
            
            if len(left.type_params) != len(right.type_params):
                raise NeuroTypeError("Generic type parameter count mismatch")
            
            for ltp, rtp in zip(left.type_params, right.type_params):
                self._unify(ltp, rtp)
            return
        
        # Default case - types don't unify
        raise NeuroTypeError(f"Cannot unify {type(left).__name__} with {type(right).__name__}")
    
    def _apply_substitution(self, type_val: Union[Type, TypeVariable]) -> Union[Type, TypeVariable]:
        """Apply current substitutions to a type."""
        if isinstance(type_val, TypeVariable):
            if type_val.id in self.substitutions:
                # Recursively apply substitution
                return self._apply_substitution(self.substitutions[type_val.id])
            return type_val
        elif isinstance(type_val, TensorType):
            return TensorType(
                self._apply_substitution(type_val.element_type),
                type_val.shape
            )
        elif isinstance(type_val, FunctionType):
            return FunctionType(
                [self._apply_substitution(pt) for pt in type_val.param_types],
                self._apply_substitution(type_val.return_type)
            )
        elif isinstance(type_val, GenericType):
            return GenericType(
                type_val.name,
                [self._apply_substitution(tp) for tp in type_val.type_params]
            )
        
        return type_val
    
    def get_error_reporter(self) -> ErrorReporter:
        """Get the error reporter."""
        return self.error_reporter 