"""
Abstract Syntax Tree Node Definitions
Defines all AST nodes for the NEURO programming language.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .errors import SourceLocation


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    def __init__(self, location: Optional[SourceLocation] = None):
        self.location = location
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor for the visitor pattern."""
        pass


# ============================================================================
# Type System
# ============================================================================

class Type(ASTNode):
    """Base class for type annotations."""
    pass


@dataclass
class PrimitiveType(Type):
    """Primitive type (int, float, string, bool)."""
    name: str
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_primitive_type(self)
    
    def __str__(self) -> str:
        return self.name


@dataclass
class GenericType(Type):
    """Generic type with type parameters."""
    name: str
    type_params: List[Type]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_generic_type(self)
    
    def __str__(self) -> str:
        if self.type_params:
            params = ", ".join(str(p) for p in self.type_params)
            return f"{self.name}<{params}>"
        return self.name


@dataclass
class TensorType(Type):
    """Tensor type with element type and optional shape."""
    element_type: Type
    shape: Optional[List[int]] = None
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_tensor_type(self)
    
    def __str__(self) -> str:
        if self.shape:
            shape_str = ", ".join(str(d) for d in self.shape)
            return f"Tensor<{self.element_type}, ({shape_str})>"
        return f"Tensor<{self.element_type}>"


@dataclass
class FunctionType(Type):
    """Function type with parameter types and return type."""
    param_types: List[Type]
    return_type: Type
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_function_type(self)
    
    def __str__(self) -> str:
        params = ", ".join(str(p) for p in self.param_types)
        return f"({params}) -> {self.return_type}"


# ============================================================================
# Expressions
# ============================================================================

class Expression(ASTNode):
    """Base class for all expressions."""
    pass


@dataclass
class LiteralExpression(Expression):
    """Literal expression (number, string, boolean)."""
    value: Any
    type_hint: Optional[Type] = None
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_literal_expression(self)
    
    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


@dataclass
class IdentifierExpression(Expression):
    """Identifier expression."""
    name: str
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_identifier_expression(self)
    
    def __str__(self) -> str:
        return self.name


@dataclass
class BinaryExpression(Expression):
    """Binary expression."""
    left: Expression
    operator: str
    right: Expression
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_binary_expression(self)
    
    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class UnaryExpression(Expression):
    """Unary expression."""
    operator: str
    operand: Expression
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_unary_expression(self)
    
    def __str__(self) -> str:
        return f"({self.operator}{self.operand})"


@dataclass
class CallExpression(Expression):
    """Function call expression."""
    function: Expression
    arguments: List[Expression]
    type_args: Optional[List[Type]] = None
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_call_expression(self)
    
    def __str__(self) -> str:
        args = ", ".join(str(arg) for arg in self.arguments)
        if self.type_args:
            type_args = ", ".join(str(t) for t in self.type_args)
            return f"{self.function}<{type_args}>({args})"
        return f"{self.function}({args})"


@dataclass
class MemberExpression(Expression):
    """Member access expression (obj.member)."""
    object: Expression
    member: str
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_member_expression(self)
    
    def __str__(self) -> str:
        return f"{self.object}.{self.member}"


@dataclass
class IndexExpression(Expression):
    """Index access expression (arr[index])."""
    object: Expression
    index: Expression
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_index_expression(self)
    
    def __str__(self) -> str:
        return f"{self.object}[{self.index}]"


@dataclass
class ArrayExpression(Expression):
    """Array literal expression."""
    elements: List[Expression]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_array_expression(self)
    
    def __str__(self) -> str:
        elements = ", ".join(str(e) for e in self.elements)
        return f"[{elements}]"


@dataclass
class StructExpression(Expression):
    """Struct literal expression."""
    type_name: str
    fields: List[tuple[str, Expression]]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_struct_expression(self)
    
    def __str__(self) -> str:
        fields = ", ".join(f"{name}: {expr}" for name, expr in self.fields)
        return f"{self.type_name} {{ {fields} }}"


# ============================================================================
# Statements
# ============================================================================

class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass
class ExpressionStatement(Statement):
    """Expression statement."""
    expression: Expression
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_expression_statement(self)
    
    def __str__(self) -> str:
        return str(self.expression)


@dataclass
class VariableDeclaration(Statement):
    """Variable declaration statement."""
    name: str
    type_annotation: Optional[Type]
    initializer: Optional[Expression]
    is_mutable: bool = True
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_variable_declaration(self)
    
    def __str__(self) -> str:
        result = f"let {self.name}"
        if self.type_annotation:
            result += f": {self.type_annotation}"
        if self.initializer:
            result += f" = {self.initializer}"
        return result


@dataclass
class AssignmentStatement(Statement):
    """Assignment statement."""
    target: Expression
    value: Expression
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_assignment_statement(self)
    
    def __str__(self) -> str:
        return f"{self.target} = {self.value}"


@dataclass
class IfStatement(Statement):
    """If statement."""
    condition: Expression
    then_body: List[Statement]
    else_body: Optional[List[Statement]] = None
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_if_statement(self)
    
    def __str__(self) -> str:
        result = f"if {self.condition} {{ ... }}"
        if self.else_body:
            result += " else { ... }"
        return result


@dataclass
class ForStatement(Statement):
    """For loop statement."""
    variable: str
    iterable: Expression
    body: List[Statement]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_for_statement(self)
    
    def __str__(self) -> str:
        return f"for {self.variable} in {self.iterable} {{ ... }}"


@dataclass
class ReturnStatement(Statement):
    """Return statement."""
    value: Optional[Expression]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_return_statement(self)
    
    def __str__(self) -> str:
        if self.value:
            return f"return {self.value}"
        return "return"


@dataclass
class BlockStatement(Statement):
    """Block statement."""
    statements: List[Statement]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_block_statement(self)
    
    def __str__(self) -> str:
        return f"{{ {len(self.statements)} statements }}"


# ============================================================================
# Declarations
# ============================================================================

class Declaration(ASTNode):
    """Base class for all declarations."""
    pass


@dataclass
class Parameter:
    """Function parameter."""
    name: str
    type_annotation: Optional[Type]
    default_value: Optional[Expression] = None
    
    def __str__(self) -> str:
        result = self.name
        if self.type_annotation:
            result += f": {self.type_annotation}"
        if self.default_value:
            result += f" = {self.default_value}"
        return result


@dataclass
class FunctionDeclaration(Declaration):
    """Function declaration."""
    name: str
    type_params: Optional[List[str]]
    parameters: List[Parameter]
    return_type: Optional[Type]
    body: List[Statement]
    is_generic: bool = False
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_function_declaration(self)
    
    def __str__(self) -> str:
        result = f"func {self.name}"
        if self.type_params:
            type_params = ", ".join(self.type_params)
            result += f"<{type_params}>"
        params = ", ".join(str(p) for p in self.parameters)
        result += f"({params})"
        if self.return_type:
            result += f" -> {self.return_type}"
        return result + " { ... }"


@dataclass
class StructField:
    """Struct field."""
    name: str
    type_annotation: Type
    
    def __str__(self) -> str:
        return f"{self.name}: {self.type_annotation}"


@dataclass
class StructDeclaration(Declaration):
    """Struct declaration."""
    name: str
    type_params: Optional[List[str]]
    fields: List[StructField]
    is_generic: bool = False
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_struct_declaration(self)
    
    def __str__(self) -> str:
        result = f"struct {self.name}"
        if self.type_params:
            type_params = ", ".join(self.type_params)
            result += f"<{type_params}>"
        return result + " { ... }"


# ============================================================================
# Program
# ============================================================================

@dataclass
class Program(ASTNode):
    """Top-level program node."""
    declarations: List[Declaration]
    statements: List[Statement]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_program(self)
    
    def __str__(self) -> str:
        decl_count = len(self.declarations)
        stmt_count = len(self.statements)
        return f"Program({decl_count} declarations, {stmt_count} statements)"


# ============================================================================
# Neural Network Specific
# ============================================================================

@dataclass
class NeuralNetworkExpression(Expression):
    """Neural network definition expression."""
    element_type: Type
    shape_params: List[int]
    layers: List[Expression]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_neural_network_expression(self)
    
    def __str__(self) -> str:
        shape = ", ".join(str(s) for s in self.shape_params)
        return f"NeuralNetwork<{self.element_type}, ({shape})> {{ ... }}"


@dataclass
class LayerExpression(Expression):
    """Neural network layer expression."""
    layer_type: str
    config: List[Expression]
    location: Optional[SourceLocation] = None
    
    def __post_init__(self):
        super().__init__(self.location)
    
    def accept(self, visitor):
        return visitor.visit_layer_expression(self)
    
    def __str__(self) -> str:
        config_str = ", ".join(str(c) for c in self.config)
        return f"{self.layer_type}({config_str})"


# ============================================================================
# Visitor Pattern
# ============================================================================

class ASTVisitor(ABC):
    """Abstract visitor for AST nodes."""
    
    @abstractmethod
    def visit_program(self, node: Program): pass
    
    @abstractmethod
    def visit_function_declaration(self, node: FunctionDeclaration): pass
    
    @abstractmethod
    def visit_struct_declaration(self, node: StructDeclaration): pass
    
    @abstractmethod
    def visit_variable_declaration(self, node: VariableDeclaration): pass
    
    @abstractmethod
    def visit_assignment_statement(self, node: AssignmentStatement): pass
    
    @abstractmethod
    def visit_expression_statement(self, node: ExpressionStatement): pass
    
    @abstractmethod
    def visit_if_statement(self, node: IfStatement): pass
    
    @abstractmethod
    def visit_for_statement(self, node: ForStatement): pass
    
    @abstractmethod
    def visit_return_statement(self, node: ReturnStatement): pass
    
    @abstractmethod
    def visit_block_statement(self, node: BlockStatement): pass
    
    @abstractmethod
    def visit_literal_expression(self, node: LiteralExpression): pass
    
    @abstractmethod
    def visit_identifier_expression(self, node: IdentifierExpression): pass
    
    @abstractmethod
    def visit_binary_expression(self, node: BinaryExpression): pass
    
    @abstractmethod
    def visit_unary_expression(self, node: UnaryExpression): pass
    
    @abstractmethod
    def visit_call_expression(self, node: CallExpression): pass
    
    @abstractmethod
    def visit_member_expression(self, node: MemberExpression): pass
    
    @abstractmethod
    def visit_index_expression(self, node: IndexExpression): pass
    
    @abstractmethod
    def visit_array_expression(self, node: ArrayExpression): pass
    
    @abstractmethod
    def visit_struct_expression(self, node: StructExpression): pass
    
    @abstractmethod
    def visit_primitive_type(self, node: PrimitiveType): pass
    
    @abstractmethod
    def visit_generic_type(self, node: GenericType): pass
    
    @abstractmethod
    def visit_tensor_type(self, node: TensorType): pass
    
    @abstractmethod
    def visit_function_type(self, node: FunctionType): pass
    
    @abstractmethod
    def visit_neural_network_expression(self, node: NeuralNetworkExpression): pass
    
    @abstractmethod
    def visit_layer_expression(self, node: LayerExpression): pass 