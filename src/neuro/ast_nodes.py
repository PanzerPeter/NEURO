"""
NEURO Abstract Syntax Tree (AST) Node Definitions

Defines all AST node types for the NEURO programming language.
These nodes represent the parsed structure of NEURO programs
and are used by the type checker and code generator.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
from enum import Enum

from .errors import SourceLocation


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    def __init__(self, location: Optional[SourceLocation] = None):
        self.location = location
    
    @abstractmethod
    def pretty_print(self, indent: int = 0) -> str:
        """Return a pretty-printed representation of this node."""
        pass
    
    def _indent(self, level: int) -> str:
        """Helper for indentation in pretty printing."""
        return "  " * level


# ============================================================================
# Type System
# ============================================================================

class Type(ASTNode):
    """Base class for type annotations."""
    pass


@dataclass
class PrimitiveType(Type):
    """Primitive types: int, float, bool, string."""
    name: str  # 'int', 'float', 'bool', 'string'
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return self.name


@dataclass
class TensorType(Type):
    """Tensor type with shape information."""
    element_type: Type
    shape: Optional[List[int]] = None  # None for dynamic shape
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        if self.shape:
            shape_str = f"({', '.join(map(str, self.shape))})"
            return f"Tensor<{self.element_type.pretty_print()}, {shape_str}>"
        return f"Tensor<{self.element_type.pretty_print()}>"


@dataclass
class FunctionType(Type):
    """Function type with parameter and return types."""
    param_types: List[Type]
    return_type: Type
    
    def pretty_print(self, indent: int = 0) -> str:
        params = ', '.join(t.pretty_print() for t in self.param_types)
        return f"({params}) -> {self.return_type.pretty_print()}"


@dataclass
class GenericType(Type):
    """Generic type parameter."""
    name: str
    constraints: List[Type] = None
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        if self.constraints:
            constraints_str = ' + '.join(c.pretty_print() for c in self.constraints)
            return f"{self.name}: {constraints_str}"
        return self.name


# ============================================================================
# Expressions
# ============================================================================

class Expression(ASTNode):
    """Base class for all expressions."""
    pass


@dataclass
class Literal(Expression):
    """Literal values: numbers, strings, booleans."""
    value: Union[int, float, str, bool]
    type_hint: Optional[Type] = None
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


@dataclass
class Identifier(Expression):
    """Variable or function identifier."""
    name: str
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return self.name


@dataclass
class BinaryOp(Expression):
    """Binary operation: +, -, *, /, etc."""
    left: Expression
    operator: str
    right: Expression
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"({self.left.pretty_print()} {self.operator} {self.right.pretty_print()})"


@dataclass
class UnaryOp(Expression):
    """Unary operation: -, !, ~."""
    operator: str
    operand: Expression
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"{self.operator}{self.operand.pretty_print()}"


@dataclass
class FunctionCall(Expression):
    """Function call expression."""
    function: Expression
    arguments: List[Expression]
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        args = ', '.join(arg.pretty_print() for arg in self.arguments)
        return f"{self.function.pretty_print()}({args})"


@dataclass
class MemberAccess(Expression):
    """Member access: obj.member."""
    object: Expression
    member: str
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"{self.object.pretty_print()}.{self.member}"


@dataclass
class IndexAccess(Expression):
    """Array/tensor indexing: arr[index]."""
    object: Expression
    index: Expression
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"{self.object.pretty_print()}[{self.index.pretty_print()}]"


@dataclass
class TensorLiteral(Expression):
    """Tensor literal: [1, 2, 3] or [[1, 2], [3, 4]]."""
    elements: List[Expression]
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        elements_str = ', '.join(elem.pretty_print() for elem in self.elements)
        return f"[{elements_str}]"


@dataclass
class Assignment(Expression):
    """Assignment expression: x = value."""
    target: Expression
    value: Expression
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"{self.target.pretty_print()} = {self.value.pretty_print()}"


# ============================================================================
# Statements
# ============================================================================

class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass
class ExpressionStatement(Statement):
    """Expression used as a statement."""
    expression: Expression
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return self._indent(indent) + self.expression.pretty_print()


@dataclass
class Block(Statement):
    """Block of statements: { ... }."""
    statements: List[Statement]
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        lines = [self._indent(indent) + "{"]
        for stmt in self.statements:
            lines.append(stmt.pretty_print(indent + 1))
        lines.append(self._indent(indent) + "}")
        return '\n'.join(lines)


@dataclass
class VariableDeclaration(Statement):
    """Variable declaration: let x: Type = value."""
    name: str
    type_annotation: Optional[Type] = None
    initializer: Optional[Expression] = None
    is_mutable: bool = False
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        mut_str = "mut " if self.is_mutable else ""
        type_str = f": {self.type_annotation.pretty_print()}" if self.type_annotation else ""
        init_str = f" = {self.initializer.pretty_print()}" if self.initializer else ""
        return f"{self._indent(indent)}let {mut_str}{self.name}{type_str}{init_str}"


@dataclass
class IfStatement(Statement):
    """If statement with optional else clause."""
    condition: Expression
    then_branch: Statement
    else_branch: Optional[Statement] = None
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        lines = [f"{self._indent(indent)}if {self.condition.pretty_print()}"]
        lines.append(self.then_branch.pretty_print(indent))
        if self.else_branch:
            lines.append(f"{self._indent(indent)}else")
            lines.append(self.else_branch.pretty_print(indent))
        return '\n'.join(lines)


@dataclass
class WhileStatement(Statement):
    """While loop statement."""
    condition: Expression
    body: Statement
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        lines = [f"{self._indent(indent)}while {self.condition.pretty_print()}"]
        lines.append(self.body.pretty_print(indent))
        return '\n'.join(lines)


@dataclass
class ForStatement(Statement):
    """For loop statement."""
    variable: str
    iterable: Expression
    body: Statement
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        lines = [f"{self._indent(indent)}for {self.variable} in {self.iterable.pretty_print()}"]
        lines.append(self.body.pretty_print(indent))
        return '\n'.join(lines)


@dataclass
class ReturnStatement(Statement):
    """Return statement with optional value."""
    value: Optional[Expression] = None
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        if self.value:
            return f"{self._indent(indent)}return {self.value.pretty_print()}"
        return f"{self._indent(indent)}return"


@dataclass
class BreakStatement(Statement):
    """Break statement for loops."""
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}break"


@dataclass
class ContinueStatement(Statement):
    """Continue statement for loops."""
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}continue"


# ============================================================================
# Function and Struct Definitions
# ============================================================================

@dataclass
class Parameter:
    """Function parameter definition."""
    name: str
    type_annotation: Type
    default_value: Optional[Expression] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        default_str = f" = {self.default_value.pretty_print()}" if self.default_value else ""
        return f"{self.name}: {self.type_annotation.pretty_print()}{default_str}"


@dataclass
class FunctionDeclaration(Statement):
    """Function declaration with body."""
    name: str
    parameters: List[Parameter]
    return_type: Optional[Type]
    body: Statement
    generic_params: List[GenericType] = None
    is_gpu: bool = False
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        gpu_str = "@gpu " if self.is_gpu else ""
        generic_str = ""
        if self.generic_params:
            generic_names = ', '.join(g.pretty_print() for g in self.generic_params)
            generic_str = f"<{generic_names}>"
        
        params = ', '.join(p.pretty_print() for p in self.parameters)
        return_str = f" -> {self.return_type.pretty_print()}" if self.return_type else ""
        
        lines = [f"{self._indent(indent)}{gpu_str}func {self.name}{generic_str}({params}){return_str}"]
        lines.append(self.body.pretty_print(indent))
        return '\n'.join(lines)


@dataclass
class StructField:
    """Struct field definition."""
    name: str
    type_annotation: Type
    default_value: Optional[Expression] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        default_str = f" = {self.default_value.pretty_print()}" if self.default_value else ""
        return f"{self.name}: {self.type_annotation.pretty_print()}{default_str}"


@dataclass
class StructDeclaration(Statement):
    """Struct declaration."""
    name: str
    fields: List[StructField]
    generic_params: List[GenericType] = None
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        generic_str = ""
        if self.generic_params:
            generic_names = ', '.join(g.pretty_print() for g in self.generic_params)
            generic_str = f"<{generic_names}>"
        
        lines = [f"{self._indent(indent)}struct {self.name}{generic_str} {{"]
        for field in self.fields:
            lines.append(f"{self._indent(indent + 1)}{field.pretty_print()}")
        lines.append(f"{self._indent(indent)}}}")
        return '\n'.join(lines)


# ============================================================================
# Neural Network Specific AST Nodes
# ============================================================================

@dataclass
class LayerDefinition(Expression):
    """Neural network layer definition."""
    layer_type: str  # 'dense', 'conv2d', 'lstm', etc.
    parameters: Dict[str, Expression]
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        params = ', '.join(f"{k}={v.pretty_print()}" for k, v in self.parameters.items())
        return f"{self.layer_type}({params})"


@dataclass
class ModelDefinition(Expression):
    """Neural network model definition."""
    name: str
    layers: List[LayerDefinition]
    generics: Optional[List[Type]] = None
    location: Optional[SourceLocation] = None
    
    def pretty_print(self, indent: int = 0) -> str:
        generic_str = ""
        if self.generics:
            generic_types = ', '.join(g.pretty_print() for g in self.generics)
            generic_str = f"<{generic_types}>"
        
        lines = [f"{self._indent(indent)}{self.name}{generic_str} {{"]
        for layer in self.layers:
            lines.append(f"{self._indent(indent + 1)}{layer.pretty_print()}")
        lines.append(f"{self._indent(indent)}}}")
        return '\n'.join(lines)


# ============================================================================
# Program Structure
# ============================================================================

@dataclass
class ImportStatement(Statement):
    """Import statement: import module."""
    module_path: str
    alias: Optional[str] = None
    items: Optional[List[str]] = None  # For selective imports
    
    def pretty_print(self, indent: int = 0) -> str:
        if self.items:
            items_str = ', '.join(self.items)
            return f"{self._indent(indent)}import {{{items_str}}} from {self.module_path}"
        elif self.alias:
            return f"{self._indent(indent)}import {self.module_path} as {self.alias}"
        else:
            return f"{self._indent(indent)}import {self.module_path}"


@dataclass
class Program(ASTNode):
    """Root node representing a complete NEURO program."""
    statements: List[Statement]
    
    def pretty_print(self, indent: int = 0) -> str:
        lines = []
        for stmt in self.statements:
            lines.append(stmt.pretty_print(indent))
            lines.append("")  # Add blank line between top-level statements
        return '\n'.join(lines).rstrip()


# ============================================================================
# Pattern Matching (Future Extension)
# ============================================================================

@dataclass
class Pattern(ASTNode):
    """Base class for pattern matching patterns."""
    pass


@dataclass
class LiteralPattern(Pattern):
    """Pattern that matches a literal value."""
    value: Union[int, float, str, bool]
    
    def pretty_print(self, indent: int = 0) -> str:
        return str(self.value)


@dataclass
class IdentifierPattern(Pattern):
    """Pattern that binds to an identifier."""
    name: str
    
    def pretty_print(self, indent: int = 0) -> str:
        return self.name


@dataclass
class MatchCase:
    """A single case in a match expression."""
    pattern: Pattern
    guard: Optional[Expression] = None
    body: Statement = None
    
    def pretty_print(self, indent: int = 0) -> str:
        guard_str = f" if {self.guard.pretty_print()}" if self.guard else ""
        lines = [f"{self._indent(indent)}{self.pattern.pretty_print()}{guard_str} =>"]
        lines.append(self.body.pretty_print(indent + 1))
        return '\n'.join(lines)


@dataclass
class MatchExpression(Expression):
    """Pattern matching expression."""
    expression: Expression
    cases: List[MatchCase]
    
    def pretty_print(self, indent: int = 0) -> str:
        lines = [f"{self._indent(indent)}match {self.expression.pretty_print()} {{"]
        for case in self.cases:
            lines.append(case.pretty_print(indent + 1))
        lines.append(f"{self._indent(indent)}}}")
        return '\n'.join(lines) 