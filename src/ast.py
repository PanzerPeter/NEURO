"""
NEURO Abstract Syntax Tree Nodes

This module defines the AST node classes for the NEURO language.
These nodes represent the hierarchical structure of NEURO programs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type

# Base node class
class ASTNode:
    """Base class for all AST nodes"""
    def __init__(self, line: int = 0, column: int = 0):
        self.line = line
        self.column = column
        self._children = []
        self._parent = None

    def add_child(self, child: 'ASTNode') -> None:
        """Add a child node and set its parent."""
        if child is not None:
            self._children.append(child)
            child._parent = self

    def get_children(self) -> List['ASTNode']:
        """Get all child nodes."""
        return self._children

    def get_parent(self) -> Optional['ASTNode']:
        """Get the parent node."""
        return self._parent

    def get_ancestors(self) -> List['ASTNode']:
        """Get all ancestor nodes in order from parent to root."""
        ancestors = []
        current = self._parent
        while current is not None:
            ancestors.append(current)
            current = current._parent
        return ancestors

    def get_root(self) -> 'ASTNode':
        """Get the root node of the AST."""
        current = self
        while current._parent is not None:
            current = current._parent
        return current

    def get_siblings(self) -> List['ASTNode']:
        """Get all sibling nodes (excluding self)."""
        if self._parent is None:
            return []
        return [child for child in self._parent._children if child is not self]

    def accept(self, visitor: 'NodeVisitor') -> Any:
        """Accept a visitor."""
        method_name = f'visit_{self.__class__.__name__}'
        method = getattr(visitor, method_name, visitor.generic_visit)
        return method(self)

# Program structure
@dataclass
class Program(ASTNode):
    """Root node of a NEURO program"""
    statements: List[ASTNode]
    line: int = 0
    column: int = 0

    def __init__(self, statements: List[ASTNode], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.statements = statements
        for statement in statements:
            self.add_child(statement)

# Neural Network nodes
@dataclass
class NeuralNetworkNode(ASTNode):
    """Neural network definition node"""
    input_size: NumberNode
    output_size: NumberNode
    layers: List[LayerNode]
    line: int = 0
    column: int = 0

@dataclass
class LayerNode(ASTNode):
    """Neural network layer node"""
    layer_type: str
    parameters: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class ConfigNode(ASTNode):
    """Configuration object node"""
    name: str
    items: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class DecoratorNode(ASTNode):
    """Decorator node"""
    type: str
    function: FunctionNode
    line: int = 0
    column: int = 0

# Expression nodes
class ExpressionNode(ASTNode):
    """Base class for all expression nodes"""
    def __init__(self, value: Any = None, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value

@dataclass
class NumberNode(ExpressionNode):
    """Numeric literal node"""
    value: Union[int, float]
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.value, self.line, self.column)

@dataclass
class StringNode(ExpressionNode):
    """String literal node"""
    value: str
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.value, self.line, self.column)

@dataclass
class BooleanNode(ExpressionNode):
    """Boolean literal node"""
    value: bool
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.value, self.line, self.column)

@dataclass
class NullNode(ExpressionNode):
    """Null literal node"""
    value: None = None
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.value, self.line, self.column)

@dataclass
class IdentifierNode(ExpressionNode):
    """Identifier node"""
    name: str
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.name, self.line, self.column)

@dataclass
class ListNode(ExpressionNode):
    """List expression node"""
    elements: List[ExpressionNode]
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(None, self.line, self.column)
        for element in self.elements:
            self.add_child(element)

@dataclass
class DictNode(ExpressionNode):
    """Dictionary expression node"""
    items: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(None, self.line, self.column)
        for value in self.items.values():
            self.add_child(value)

@dataclass
class BinaryOpNode(ExpressionNode):
    """Binary operation node"""
    operator: str
    left: ExpressionNode
    right: ExpressionNode
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(None, self.line, self.column)
        self.add_child(self.left)
        self.add_child(self.right)

@dataclass
class UnaryOpNode(ExpressionNode):
    """Unary operation node"""
    operator: str
    operand: ExpressionNode
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(None, self.line, self.column)
        self.add_child(self.operand)

@dataclass
class CallNode(ExpressionNode):
    """Function call node"""
    func: ExpressionNode
    args: List[ExpressionNode]
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(None, self.line, self.column)
        self.add_child(self.func)
        for arg in self.args:
            self.add_child(arg)

# Statement nodes
@dataclass
class FunctionNode(ASTNode):
    """Function definition node"""
    name: str
    params: List[IdentifierNode]
    body: List[ASTNode]
    decorators: Optional[List[IdentifierNode]] = None
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)
        for param in self.params:
            self.add_child(param)
        for stmt in self.body:
            self.add_child(stmt)
        if self.decorators:
            for decorator in self.decorators:
                self.add_child(decorator)

@dataclass
class ReturnNode(ASTNode):
    """Return statement node"""
    value: Optional[ExpressionNode] = None
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)
        if self.value:
            self.add_child(self.value)

@dataclass
class AssignmentNode(ASTNode):
    """Assignment statement node"""
    target: IdentifierNode
    value: ExpressionNode
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)
        self.add_child(self.target)
        self.add_child(self.value)

@dataclass
class IfNode(ASTNode):
    """If statement node"""
    condition: ExpressionNode
    then_body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)
        self.add_child(self.condition)
        for stmt in self.then_body:
            self.add_child(stmt)
        if self.else_body:
            for stmt in self.else_body:
                self.add_child(stmt)

@dataclass
class ForNode(ASTNode):
    """For loop node"""
    iterator: IdentifierNode
    iterable: ExpressionNode
    body: List[ASTNode]
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)
        self.add_child(self.iterator)
        self.add_child(self.iterable)
        for stmt in self.body:
            self.add_child(stmt)

@dataclass
class ParameterNode(ASTNode):
    """Parameter node for function and method parameters"""
    name: str
    value: ExpressionNode
    line: int = 0
    column: int = 0

@dataclass
class BreakNode(ASTNode):
    """Break statement node"""
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)

@dataclass
class ContinueNode(ASTNode):
    """Continue statement node"""
    line: int = 0
    column: int = 0
    
    def __post_init__(self):
        super().__init__(self.line, self.column)

@dataclass
class MethodChainNode(ASTNode):
    """Method chain node for fluent interfaces"""
    target: IdentifierNode
    calls: List[CallNode]
    line: int = 0
    column: int = 0

@dataclass
class CustomLayerNode(ASTNode):
    """Custom layer definition node"""
    name: str
    parameters: List[ParameterNode]
    body: List[ASTNode]
    line: int = 0
    column: int = 0

@dataclass
class BranchNode(ASTNode):
    """Branch definition node for parallel processing"""
    name: str
    parameters: List[ParameterNode]
    body: List[ASTNode]
    line: int = 0
    column: int = 0

@dataclass
class PretrainedNode(ASTNode):
    """Pretrained model loading node"""
    name: str
    source: str
    line: int = 0
    column: int = 0

@dataclass
class LossNode(ASTNode):
    """Loss function definition node"""
    type: str
    parameters: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class OptimizerNode(ASTNode):
    """Optimizer definition node"""
    type: str
    parameters: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class TrainNode(ASTNode):
    """Training operation node"""
    model: IdentifierNode
    parameters: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class EvaluateNode(ASTNode):
    """Model evaluation node"""
    model: IdentifierNode
    parameters: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class PredictNode(ASTNode):
    """Prediction operation node"""
    model: IdentifierNode
    parameters: Dict[str, ExpressionNode]
    line: int = 0
    column: int = 0

@dataclass
class PrintNode(ASTNode):
    """Print statement node"""
    expression: ExpressionNode
    line: int = 0
    column: int = 0

@dataclass
class ImportNode(ASTNode):
    """Import statement node"""
    module: str
    alias: Optional[str] = None
    line: int = 0
    column: int = 0

# Node visitor base class
class NodeVisitor:
    """Base class for AST node visitors"""
    
    def visit(self, node: Any) -> Any:
        """Visit a node"""
        method_name = f'visit_{node.__class__.__name__}'
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node: Any) -> Any:
        """Default visitor method"""
        raise NotImplementedError(
            f"No visit method for {node.__class__.__name__}"
        )

    def visit_children(self, node: ASTNode) -> List[Any]:
        """Visit all children of a node and return their results."""
        return [self.visit(child) for child in node.get_children()]

    def visit_if_exists(self, node: Optional[ASTNode]) -> Optional[Any]:
        """Visit a node if it exists, otherwise return None."""
        return self.visit(node) if node is not None else None