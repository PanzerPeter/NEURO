"""
NEURO Parser

Converts tokens from the lexer into an Abstract Syntax Tree (AST).
Implements a recursive descent parser for the NEURO programming language.

Grammar (simplified):
    program := statement*
    statement := function_decl | struct_decl | var_decl | expr_stmt | control_flow
    expression := assignment | logical_or
    assignment := logical_or ('=' assignment)?
    logical_or := logical_and ('||' logical_and)*
    logical_and := equality ('&&' equality)*
    equality := comparison (('==' | '!=') comparison)*
    comparison := term (('<' | '<=' | '>' | '>=') term)*
    term := factor (('+' | '-') factor)*
    factor := unary (('*' | '/' | '%' | '@') unary)*
    unary := ('!' | '-' | '~') unary | postfix
    postfix := primary ('.' IDENTIFIER | '[' expression ']' | '(' arguments? ')')*
    primary := literal | identifier | '(' expression ')' | tensor_literal
"""

from typing import List, Optional, Union, Dict, Any
from .lexer import Token, TokenType
from .ast_nodes import *
from .errors import ParseError, SourceLocation


class NeuroParser:
    """
    Recursive descent parser for the NEURO programming language.
    
    Converts a stream of tokens into an Abstract Syntax Tree (AST)
    that represents the structure of the program.
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.current = 0
    
    def parse(self, tokens: List[Token]) -> Program:
        """
        Parse a list of tokens into a Program AST node.
        
        Args:
            tokens: List of tokens from the lexer
            
        Returns:
            Program AST node
            
        Raises:
            ParseError: If the tokens don't form a valid program
        """
        self.tokens = tokens
        self.current = 0
        self._statements_parsed = 0  # Track number of successfully parsed statements

        statements = []
        
        while not self.is_at_end():
            # Skip newlines at the top level
            if self.check(TokenType.NEWLINE):
                self.advance()
                continue
            
            stmt = self.declaration()
            if stmt:
                statements.append(stmt)
                self._statements_parsed += 1

        return Program(statements)
    
    # ========================================================================
    # Declaration Parsing
    # ========================================================================
    
    def declaration(self) -> Optional[Statement]:
        """Parse a declaration (function, struct, variable) or statement."""
        try:
            if self.match(TokenType.FUNC):
                return self.function_declaration()
            if self.match(TokenType.STRUCT):
                return self.struct_declaration()
            if self.match(TokenType.LET):
                return self.variable_declaration()
            if self.match(TokenType.IMPORT):
                return self.import_statement()
            
            return self.statement()
        except ParseError as e:
            # If this is a simple program and we haven't successfully parsed any statements yet,
            # re-raise the error instead of recovering (for better error reporting in simple cases)
            if len(self.tokens) <= 10 and self._statements_parsed == 0:
                raise e
            
            # Otherwise, synchronize after errors for error recovery
            self.synchronize()
            return None
    
    def function_declaration(self) -> FunctionDeclaration:
        """Parse a function declaration."""
        # Check for @gpu attribute
        is_gpu = False
        if self.previous().type == TokenType.GPU:
            is_gpu = True
            self.consume(TokenType.FUNC, "Expected 'func' after @gpu")
        
        # Ensure we have a function name - this will raise ParseError if missing
        name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        
        # Parse generic parameters
        generic_params = None
        if self.check(TokenType.LESS):
            generic_params = self.generic_parameters()
        
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        
        # Parse parameters
        parameters = []
        if not self.check(TokenType.RPAREN):
            parameters.append(self.parameter())
            while self.match(TokenType.COMMA):
                parameters.append(self.parameter())
        
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        # Parse return type
        return_type = None
        if self.match(TokenType.ARROW):
            return_type = self.type_annotation()
        
        # Parse body - consume opening brace first
        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self.block_statement()
        
        return FunctionDeclaration(
            name=name,
            parameters=parameters,
            return_type=return_type,
            body=body,
            generic_params=generic_params,
            is_gpu=is_gpu,
            location=self.previous().location
        )
    
    def struct_declaration(self) -> StructDeclaration:
        """Parse a struct declaration."""
        name = self.consume(TokenType.IDENTIFIER, "Expected struct name").value
        
        # Parse generic parameters
        generic_params = None
        if self.check(TokenType.LESS):
            generic_params = self.generic_parameters()
        
        self.consume(TokenType.LBRACE, "Expected '{' after struct name")
        
        # Parse fields
        fields = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            fields.append(self.struct_field())
        
        self.consume(TokenType.RBRACE, "Expected '}' after struct fields")
        
        return StructDeclaration(
            name=name,
            fields=fields,
            generic_params=generic_params,
            location=self.previous().location
        )
    
    def variable_declaration(self) -> VariableDeclaration:
        """Parse a variable declaration."""
        is_mutable = self.match(TokenType.MUT)
        name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        # Parse type annotation
        type_annotation = None
        if self.match(TokenType.COLON):
            type_annotation = self.type_annotation()
        
        # Parse initializer
        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.expression()
            # Validate that we actually got an expression
            if initializer is None:
                raise ParseError("Expected expression after '='", self.previous().location)
        
        return VariableDeclaration(
            name=name,
            type_annotation=type_annotation,
            initializer=initializer,
            is_mutable=is_mutable,
            location=self.previous().location
        )
    
    def import_statement(self) -> ImportStatement:
        """Parse an import statement."""
        # Handle different import forms:
        # import module
        # import module as alias
        # import {item1, item2} from module
        
        if self.check(TokenType.LBRACE):
            # Selective import: import {items} from module
            self.advance()  # consume '{'
            items = []
            items.append(self.consume(TokenType.IDENTIFIER, "Expected identifier").value)
            while self.match(TokenType.COMMA):
                items.append(self.consume(TokenType.IDENTIFIER, "Expected identifier").value)
            
            self.consume(TokenType.RBRACE, "Expected '}' after import items")
            self.consume(TokenType.IDENTIFIER, "Expected 'from'")  # TODO: add FROM token
            module_path = self.consume(TokenType.IDENTIFIER, "Expected module path").value
            
            return ImportStatement(module_path=module_path, items=items)
        else:
            # Regular import
            module_path = self.consume(TokenType.IDENTIFIER, "Expected module path").value
            
            alias = None
            if self.match(TokenType.IDENTIFIER):  # 'as'
                if self.previous().value == "as":
                    alias = self.consume(TokenType.IDENTIFIER, "Expected alias").value
                else:
                    raise ParseError("Unexpected token in import", self.previous().location)
            
            return ImportStatement(module_path=module_path, alias=alias)
    
    # ========================================================================
    # Statement Parsing
    # ========================================================================
    
    def statement(self) -> Statement:
        """Parse a statement."""
        if self.match(TokenType.IF):
            return self.if_statement()
        if self.match(TokenType.WHILE):
            return self.while_statement()
        if self.match(TokenType.FOR):
            return self.for_statement()
        if self.match(TokenType.RETURN):
            return self.return_statement()
        if self.match(TokenType.BREAK):
            return BreakStatement(location=self.previous().location)
        if self.match(TokenType.CONTINUE):
            return ContinueStatement(location=self.previous().location)
        if self.match(TokenType.LBRACE):
            return self.block_statement()
        
        return self.expression_statement()
    
    def if_statement(self) -> IfStatement:
        """Parse an if statement."""
        condition = self.expression()
        # Validate that we got a valid condition
        if condition is None:
            raise ParseError("Expected condition after 'if'", self.previous().location)
            
        then_branch = self.statement()
        
        else_branch = None
        if self.match(TokenType.ELSE):
            else_branch = self.statement()
        
        return IfStatement(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
            location=self.previous().location
        )
    
    def while_statement(self) -> WhileStatement:
        """Parse a while statement."""
        condition = self.expression()
        body = self.statement()
        
        return WhileStatement(
            condition=condition,
            body=body,
            location=self.previous().location
        )
    
    def for_statement(self) -> ForStatement:
        """Parse a for statement."""
        variable = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        self.consume(TokenType.IDENTIFIER, "Expected 'in'")  # TODO: add IN token
        iterable = self.expression()
        body = self.statement()
        
        return ForStatement(
            variable=variable,
            iterable=iterable,
            body=body,
            location=self.previous().location
        )
    
    def return_statement(self) -> ReturnStatement:
        """Parse a return statement."""
        value = None
        if not self.check(TokenType.NEWLINE) and not self.check(TokenType.SEMICOLON):
            value = self.expression()
        
        return ReturnStatement(
            value=value,
            location=self.previous().location
        )
    
    def block_statement(self) -> Block:
        """Parse a block statement."""
        statements = []
        
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            
            stmt = self.declaration()
            if stmt:
                statements.append(stmt)
        
        self.consume(TokenType.RBRACE, "Expected '}' after block")
        
        return Block(statements=statements, location=self.previous().location)
    
    def expression_statement(self) -> ExpressionStatement:
        """Parse an expression statement."""
        expr = self.expression()
        return ExpressionStatement(expression=expr, location=expr.location)
    
    # ========================================================================
    # Expression Parsing (Precedence Climbing)
    # ========================================================================
    
    def expression(self) -> Expression:
        """Parse an expression."""
        return self.assignment()
    
    def assignment(self) -> Expression:
        """Parse assignment expression."""
        expr = self.logical_or()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, 
                      TokenType.MINUS_ASSIGN, TokenType.MULTIPLY_ASSIGN, 
                      TokenType.DIVIDE_ASSIGN):
            operator = self.previous().value
            value = self.assignment()
            
            if operator != '=':
                # Convert += to = + for AST simplicity
                binary_op = operator[:-1]  # Remove '='
                value = BinaryOp(expr, binary_op, value, location=expr.location)
            
            return Assignment(target=expr, value=value, location=expr.location)
        
        return expr
    
    def logical_or(self) -> Expression:
        """Parse logical OR expression."""
        expr = self.logical_and()
        
        while self.match(TokenType.OR):
            operator = self.previous().value
            right = self.logical_and()
            expr = BinaryOp(expr, operator, right, location=expr.location)
        
        return expr
    
    def logical_and(self) -> Expression:
        """Parse logical AND expression."""
        expr = self.equality()
        
        while self.match(TokenType.AND):
            operator = self.previous().value
            right = self.equality()
            expr = BinaryOp(expr, operator, right, location=expr.location)
        
        return expr
    
    def equality(self) -> Expression:
        """Parse equality expression."""
        expr = self.comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.previous().value
            right = self.comparison()
            expr = BinaryOp(expr, operator, right, location=expr.location)
        
        return expr
    
    def comparison(self) -> Expression:
        """Parse comparison expression."""
        expr = self.term()
        
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL,
                          TokenType.LESS, TokenType.LESS_EQUAL):
            operator = self.previous().value
            right = self.term()
            expr = BinaryOp(expr, operator, right, location=expr.location)
        
        return expr
    
    def term(self) -> Expression:
        """Parse addition/subtraction expression."""
        expr = self.factor()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous().value
            right = self.factor()
            expr = BinaryOp(expr, operator, right, location=expr.location)
        
        return expr
    
    def factor(self) -> Expression:
        """Parse multiplication/division expression."""
        expr = self.unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, 
                          TokenType.MODULO, TokenType.MATRIX_MUL):
            operator = self.previous().value
            right = self.unary()
            expr = BinaryOp(expr, operator, right, location=expr.location)
        
        return expr
    
    def unary(self) -> Expression:
        """Parse unary expression."""
        if self.match(TokenType.NOT, TokenType.MINUS, TokenType.BIT_NOT):
            operator = self.previous().value
            right = self.unary()
            return UnaryOp(operator, right, location=self.previous().location)
        
        return self.postfix()
    
    def postfix(self) -> Expression:
        """Parse postfix expression (member access, indexing, calls)."""
        expr = self.primary()
        
        while True:
            if self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expected property name").value
                expr = MemberAccess(expr, name, location=expr.location)
            elif self.match(TokenType.LBRACKET):
                index = self.expression()
                self.consume(TokenType.RBRACKET, "Expected ']' after index")
                expr = IndexAccess(expr, index, location=expr.location)
            elif self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            else:
                break
        
        return expr
    
    def primary(self) -> Expression:
        """Parse primary expression."""
        if self.match(TokenType.BOOLEAN):
            value = self.previous().value == "true"
            return Literal(value, location=self.previous().location)
        
        if self.match(TokenType.INTEGER):
            value = int(self.previous().value)
            return Literal(value, location=self.previous().location)
        
        if self.match(TokenType.FLOAT):
            value = float(self.previous().value)
            return Literal(value, location=self.previous().location)
        
        if self.match(TokenType.STRING):
            value = self.previous().value
            return Literal(value, location=self.previous().location)
        
        if self.match(TokenType.IDENTIFIER):
            return Identifier(self.previous().value, location=self.previous().location)
        
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        if self.match(TokenType.LBRACKET):
            return self.tensor_literal()
        
        # Neural network model definitions
        if self.check(TokenType.NEURALNETWORK):
            return self.model_definition()
        
        raise ParseError(
            f"Unexpected token: {self.peek().value}",
            self.peek().location
        )
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def finish_call(self, callee: Expression) -> FunctionCall:
        """Parse function call arguments."""
        arguments = []
        
        if not self.check(TokenType.RPAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
        
        self.consume(TokenType.RPAREN, "Expected ')' after arguments")
        return FunctionCall(callee, arguments, location=callee.location)
    
    def tensor_literal(self) -> TensorLiteral:
        """Parse tensor literal: [1, 2, 3] or [[1, 2], [3, 4]]."""
        elements = []
        
        if not self.check(TokenType.RBRACKET):
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                elements.append(self.expression())
        
        self.consume(TokenType.RBRACKET, "Expected ']' after tensor elements")
        return TensorLiteral(elements, location=self.previous().location)
    
    def model_definition(self) -> ModelDefinition:
        """Parse neural network model definition."""
        self.advance()  # consume NEURALNETWORK
        
        # Parse generics
        generics = None
        if self.match(TokenType.LESS):
            generics = []
            generics.append(self.type_annotation())
            while self.match(TokenType.COMMA):
                generics.append(self.type_annotation())
            self.consume(TokenType.GREATER, "Expected '>' after generics")
        
        name = self.consume(TokenType.IDENTIFIER, "Expected model name").value
        self.consume(TokenType.LBRACE, "Expected '{' after model name")
        
        # Parse layers
        layers = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            layers.append(self.layer_definition())
        
        self.consume(TokenType.RBRACE, "Expected '}' after model layers")
        
        return ModelDefinition(
            name=name,
            layers=layers,
            generics=generics,
            location=self.previous().location
        )
    
    def layer_definition(self) -> LayerDefinition:
        """Parse neural network layer definition."""
        layer_type = self.consume(TokenType.IDENTIFIER, "Expected layer type").value
        self.consume(TokenType.LPAREN, "Expected '(' after layer type")
        
        # Parse layer parameters
        parameters = {}
        if not self.check(TokenType.RPAREN):
            name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
            self.consume(TokenType.ASSIGN, "Expected '=' after parameter name")
            value = self.expression()
            parameters[name] = value
            
            while self.match(TokenType.COMMA):
                name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
                self.consume(TokenType.ASSIGN, "Expected '=' after parameter name")
                value = self.expression()
                parameters[name] = value
        
        self.consume(TokenType.RPAREN, "Expected ')' after layer parameters")
        
        return LayerDefinition(
            layer_type=layer_type,
            parameters=parameters,
            location=self.previous().location
        )
    
    def parameter(self) -> Parameter:
        """Parse function parameter."""
        name = self.consume(TokenType.IDENTIFIER, "Expected parameter name").value
        self.consume(TokenType.COLON, "Expected ':' after parameter name")
        type_annotation = self.type_annotation()
        # type_annotation() will raise ParseError if no valid type is found
        
        default_value = None
        if self.match(TokenType.ASSIGN):
            default_value = self.expression()
        
        return Parameter(name, type_annotation, default_value)
    
    def struct_field(self) -> StructField:
        """Parse struct field."""
        name = self.consume(TokenType.IDENTIFIER, "Expected field name").value
        self.consume(TokenType.COLON, "Expected ':' after field name")
        type_annotation = self.type_annotation()
        
        default_value = None
        if self.match(TokenType.ASSIGN):
            default_value = self.expression()
        
        return StructField(name, type_annotation, default_value)
    
    def type_annotation(self) -> Type:
        """Parse type annotation."""
        if self.match(TokenType.INT):
            return PrimitiveType("int", location=self.previous().location)
        if self.match(TokenType.FLOAT_TYPE):
            return PrimitiveType("float", location=self.previous().location)
        if self.match(TokenType.BOOL):
            return PrimitiveType("bool", location=self.previous().location)
        if self.match(TokenType.STRING_TYPE):
            return PrimitiveType("string", location=self.previous().location)
        
        if self.match(TokenType.TENSOR):
            self.consume(TokenType.LESS, "Expected '<' after 'Tensor'")
            element_type = self.type_annotation()
            
            shape = None
            if self.match(TokenType.COMMA):
                # Parse shape tuple
                self.consume(TokenType.LPAREN, "Expected '(' for shape")
                shape = []
                shape.append(int(self.consume(TokenType.INTEGER, "Expected integer").value))
                while self.match(TokenType.COMMA):
                    shape.append(int(self.consume(TokenType.INTEGER, "Expected integer").value))
                self.consume(TokenType.RPAREN, "Expected ')' after shape")
            
            self.consume(TokenType.GREATER, "Expected '>' after tensor type")
            return TensorType(element_type, shape, location=self.previous().location)
        
        if self.match(TokenType.IDENTIFIER):
            name = self.previous().value
            
            # Check if this is a Tensor type specified as identifier
            if name == "Tensor" and self.check(TokenType.LESS):
                self.advance()  # consume '<'
                element_type = self.type_annotation()
                
                shape = None
                if self.match(TokenType.COMMA):
                    # Parse shape tuple
                    self.consume(TokenType.LPAREN, "Expected '(' for shape")
                    shape = []
                    shape.append(int(self.consume(TokenType.INTEGER, "Expected integer").value))
                    while self.match(TokenType.COMMA):
                        shape.append(int(self.consume(TokenType.INTEGER, "Expected integer").value))
                    self.consume(TokenType.RPAREN, "Expected ')' after shape")
                
                self.consume(TokenType.GREATER, "Expected '>' after tensor type")
                return TensorType(element_type, shape, location=self.previous().location)
            
            return GenericType(name, location=self.previous().location)
        
        # If we reach here, no valid type was found
        raise ParseError(
            f"Expected type annotation, found '{self.peek().value}'",
            self.peek().location
        )
    
    def generic_parameters(self) -> List[GenericType]:
        """Parse generic type parameters."""
        self.consume(TokenType.LESS, "Expected '<'")
        
        params = []
        params.append(GenericType(self.consume(TokenType.IDENTIFIER, "Expected type parameter").value))
        
        while self.match(TokenType.COMMA):
            params.append(GenericType(self.consume(TokenType.IDENTIFIER, "Expected type parameter").value))
        
        self.consume(TokenType.GREATER, "Expected '>' after type parameters")
        return params
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def advance(self) -> Token:
        """Consume current token and return it."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        """Check if we're at the end of tokens."""
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        """Return current token without advancing."""
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        """Return previous token."""
        return self.tokens[self.current - 1]
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error."""
        if self.check(token_type):
            return self.advance()
        
        current_token = self.peek()
        raise ParseError(
            message,
            current_token.location,
            expected=token_type.name,
            found=current_token.type.name
        )
    
    def synchronize(self) -> None:
        """Recover from parse errors by advancing to next statement."""
        self.advance()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.NEWLINE:
                return
            
            if self.peek().type in {
                TokenType.FUNC, TokenType.STRUCT, TokenType.LET,
                TokenType.IF, TokenType.WHILE, TokenType.FOR,
                TokenType.RETURN
            }:
                return
            
            self.advance()
    
    def is_simple_program(self) -> bool:
        """Check if this appears to be a simple single-statement program (for better error reporting)."""
        # If we have fewer than 10 tokens total, treat it as a simple program
        return len(self.tokens) <= 10 