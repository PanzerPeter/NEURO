"""
Parsing Slice
Converts tokens into an Abstract Syntax Tree for the NEURO language.
"""

from typing import List, Optional, Union, Any
from .lexer import Token, TokenType
from .ast_nodes import *
from .errors import ErrorReporter, NeuroSyntaxError, SourceLocation


class NeuroParser:
    """Parser for the NEURO programming language."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.error_reporter = ErrorReporter()
    
    def current_token(self) -> Token:
        """Get the current token."""
        if self.current >= len(self.tokens):
            return self.tokens[-1]  # Return EOF token
        return self.tokens[self.current]
    
    def peek_token(self, offset: int = 1) -> Token:
        """Peek at a token ahead."""
        pos = self.current + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF token
        return self.tokens[pos]
    
    def advance(self) -> Token:
        """Move to the next token and return the current one."""
        token = self.current_token()
        if self.current < len(self.tokens) - 1:
            self.current += 1
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        return self.current_token().type in token_types
    
    def check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        return self.current_token().type == token_type
    
    def consume(self, token_type: TokenType, message: str = "") -> Token:
        """Consume a token of given type or report error."""
        if self.check(token_type):
            return self.advance()
        
        current = self.current_token()
        if not message:
            message = f"Expected {token_type.name}"
        
        self.error_reporter.syntax_error(
            message,
            current.location,
            expected=token_type.name,
            found=current.type.name
        )
        return current
    
    def skip_newlines(self) -> None:
        """Skip newline tokens."""
        while self.match(TokenType.NEWLINE, TokenType.COMMENT):
            self.advance()
    
    def synchronize(self) -> None:
        """Synchronize after a parse error."""
        self.advance()
        
        while not self.check(TokenType.EOF):
            if self.tokens[self.current - 1].type == TokenType.NEWLINE:
                return
            
            if self.match(TokenType.FUNC, TokenType.STRUCT, TokenType.LET,
                         TokenType.IF, TokenType.FOR, TokenType.RETURN):
                return
            
            self.advance()
    
    # =========================================================================
    # Main parsing methods
    # =========================================================================
    
    def parse(self) -> Program:
        """Parse the tokens into a program AST."""
        declarations = []
        statements = []
        
        self.skip_newlines()
        
        while not self.check(TokenType.EOF):
            try:
                if self.match(TokenType.FUNC):
                    declarations.append(self.parse_function_declaration())
                elif self.match(TokenType.STRUCT):
                    declarations.append(self.parse_struct_declaration())
                elif self.match(TokenType.LET):
                    # Global variable declaration
                    stmt = self.parse_variable_declaration()
                    statements.append(stmt)
                else:
                    # Top-level expression or statement
                    stmt = self.parse_statement()
                    if stmt:
                        statements.append(stmt)
                
                self.skip_newlines()
                
            except NeuroSyntaxError:
                self.synchronize()
        
        if self.error_reporter.has_errors():
            self.error_reporter.print_errors()
            raise NeuroSyntaxError("Parsing failed")
        
        return Program(declarations, statements)
    
    # =========================================================================
    # Declaration parsing
    # =========================================================================
    
    def parse_function_declaration(self) -> FunctionDeclaration:
        """Parse a function declaration."""
        location = self.current_token().location
        self.consume(TokenType.FUNC)
        
        name = self.consume(TokenType.IDENTIFIER).value
        
        # Parse optional type parameters
        type_params = None
        if self.match(TokenType.LESS_THAN):
            self.advance()
            type_params = []
            
            type_params.append(self.consume(TokenType.IDENTIFIER).value)
            while self.match(TokenType.COMMA):
                self.advance()
                type_params.append(self.consume(TokenType.IDENTIFIER).value)
            
            self.consume(TokenType.GREATER_THAN)
        
        # Parse parameters
        self.consume(TokenType.LPAREN)
        parameters = []
        
        if not self.match(TokenType.RPAREN):
            parameters.append(self.parse_parameter())
            while self.match(TokenType.COMMA):
                self.advance()
                parameters.append(self.parse_parameter())
        
        self.consume(TokenType.RPAREN)
        
        # Parse optional return type
        return_type = None
        if self.match(TokenType.ARROW):
            self.advance()
            return_type = self.parse_type()
        
        # Parse body
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        body = []
        while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        
        return FunctionDeclaration(
            name=name,
            type_params=type_params,
            parameters=parameters,
            return_type=return_type,
            body=body,
            is_generic=type_params is not None,
            location=location
        )
    
    def parse_parameter(self) -> Parameter:
        """Parse a function parameter."""
        name = self.consume(TokenType.IDENTIFIER).value
        
        type_annotation = None
        if self.match(TokenType.COLON):
            self.advance()
            type_annotation = self.parse_type()
        
        default_value = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            default_value = self.parse_expression()
        
        return Parameter(name, type_annotation, default_value)
    
    def parse_struct_declaration(self) -> StructDeclaration:
        """Parse a struct declaration."""
        location = self.current_token().location
        self.consume(TokenType.STRUCT)
        
        name = self.consume(TokenType.IDENTIFIER).value
        
        # Parse optional type parameters
        type_params = None
        if self.match(TokenType.LESS_THAN):
            self.advance()
            type_params = []
            
            type_params.append(self.consume(TokenType.IDENTIFIER).value)
            while self.match(TokenType.COMMA):
                self.advance()
                type_params.append(self.consume(TokenType.IDENTIFIER).value)
            
            self.consume(TokenType.GREATER_THAN)
        
        # Parse fields
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        fields = []
        while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
            field_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.COLON)
            field_type = self.parse_type()
            
            fields.append(StructField(field_name, field_type))
            
            if self.match(TokenType.COMMA):
                self.advance()
            
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        
        return StructDeclaration(
            name=name,
            type_params=type_params,
            fields=fields,
            is_generic=type_params is not None,
            location=location
        )
    
    # =========================================================================
    # Statement parsing
    # =========================================================================
    
    def parse_statement(self) -> Optional[Statement]:
        """Parse a statement."""
        if self.match(TokenType.LET):
            return self.parse_variable_declaration()
        if self.match(TokenType.IF):
            return self.parse_if_statement()
        if self.match(TokenType.FOR):
            return self.parse_for_statement()
        if self.match(TokenType.RETURN):
            return self.parse_return_statement()
        if self.match(TokenType.LBRACE):
            return self.parse_block_statement()
        
        # Fallback to expression statement, which can include assignment
        expr = self.parse_expression()
        
        if self.match(TokenType.ASSIGN):
            # It's an assignment statement
            self.advance()
            value = self.parse_expression()
            
            # The target must be a valid l-value (identifier, member access, index)
            if isinstance(expr, (IdentifierExpression, MemberExpression, IndexExpression)):
                return AssignmentStatement(expr, value, expr.location)
            else:
                self.error_reporter.syntax_error("Invalid assignment target", expr.location)
                return None
        
        return ExpressionStatement(expr, expr.location)
    
    def parse_variable_declaration(self) -> VariableDeclaration:
        """Parse a variable declaration."""
        location = self.current_token().location
        self.consume(TokenType.LET)
        
        name = self.consume(TokenType.IDENTIFIER).value
        
        type_annotation = None
        if self.match(TokenType.COLON):
            self.advance()
            type_annotation = self.parse_type()
        
        initializer = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            initializer = self.parse_expression()
        
        return VariableDeclaration(name, type_annotation, initializer, True, location)
    
    def parse_if_statement(self) -> IfStatement:
        """Parse an if statement."""
        location = self.current_token().location
        self.consume(TokenType.IF)
        
        condition = self.parse_expression()
        
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        then_body = []
        while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                then_body.append(stmt)
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        
        else_body = None
        if self.match(TokenType.ELSE):
            self.advance()
            
            if self.match(TokenType.LBRACE):
                self.consume(TokenType.LBRACE)
                self.skip_newlines()
                
                else_body = []
                while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
                    stmt = self.parse_statement()
                    if stmt:
                        else_body.append(stmt)
                    self.skip_newlines()
                
                self.consume(TokenType.RBRACE)
            elif self.match(TokenType.IF):
                # Handle "else if"
                else_if = self.parse_if_statement()
                else_body = [else_if]
        
        return IfStatement(condition, then_body, else_body, location)
    
    def parse_for_statement(self) -> ForStatement:
        """Parse a for statement."""
        location = self.current_token().location
        self.consume(TokenType.FOR)
        
        variable = self.consume(TokenType.IDENTIFIER).value
        
        self.consume(TokenType.IN)
        
        iterable = self.parse_expression()
        
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        body = []
        while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        
        return ForStatement(variable, iterable, body, location)
    
    def parse_return_statement(self) -> ReturnStatement:
        """Parse a return statement."""
        location = self.current_token().location
        self.consume(TokenType.RETURN)
        
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.RBRACE, TokenType.EOF):
            value = self.parse_expression()
        
        return ReturnStatement(value, location)
    
    def parse_block_statement(self) -> BlockStatement:
        """Parse a block statement."""
        location = self.current_token().location
        self.consume(TokenType.LBRACE)
        self.skip_newlines()
        
        statements = []
        while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE)
        
        return BlockStatement(statements, location)
    
    # =========================================================================
    # Expression parsing (precedence climbing)
    # =========================================================================
    
    def parse_expression(self) -> Expression:
        """Parse an expression."""
        return self.parse_or_expression()
    
    def parse_or_expression(self) -> Expression:
        """Parse logical OR expression."""
        expr = self.parse_and_expression()
        
        while self.match(TokenType.OR):
            operator = self.advance().value
            right = self.parse_and_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_and_expression(self) -> Expression:
        """Parse logical AND expression."""
        expr = self.parse_equality_expression()
        
        while self.match(TokenType.AND):
            operator = self.advance().value
            right = self.parse_equality_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_equality_expression(self) -> Expression:
        """Parse equality expression."""
        expr = self.parse_comparison_expression()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.advance().value
            right = self.parse_comparison_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_comparison_expression(self) -> Expression:
        """Parse comparison expression."""
        expr = self.parse_term_expression()
        
        while self.match(TokenType.LESS_THAN, TokenType.LESS_EQUAL,
                         TokenType.GREATER_THAN, TokenType.GREATER_EQUAL):
            operator = self.advance().value
            right = self.parse_term_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_term_expression(self) -> Expression:
        """Parse addition/subtraction expression."""
        expr = self.parse_factor_expression()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.advance().value
            right = self.parse_factor_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_factor_expression(self) -> Expression:
        """Parse multiplication/division expression."""
        expr = self.parse_matrix_expression()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.advance().value
            right = self.parse_matrix_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_matrix_expression(self) -> Expression:
        """Parse matrix multiplication expression."""
        expr = self.parse_unary_expression()
        
        while self.match(TokenType.MATRIX_MULT):
            operator = self.advance().value
            right = self.parse_unary_expression()
            expr = BinaryExpression(expr, operator, right, expr.location)
        
        return expr
    
    def parse_unary_expression(self) -> Expression:
        """Parse unary expression."""
        if self.match(TokenType.NOT, TokenType.MINUS):
            operator = self.advance().value
            operand = self.parse_unary_expression()
            return UnaryExpression(operator, operand, operand.location)
        
        return self.parse_postfix_expression()
    
    def parse_postfix_expression(self) -> Expression:
        """Parse postfix expression (calls, member access, indexing)."""
        expr = self.parse_primary_expression()
        
        while True:
            if self.match(TokenType.LESS_THAN):
                # Ambiguity: could be a generic call `func<T>()` or a comparison `a < b`.
                # We need to look ahead to resolve this.
                checkpoint = self.current
                
                # Look for a matching `>` followed by a `(`.
                is_generic_call = False
                balance = 1
                temp_pos = self.current + 1
                while temp_pos < len(self.tokens):
                    if self.tokens[temp_pos].type == TokenType.LESS_THAN:
                        balance += 1
                    elif self.tokens[temp_pos].type == TokenType.GREATER_THAN:
                        balance -= 1
                        if balance == 0:
                            # Found matching '>', check for '('
                            if temp_pos + 1 < len(self.tokens) and \
                               self.tokens[temp_pos + 1].type == TokenType.LPAREN:
                                is_generic_call = True
                            break
                    temp_pos += 1

                if is_generic_call:
                    try:
                        self.advance() # Consume '<'
                        
                        type_args = []
                        if not self.check(TokenType.GREATER_THAN):
                            type_args.append(self.parse_type())
                            while self.match(TokenType.COMMA):
                                self.advance()
                                type_args.append(self.parse_type())
                        
                        self.consume(TokenType.GREATER_THAN)
                        self.consume(TokenType.LPAREN)
                        
                        arguments = []
                        if not self.match(TokenType.RPAREN):
                            arguments.append(self.parse_expression())
                            while self.match(TokenType.COMMA):
                                self.advance()
                                arguments.append(self.parse_expression())
                        
                        self.consume(TokenType.RPAREN)
                        expr = CallExpression(expr, arguments, type_args, expr.location)
                        continue
                    except NeuroSyntaxError:
                        # Backtrack if parsing fails
                        self.current = checkpoint
                        # Fall through to treat as binary operator
                
            if self.match(TokenType.LPAREN):
                # Function call
                self.advance()
                arguments = []
                
                if not self.match(TokenType.RPAREN):
                    arguments.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        self.advance()
                        arguments.append(self.parse_expression())
                
                self.consume(TokenType.RPAREN)
                expr = CallExpression(expr, arguments, None, expr.location)
                
            elif self.match(TokenType.DOT):
                # Member access
                self.advance()
                member = self.consume(TokenType.IDENTIFIER).value
                expr = MemberExpression(expr, member, expr.location)
                
            elif self.match(TokenType.LBRACKET):
                # Array indexing
                self.advance()
                index = self.parse_expression()
                self.consume(TokenType.RBRACKET)
                expr = IndexExpression(expr, index, expr.location)
                
            else:
                break
        
        return expr
    
    def parse_primary_expression(self) -> Expression:
        """Parse primary expression."""
        location = self.current_token().location
        
        # Literals
        if self.match(TokenType.INTEGER):
            value = int(self.advance().value)
            return LiteralExpression(value, None, location)
        
        if self.match(TokenType.FLOAT):
            value = float(self.advance().value)
            return LiteralExpression(value, None, location)
        
        if self.match(TokenType.STRING):
            value = self.advance().value
            return LiteralExpression(value, None, location)
        
        if self.match(TokenType.TRUE, TokenType.FALSE):
            value = self.advance().value == "true"
            return LiteralExpression(value, None, location)
        
        # Neural network literal
        if self.current_token().type == TokenType.IDENTIFIER and self.current_token().value == "NeuralNetwork":
            return self.parse_neural_network_literal(location)
        
        # Identifier
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            
            # Check for struct literal
            if self.match(TokenType.LBRACE):
                return self.parse_struct_literal(name, location)
            
            return IdentifierExpression(name, location)
        
        # Array literal
        if self.match(TokenType.LBRACKET):
            return self.parse_array_literal(location)
        
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.advance()
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN)
            return expr
        
        # Error
        self.error_reporter.syntax_error(
            "Expected expression",
            self.current_token().location
        )
        self.advance()  # Skip the problematic token
        return LiteralExpression(None, None, location)
    
    def parse_array_literal(self, location: SourceLocation) -> ArrayExpression:
        """Parse an array literal."""
        self.consume(TokenType.LBRACKET)
        
        elements = []
        if not self.match(TokenType.RBRACKET):
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                self.advance()
                if self.match(TokenType.RBRACKET):  # Trailing comma
                    break
                elements.append(self.parse_expression())
        
        self.consume(TokenType.RBRACKET)
        return ArrayExpression(elements, location)
    
    def parse_struct_literal(self, type_name: str, location: SourceLocation) -> StructExpression:
        """Parse a struct literal."""
        self.consume(TokenType.LBRACE)
        
        fields = []
        if not self.match(TokenType.RBRACE):
            # Parse field: value pairs
            field_name = self.consume(TokenType.IDENTIFIER).value
            self.consume(TokenType.COLON)
            field_value = self.parse_expression()
            fields.append((field_name, field_value))
            
            while self.match(TokenType.COMMA):
                self.advance()
                if self.match(TokenType.RBRACE):  # Trailing comma
                    break
                field_name = self.consume(TokenType.IDENTIFIER).value
                self.consume(TokenType.COLON)
                field_value = self.parse_expression()
                fields.append((field_name, field_value))
        
        self.consume(TokenType.RBRACE)
        return StructExpression(type_name, fields, location)
    
    def parse_neural_network_literal(self, location: SourceLocation) -> NeuralNetworkExpression:
        """Parse a neural network literal."""
        self.consume(TokenType.IDENTIFIER)  # NeuralNetwork
        
        # Parse type parameters
        self.consume(TokenType.LESS_THAN)
        element_type = self.parse_type()
        self.consume(TokenType.COMMA)
        
        # Parse shape parameters
        self.consume(TokenType.LPAREN)
        shape_params = []
        shape_params.append(int(self.consume(TokenType.INTEGER).value))
        while self.match(TokenType.COMMA):
            self.advance()
            shape_params.append(int(self.consume(TokenType.INTEGER).value))
        self.consume(TokenType.RPAREN)
        self.consume(TokenType.GREATER_THAN)
        
        # Parse layers
        self.consume(TokenType.LBRACE)
        layers = []
        
        while not self.match(TokenType.RBRACE) and not self.check(TokenType.EOF):
            layer = self.parse_expression()
            layers.append(layer)
            
            if self.match(TokenType.COMMA):
                self.advance()
        
        self.consume(TokenType.RBRACE)
        
        return NeuralNetworkExpression(element_type, shape_params, layers, location)
    
    # =========================================================================
    # Type parsing
    # =========================================================================
    
    def parse_type(self) -> Type:
        """Parse a type annotation."""
        location = self.current_token().location
        
        if self.match(TokenType.INT_TYPE, TokenType.FLOAT_TYPE,
                     TokenType.STRING_TYPE, TokenType.BOOL_TYPE):
            name = self.advance().value
            return PrimitiveType(name, location)
        
        if self.match(TokenType.TENSOR):
            return self.parse_tensor_type(location)
        
        if self.match(TokenType.IDENTIFIER):
            name = self.advance().value
            
            # Check for generic type
            if self.match(TokenType.LESS_THAN):
                self.advance()
                type_params = []
                
                type_params.append(self.parse_type())
                while self.match(TokenType.COMMA):
                    self.advance()
                    type_params.append(self.parse_type())
                
                self.consume(TokenType.GREATER_THAN)
                return GenericType(name, type_params, location)
            
            return PrimitiveType(name, location)
        
        self.error_reporter.syntax_error(
            "Expected type",
            self.current_token().location
        )
        return PrimitiveType("unknown", location)
    
    def parse_tensor_type(self, location: SourceLocation) -> TensorType:
        """Parse a tensor type."""
        self.consume(TokenType.TENSOR)
        self.consume(TokenType.LESS_THAN)
        
        element_type = self.parse_type()
        
        shape = None
        if self.match(TokenType.COMMA):
            self.advance()
            self.consume(TokenType.LPAREN)
            
            shape = []
            shape.append(int(self.consume(TokenType.INTEGER).value))
            while self.match(TokenType.COMMA):
                self.advance()
                shape.append(int(self.consume(TokenType.INTEGER).value))
            
            self.consume(TokenType.RPAREN)
        
        self.consume(TokenType.GREATER_THAN)
        
        return TensorType(element_type, shape, location)
    
    def get_error_reporter(self) -> ErrorReporter:
        """Get the error reporter."""
        return self.error_reporter 