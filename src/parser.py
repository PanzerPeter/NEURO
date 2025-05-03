# NEURO Parser: Converts .nr code into an Abstract Syntax Tree (AST)

from src.lexer import NeuroLexer
import src.neuro_ast as neuro_ast
from src.errors import NeuroSyntaxError
import logging

class NeuroParser:
    def __init__(self, use_real_lexer=False):
        self.lexer = NeuroLexer()
        self.lexer.set_use_real_implementation(use_real_lexer)
        self.tokens = []
        self.current_token_index = 0
        self._use_real_parser = False

    def parse(self, code):
        """Parses the NEURO code string and returns an AST Root (Program node)."""
        # print(f"Parsing code:\n{code[:100]}...") # Debug print
        
        if not self._use_real_parser:
            # Placeholder: Still returning a dummy structure
            print("Using placeholder parser...")
            return self._parse_placeholder()
        else:
            # Use the real lexer
            self.tokens = list(self.lexer.tokenize(code, use_placeholder=False))
            self.current_token_index = 0
            # Start parsing from the program level
            program_node = self._parse_program()
            print("Parsing complete.")
            return program_node
    
    def _parse_placeholder(self):
        """Returns a placeholder AST for compatibility."""
        dummy_layers = [
            neuro_ast.Layer(layer_type='Dense', params={'units': 64, 'activation': 'relu'}),
            neuro_ast.Layer(layer_type='Dense', params={'units': 32, 'activation': 'relu'}),
            neuro_ast.Layer(layer_type='Dense', params={'units': 1, 'activation': 'sigmoid'})
        ]
        dummy_model_def = neuro_ast.ModelDefinition(name='model', params={'input_size': 3, 'output_size': 1}, layers=dummy_layers)
        program_body = [dummy_model_def]
        return neuro_ast.Program(body=program_body)
    
    def set_use_real_parser(self, use_real=True):
        """Sets whether to use the real parser implementation."""
        self._use_real_parser = use_real
        # Also update the lexer to use the real implementation
        self.lexer.set_use_real_implementation(use_real)

    # --- Recursive Descent Parsing Methods ---
    
    def _parse_program(self):
        """ Parses a program: a sequence of statements until EOF."""
        body = []
        while self._current_token().type != 'EOF':
            statement = self._parse_statement()
            body.append(statement)
            # Expect a semicolon after most statements
            # Model definitions end with '}', method calls don't have a separate ';'
            # Assignments are handled within _parse_assignment_statement
            if isinstance(statement, neuro_ast.AssignmentStatement):
                # Check if the assigned value was a model def; if so, no semicolon needed here
                if not isinstance(statement.value, neuro_ast.ModelDefinition):
                    self._expect('SEMICOLON', f"Expected ';' after assignment statement ending with {type(statement.value)}")
            elif isinstance(statement, (neuro_ast.TrainStatement, neuro_ast.EvaluateStatement)):
                # These method calls are statements themselves, expect semicolon
                 self._expect('SEMICOLON', f"Expected ';' after {type(statement).__name__} statement")
            # Add other statement types that need semicolons here
            
        return neuro_ast.Program(body=body)

    def _parse_statement(self):
        """ Parses a single statement (e.g., assignment, model call). """
        # Check for keywords or patterns that identify the statement type
        token = self._current_token()
        
        # Assignment: IDENTIFIER ASSIGN ...
        if token.type == 'IDENTIFIER':
            # Look ahead to see if it's an assignment or a model call
            if self._peek_token().type == 'ASSIGN':
                return self._parse_assignment_statement()
            elif self._peek_token().type == 'DOT': # e.g., model.train(...)
                # Parse the method call as a statement itself
                call_statement = self._parse_method_call_expression() 
                return call_statement
            else:
                raise self._syntax_error(f"Unexpected token '{self._peek_token().value}' after identifier '{token.value}'")
        else:
            # Handle other potential statement types here if needed
            raise self._syntax_error(f"Unexpected token '{token.value}' at start of statement")

    def _parse_assignment_statement(self):
        """ Parses: IDENTIFIER ASSIGN expression """
        target_token = self._expect('IDENTIFIER', "Expected variable name in assignment")
        target = neuro_ast.Identifier(name=target_token.value)
        self._expect('ASSIGN', "Expected '=' in assignment")
        
        # Parse the right-hand side expression/call
        value = self._parse_expression()
        
        # If the value is a ModelDefinition, set its name
        if isinstance(value, neuro_ast.ModelDefinition):
            value.name = target.name
            
        return neuro_ast.AssignmentStatement(target=target, value=value)

    def _parse_expression(self):
        """ Parses an expression (e.g., function call, literal, model def). """
        token = self._current_token()
        
        # Model Definition: NeuralNetwork(...) { ... }
        if token.type == 'IDENTIFIER' and token.value == 'NeuralNetwork':
            return self._parse_model_definition()
        # Loss Definition: Loss(...)
        elif token.type == 'IDENTIFIER' and token.value == 'Loss':
             return self._parse_simple_call('Loss', neuro_ast.LossDefinition)
        # Optimizer Definition: Optimizer(...)
        elif token.type == 'IDENTIFIER' and token.value == 'Optimizer':
            return self._parse_simple_call('Optimizer', neuro_ast.OptimizerDefinition)
        # Evaluate call (RHS of assignment): model.evaluate(...)
        elif token.type == 'IDENTIFIER' and self._peek_token().type == 'DOT' and self._peek_token(2).value == 'evaluate':
             return self._parse_method_call_expression() # Parse as an expression
        # General Function Call: IDENTIFIER(...)
        elif token.type == 'IDENTIFIER' and self._peek_token().type == 'LPAREN':
            return self._parse_general_function_call()
        # Literals
        elif token.type == 'NUMBER':
            self._advance()
            return neuro_ast.NumberLiteral(value=token.value)
        elif token.type == 'STRING':
            self._advance()
            return neuro_ast.StringLiteral(value=token.value)
        else:
            raise self._syntax_error(f"Unexpected start of expression: '{token.value}'")
            
    def _parse_model_definition(self):
        """ Parses: NeuralNetwork(params...) { layers... } """
        self._expect('IDENTIFIER', expected_value='NeuralNetwork')
        params = self._parse_parameters()
        
        self._expect('LBRACE', "Expected '{' to start model body")
        layers = []
        while self._current_token().type != 'RBRACE' and self._current_token().type != 'EOF':
            layer = self._parse_layer()
            layers.append(layer)
            self._expect('SEMICOLON', "Expected ';' after layer definition")
            
        self._expect('RBRACE', "Expected '}' to end model body")
        
        # Note: The model name is handled by the assignment statement that calls this
        # This method just returns the structure to be assigned.
        return neuro_ast.ModelDefinition(name=None, params=params, layers=layers) # Name set by caller

    def _parse_layer(self):
        """ Parses: LayerType(params...) """
        layer_type_token = self._expect('IDENTIFIER', "Expected layer type (e.g., Dense)")
        layer_type = layer_type_token.value
        params = self._parse_parameters()
        return neuro_ast.Layer(layer_type=layer_type, params=params)

    def _parse_simple_call(self, expected_name, node_class):
        """ Parses: FunctionName(params...) """
        self._expect('IDENTIFIER', expected_value=expected_name)
        params = self._parse_parameters()
        return node_class(params=params)
        
    def _parse_method_call_expression(self):
        """ Parses object.method(...) as an expression or statement """
        object_name_token = self._expect('IDENTIFIER', "Expected object name")
        object_name = object_name_token.value
        self._expect('DOT', "Expected '.' for method call")
        method_name_token = self._expect('IDENTIFIER', "Expected method name")
        method_name = method_name_token.value
        
        self._expect('LPAREN', "Expected '(' to start method arguments")
        
        # Arguments differ for train vs evaluate
        if method_name == 'train':
            args = self._parse_key_value_pairs()
            # Separate data source if present as first positional arg (needs refinement based on exact grammar)
            data_source_value = args.pop('data', None) # Simple assumption
            if not isinstance(data_source_value, neuro_ast.Identifier):
                # TODO: Improve error location reporting
                raise self._syntax_error("Expected 'data' parameter to be an identifier")
            params = args # Remaining key-value pairs are params
            self._expect('RPAREN', "Expected ')' after train arguments")
            return neuro_ast.TrainStatement(model_name=object_name, data_source=data_source_value, params=params)
        elif method_name == 'evaluate':
            # Evaluate might just take a single data source identifier
            data_source_token = self._expect('IDENTIFIER', "Expected data source identifier for evaluate")
            data_source = neuro_ast.Identifier(name=data_source_token.value)
            self._expect('RPAREN', "Expected ')' after evaluate argument")
            return neuro_ast.EvaluateStatement(model_name=object_name, data_source=data_source)
        else:
            raise self._syntax_error(f"Unknown method name: '{method_name}'")

    def _parse_parameters(self):
        """ Parses: (key=value, key=value, ...) """
        self._expect('LPAREN', "Expected '(' to start parameters")
        params = self._parse_key_value_pairs()
        self._expect('RPAREN', "Expected ')' to end parameters")
        return params
        
    def _parse_key_value_pairs(self):
        """ Parses key=value pairs inside parentheses, separated by commas."""
        params = {}
        if self._current_token().type != 'RPAREN':
            while True:
                key_token = self._expect('IDENTIFIER', "Expected parameter name")
                key = key_token.value
                self._expect('ASSIGN', f"Expected '=' after parameter name '{key}'")
                value_token = self._current_token()
                if value_token.type == 'STRING':
                    value = value_token.value
                    self._advance()
                elif value_token.type == 'NUMBER':
                    value = value_token.value
                    self._advance()
                elif value_token.type == 'IDENTIFIER': # For data source names
                    value = neuro_ast.Identifier(name=value_token.value)
                    self._advance()
                else:
                    raise self._syntax_error("Expected STRING, NUMBER, or IDENTIFIER as parameter value")
                
                params[key] = value
                
                if self._current_token().type == 'COMMA':
                    self._advance()
                elif self._current_token().type == 'RPAREN':
                    break
                else:
                    raise self._syntax_error("Expected ',' or ')' after parameter value")
        return params

    def _parse_general_function_call(self):
        """ Parses: function_name(arg1, arg2, ...) """
        func_name_token = self._expect('IDENTIFIER', "Expected function name")
        func_name = func_name_token.value
        self._expect('LPAREN', "Expected '(' to start function arguments")
        
        args = []
        if self._current_token().type != 'RPAREN':
            while True:
                # Arguments are expressions themselves
                arg_expr = self._parse_expression()
                args.append(arg_expr)
                
                if self._current_token().type == 'COMMA':
                    self._advance()
                elif self._current_token().type == 'RPAREN':
                    break
                else:
                    raise self._syntax_error("Expected ',' or ')' after function argument")
        
        self._expect('RPAREN', "Expected ')' to end function arguments")
        return neuro_ast.FunctionCall(func_name=func_name, args=args)

    # --- Helper methods for parsing --- 
    def _current_token(self):
        """Returns the current token being examined."""
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        # Should not happen if EOF is handled correctly, but return None defensively
        return None 
    
    def _peek_token(self, offset=1):
        """Looks ahead at a future token without consuming the current one."""
        peek_index = self.current_token_index + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index]
        return None # Or perhaps the EOF token itself

    def _advance(self):
        """Advances to the next token."""
        if self.current_token_index < len(self.tokens) - 1: # Stop before EOF
             self.current_token_index += 1
        elif self._current_token() and self._current_token().type != 'EOF':
            # If we are at the last non-EOF token, advancing moves us effectively to EOF
            self.current_token_index += 1

    def _expect(self, token_type, error_message=None, expected_value=None):
        """Verifies the current token is of the expected type/value and advances."""
        token = self._current_token()
        if not token:
             # Reached end of tokens unexpectedly
             raise self._syntax_error(f"Unexpected end of input, expected {token_type}", is_eof=True)
             
        if token.type == token_type:
            if expected_value is not None and token.value != expected_value:
                 err_msg = error_message or f"Expected token value '{expected_value}' but found '{token.value}'"
                 raise self._syntax_error(err_msg)
            self._advance()
            return token
        else:
            err_msg = error_message or f"Expected {token_type} but found {token.type} ('{token.value}')"
            raise self._syntax_error(err_msg)
            
    def _syntax_error(self, message, is_eof=False):
        """Helper to raise a NeuroSyntaxError with location info."""
        token = self._current_token()
        line = token.line if token and not is_eof else -1
        col = token.column if token and not is_eof else -1
        # Add token info to message for clarity
        full_message = f"{message} (at token: {token})" if token and not is_eof else message
        return NeuroSyntaxError(full_message, line, col) 