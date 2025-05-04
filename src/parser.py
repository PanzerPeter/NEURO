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
        self._source_lines = [] # Store source code lines for error reporting

    def parse(self, code):
        """Parses the NEURO code string and returns an AST Root (Program node)."""
        # print(f"Parsing code:\n{code[:100]}...") # Debug print
        self._source_lines = code.splitlines() # Store lines for error context
        
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
            elif isinstance(statement, (neuro_ast.TrainStatement, neuro_ast.EvaluateStatement, neuro_ast.FunctionCall)):
                # These method calls/function calls are statements themselves, expect semicolon
                 self._expect('SEMICOLON', f"Expected ';' after {type(statement).__name__} statement")
            # Add other statement types that need semicolons here
            
        return neuro_ast.Program(body=body)

    def _parse_statement(self):
        """ Parses a single statement (e.g., assignment, model call). """
        # Check for keywords or patterns that identify the statement type
        token = self._current_token()
        
        # Check for Assignment (potentially tuple assignment)
        if token.type == 'IDENTIFIER':
            # Need to look ahead past potential comma-separated identifiers for the ASSIGN token
            lookahead_index = 1
            peek_token = self._peek_token(lookahead_index)
            is_potentially_assignment = True
            while peek_token and peek_token.type == 'COMMA':
                lookahead_index += 1
                peek_token = self._peek_token(lookahead_index) # Should be IDENTIFIER
                if not peek_token or peek_token.type != 'IDENTIFIER':
                    # Invalid syntax like "a," or "a, 1"
                    is_potentially_assignment = False
                    break 
                lookahead_index += 1
                peek_token = self._peek_token(lookahead_index) # Token after the identifier
            
            # If we found an ASSIGN after the identifier(s)
            if is_potentially_assignment and peek_token and peek_token.type == 'ASSIGN':
                return self._parse_assignment_statement()
            # Check for method call (e.g., object.method())
            elif self._peek_token(1) and self._peek_token(1).type == 'DOT':
                # Parse the method call as a statement itself
                # Method calls like train/evaluate are parsed by _parse_method_call_expression
                # This returns a specific statement node (TrainStatement, EvaluateStatement)
                call_statement = self._parse_method_call_expression()
                return call_statement
            # NEW: Check for general function call (e.g., print(...))
            elif self._peek_token(1) and self._peek_token(1).type == 'LPAREN':
                 # Parse the function call as a statement
                 func_call_node = self._parse_general_function_call()
                 # Ensure it returns a FunctionCall node
                 if not isinstance(func_call_node, neuro_ast.FunctionCall):
                      # This should ideally not happen if _parse_general_function_call is correct
                      raise self._syntax_error(f"Expected a FunctionCall node, but parsing returned {type(func_call_node)}")
                 return func_call_node
            else:
                # If it wasn't an assignment or a method call or a function call, it's an error
                next_token_type = self._peek_token(1).type if self._peek_token(1) else '[EOF]'
                raise self._syntax_error(f"Unexpected token type '{next_token_type}' after identifier '{token.value}'")
        else:
            # Handle other potential statement types here if needed
            raise self._syntax_error(f"Unexpected token '{token.value}' ({token.type}) at start of statement")

    def _parse_assignment_statement(self):
        """ Parses: IDENTIFIER ASSIGN expression 
        Parses: IDENTIFIER [, IDENTIFIER]* ASSIGN expression """
        targets = []
        # Parse the first identifier
        target_token = self._expect('IDENTIFIER', "Expected variable name in assignment")
        targets.append(neuro_ast.Identifier(name=target_token.value))
        
        # Check for more identifiers separated by commas
        while self._current_token().type == 'COMMA':
            self._advance() # Consume comma
            next_target_token = self._expect('IDENTIFIER', "Expected variable name after comma in tuple assignment")
            targets.append(neuro_ast.Identifier(name=next_target_token.value))
            
        self._expect('ASSIGN', "Expected '=' in assignment")
        
        # Parse the right-hand side expression/call
        value = self._parse_expression()
        
        # If only one target, store it directly, otherwise store the list
        final_target = targets[0] if len(targets) == 1 else targets
        
        # If the value is a ModelDefinition, set its name (only makes sense for single target assignment)
        # if isinstance(value, neuro_ast.ModelDefinition):
        #     value.name = target.name
        if isinstance(final_target, neuro_ast.Identifier) and isinstance(value, neuro_ast.ModelDefinition):
            value.name = final_target.name
        elif isinstance(targets, list) and isinstance(value, neuro_ast.ModelDefinition):
             raise self._syntax_error("Cannot assign a ModelDefinition to multiple targets")
            
        # return neuro_ast.AssignmentStatement(target=target, value=value)
        return neuro_ast.AssignmentStatement(target=final_target, value=value)

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
        # Identifier (as a value/variable)
        elif token.type == 'IDENTIFIER':
             self._advance()
             return neuro_ast.Identifier(name=token.value)
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
        layer_type_token = self._expect('IDENTIFIER', "Expected layer type (e.g., Dense, Conv2D)")
        layer_type = layer_type_token.value
        
        # Validate known layer types (optional but good practice)
        # known_layers = {'Dense', 'Conv2D', 'Dropout'} # Expand as needed
        # if layer_type not in known_layers:
        #     logging.warning(f"Parser encountered potentially unknown layer type: '{layer_type}' at line {layer_type_token.lineno}")
        #     # Depending on strictness, could raise NeuroSyntaxError here

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
        
        # Arguments differ for train vs evaluate vs others (like split)
        if method_name == 'train':
            data_source_node = None
            params = {}
            
            # Check for positional data argument first
            current = self._current_token()
            peek = self._peek_token()
            
            # If current is IDENTIFIER and next is COMMA or RPAREN, it's positional
            if current.type == 'IDENTIFIER' and (peek.type == 'COMMA' or peek.type == 'RPAREN'):
                data_source_node = neuro_ast.Identifier(name=current.value)
                self._advance() # Consume the identifier
                if self._current_token().type == 'COMMA':
                    self._advance() # Consume the comma
                    # Now parse remaining key-value pairs
                    params = self._parse_key_value_pairs()
                # Else (if RPAREN), params remains empty
            else:
                # Assume all arguments are key-value pairs
                all_args = self._parse_key_value_pairs()
                if 'data' in all_args:
                    # Allow data=... syntax
                    data_value = all_args.pop('data')
                    if not isinstance(data_value, neuro_ast.Identifier):
                        raise self._syntax_error(f"Expected 'data' argument for train to be an identifier, got {type(data_value)}")
                    data_source_node = data_value
                params = all_args # Remaining are the training parameters

            # Validate data source
            if not data_source_node:
                 raise self._syntax_error("Missing required 'data' argument for train statement")
            # if not isinstance(data_source_node, neuro_ast.Identifier): # Already checked if data= syntax used
            #     raise self._syntax_error(f"Expected 'data' argument to be an identifier, got {type(data_source_node)}")

            self._expect('RPAREN', "Expected ')' after train arguments")
            return neuro_ast.TrainStatement(model_name=object_name, data_source=data_source_node, params=params)
            
        elif method_name == 'evaluate':
            # Evaluate currently expects a single positional data source identifier
            # or data=... keyword argument.
            data_source_node = None
            current = self._current_token()
            peek = self._peek_token()

            if current.type == 'IDENTIFIER' and peek.type == 'RPAREN':
                 # Positional: evaluate(my_data)
                 data_source_node = neuro_ast.Identifier(name=current.value)
                 self._advance()
            elif current.type == 'IDENTIFIER' and current.value == 'data' and peek.type == 'ASSIGN':
                 # Keyword: evaluate(data=my_data)
                 # Parse the key-value pair using the helper
                 args = self._parse_key_value_pairs()
                 if 'data' in args and isinstance(args['data'], neuro_ast.Identifier):
                     data_source_node = args['data']
                 else:
                     raise self._syntax_error("Expected 'data=<identifier>' for evaluate statement")
            else:
                 raise self._syntax_error("Evaluate statement expects a single data source argument (e.g., evaluate(my_data) or evaluate(data=my_data))")

            self._expect('RPAREN', "Expected ')' after evaluate argument")
            return neuro_ast.EvaluateStatement(model_name=object_name, data_source=data_source_node)
        elif method_name == 'split':
            # split() takes key-value parameters
             params = self._parse_key_value_pairs()
             self._expect('RPAREN', "Expected ')' after split arguments")
             # Split is typically used on the RHS of an assignment,
             # so we return a FunctionCall node representing the method call.
             # The assignment statement visitor will handle unpacking.
             # We need args for FunctionCall, which are the params dict values?
             # This seems inconsistent. Let's create a specific SplitCall node.
             # Or return a generic MethodCall node?
             # For now, let's assume _parse_general_function_call structure can be adapted
             # or that split() is only used in assignment where the interpreter handles it.
             # Let's treat it like a general function call on an object for now.
             # We need the object identifier and method name.
             # Let's return a dedicated MethodCall node.
             return neuro_ast.MethodCall(object_name=object_name, method_name=method_name, args=params) # Assuming args is a dict here

        elif method_name == 'save':
             # save() expects a single string literal argument
             path_arg = self._parse_expression()
             if not isinstance(path_arg, neuro_ast.StringLiteral):
                 raise self._syntax_error("Expected a string literal file path for save()")
             self._expect('RPAREN', "Expected ')' after save argument")
             return neuro_ast.SaveStatement(model_name=object_name, filepath=path_arg)
        else:
            raise self._syntax_error(f"Unknown method name: '{method_name}' for object '{object_name}'")

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
                elif value_token.type == 'IDENTIFIER': # Could be a data source name or the keyword None
                    if value_token.value == 'None': # Handle None keyword
                        value = None
                    else:
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
                arg_expr = self._parse_expression()
                args.append(arg_expr)
                if self._current_token().type == 'RPAREN':
                    break
                elif self._current_token().type == 'COMMA':
                    self._advance() # Consume comma, continue loop
                else:
                    self._syntax_error(f"Expected ',' or ')' after function argument, got {self._current_token().type}")

        self._expect('RPAREN', "Expected ')' to end function call")
        
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
        """Helper to raise a NeuroSyntaxError with location info and source context."""
        token = self._current_token()
        line_num = token.line if token and not is_eof else -1
        col = token.column if token and not is_eof else -1
        source_line_content = None
        if line_num > 0 and line_num <= len(self._source_lines):
            source_line_content = self._source_lines[line_num - 1] # 0-based index
            
        # Add token info to message for clarity if not already detailed
        full_message = f"{message} (at token: {token})" if token and not is_eof and f"(at token: {token})" not in message else message
        
        return NeuroSyntaxError(full_message, line_num, col, source_line=source_line_content) 