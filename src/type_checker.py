"""
Performs static type checking on the Neuro AST.
"""

import src.neuro_ast as ast
# Import layer classes to instantiate them for type checking
import src.layers as layers
# Import loss and optimizer classes
import src.losses as losses
import src.optimizers as optimizers
from typing import Any
from src.neuro_types import (
    NeuroType, AnyType, VoidType, IntType, FloatType, BoolType, StringType,
    TensorType, LayerType, ModelType, LossType, OptimizerType, FunctionType, DataType,
    NEURO_INT, NEURO_FLOAT, NEURO_BOOL, NEURO_STRING, NEURO_ANY, NEURO_VOID
)
from src.errors import NeuroTypeError

# --- Layer Registry --- Needed to map layer names to classes
# TODO: Auto-discover layers?
LAYER_CLASS_MAP = {
    "Dense": layers.DenseLayer,
    "Conv2D": layers.Conv2dLayer,
    "BatchNorm": layers.BatchNormLayer,
    "Flatten": layers.FlattenLayer,
    "MaxPool2D": layers.MaxPool2dLayer,
    "AvgPool2D": layers.AvgPool2dLayer,
    # Add other layers here as they are implemented
}

LOSS_CLASS_MAP = {
    "BCE": losses.BCELoss,
    # Add other losses (e.g., "MSE", "CrossEntropy") here
}

OPTIMIZER_CLASS_MAP = {
    "Adam": optimizers.Adam,
    # Add other optimizers (e.g., "SGD", "AdamW") here
}

# --- Built-in Function Signatures ---
# TODO: Expand this significantly
BUILTIN_FUNCTIONS = {
    "print": FunctionType(param_types=[NEURO_ANY], return_type=NEURO_VOID), # Simplified print
    # Assume load_data takes a string path and returns a generic DataType for now
    "load_data": FunctionType(param_types=[NEURO_STRING], return_type=DataType(element_type=NEURO_ANY)),
    # Add more built-ins
    "len": FunctionType(param_types=[NEURO_ANY], return_type=NEURO_INT), # Simplified len
    "type": FunctionType(param_types=[NEURO_ANY], return_type=StringType()), # Returns type name as string
    "range": FunctionType(param_types=[NEURO_INT], return_type=NEURO_ANY), # Simplified range, returns iterable Any
    # --- Added Math/Utility Built-ins ---
    # abs(Int) -> Int, abs(Float) -> Float - Need overloading or Union type handling
    # Simplified: abs(Numeric) -> Numeric (Let's use Any for now, can refine later if Union types added)
    "abs": FunctionType(param_types=[NEURO_ANY], return_type=NEURO_ANY), # TODO: Refine with NumericType/Union
    # round(Float) -> Float (Simplifying: ignoring ndigits, always returning Float)
    "round": FunctionType(param_types=[NEURO_FLOAT], return_type=NEURO_FLOAT),
    # Simplified max/min: Assume two numeric arguments for now
    # TODO: Handle variable args / require NumericType
    "max": FunctionType(param_types=[NEURO_ANY, NEURO_ANY], return_type=NEURO_ANY),
    "min": FunctionType(param_types=[NEURO_ANY, NEURO_ANY], return_type=NEURO_ANY),
    # TODO: Add more math functions (sqrt, log, sin, cos...), data manipulation functions etc.
}

# --- Method Signatures ---
# Store signatures for methods of known types (e.g., ModelType)
# Key: Type Name (e.g., "ModelType"), Value: Dict[method_name, FunctionType]
METHOD_SIGNATURES = {
    "ModelType": {
        # train(data, epochs: int, ...) -> void
        # Evaluate(data) -> results_type?
        # save(path: string) -> void
        # We checked train/evaluate as statements, but they could be expressions
        # Let's add save as an example method call
        "save": FunctionType(param_types=[NEURO_STRING], return_type=NEURO_VOID)
        # Add train, evaluate signatures if they can be called as expressions returning values
    },
    # Add signatures for other types like DataType if they have methods
    # "DataType": {
    #    "split": FunctionType(param_types=[NEURO_FLOAT, NEURO_FLOAT], return_type=DataType(...))
    # }
}

# Simple symbol table for now
# TODO: Implement proper scoping
class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def declare(self, name: str, type: NeuroType):
        if name in self.symbols:
            # Handle redeclaration error
            # Simple overwrite for now
            pass 
        self.symbols[name] = type

    def lookup(self, name: str) -> NeuroType | None:
        type = self.symbols.get(name)
        if type is None and self.parent:
            return self.parent.lookup(name)
        return type

class TypeChecker:
    """Traverses the AST and performs type checking."""

    def __init__(self):
        self.current_scope = SymbolTable()
        # Initialize scope with built-ins
        for name, type_sig in BUILTIN_FUNCTIONS.items():
             self.current_scope.declare(name, type_sig)
        self.errors = []
        self.current_expected_input_type: NeuroType = NEURO_ANY 

    def check(self, node: ast.ASTNode) -> NeuroType:
        """Main entry point to type check a node or the entire program."""
        self.errors = [] # Reset errors for a new check
        # Reset scope for a full check, ensuring built-ins are included
        self.current_scope = SymbolTable() 
        for name, type_sig in BUILTIN_FUNCTIONS.items():
            self.current_scope.declare(name, type_sig)
            
        self.current_expected_input_type = NEURO_ANY
        try:
            # For a Program node, we check each statement
            if isinstance(node, ast.Program):
                for stmt in node.body:
                    self.visit(stmt)
                # The type of a program itself is usually Void
                node.type_info = NEURO_VOID
                return NEURO_VOID
            else:
                # For other nodes, visit and return the inferred type
                return self.visit(node)
        except NeuroTypeError as e:
            # This top-level catch might hide specific error locations
            # Errors are collected in self.errors
            if not self.errors: # Add if not already added by type_error
                 self.errors.append(e)
            if hasattr(node, 'type_info'):
                node.type_info = NEURO_ANY 
            return NEURO_ANY
        # Return AnyType if errors occurred, otherwise the final type
        return NEURO_ANY if self.errors else (node.type_info if hasattr(node, 'type_info') else NEURO_VOID)

    def visit(self, node: ast.ASTNode) -> NeuroType:
        """Generic visit method that dispatches to specific node types."""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        # Wrap the call to handle potential errors within visits
        try:
            return visitor(node)
        except NeuroTypeError as e:
            if e not in self.errors:
                 self.errors.append(e)
            # Assign AnyType to the node where the error occurred
            if hasattr(node, 'type_info'):
                 node.type_info = NEURO_ANY
            return NEURO_ANY # Propagate AnyType upwards after error
        except Exception as e:
            # Catch unexpected errors during type checking
            # This indicates a bug in the type checker itself
            print(f"[Internal Type Checker Error] Visiting {type(node).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.type_error(f"Internal error during type checking: {e}", node)
            return NEURO_ANY

    def generic_visit(self, node: ast.ASTNode):
        """Called if no specific visitor method exists for a node type."""
        # For container nodes, visit children by default
        # Specific visitors might override this
        # Ensure node is an ASTNode before calling vars()
        if isinstance(node, ast.ASTNode):
            for field, value in vars(node).items():
                # Avoid infinite loops by not re-visiting the node itself via children
                if field == 'parent': continue # Example if AST nodes have parent pointers
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.ASTNode):
                            self.visit(item)
                elif isinstance(value, ast.ASTNode):
                    self.visit(value)
        # Default type if not set by a specific visitor
        # Check if type_info exists and is None (use Ellipsis for default)
        if hasattr(node, 'type_info') and getattr(node, 'type_info', ...) is None:
             node.type_info = NEURO_ANY # Or maybe VoidType? Or raise error?
        return getattr(node, 'type_info', NEURO_VOID) # Return Any/Void if no type_info

    def type_error(self, message: str, node: ast.ASTNode):
        # Add line/column info from node if available
        # Extract line/column if available, otherwise use None
        line = getattr(node, 'line', None)
        col = getattr(node, 'column', None)
        # Ideally, we'd also get the source line here, but that requires
        # passing the source code content or lines down. For now, just line/col.
        
        err_message = f"Type Error: {message}" # Keep the original message format for now
        err = NeuroTypeError(err_message, line=line, column=col) # Pass line/col
        
        if err not in self.errors: # Avoid duplicate errors from propagation
            self.errors.append(err)
        # Potentially raise immediately or just collect
        # raise err
        # Set node type to Any to prevent cascade errors, return Any
        if hasattr(node, 'type_info'):
            node.type_info = NEURO_ANY
        return NEURO_ANY

    # --- Value Resolving Helper --- 
    # Type checker sometimes needs concrete values (e.g., layer units)
    def _resolve_value(self, node: ast.ASTNode) -> Any:
        """Attempts to resolve the static value of an AST node (e.g., a literal) or returns raw Python values."""
        # If input is already a basic Python type, return it directly
        if isinstance(node, (int, float, str, bool)) or node is None:
             return node
             
        # If input is an AST node, try to resolve its value
        if isinstance(node, ast.NumberLiteral):
            return node.value
        elif isinstance(node, ast.StringLiteral):
            return node.value
        # Handle boolean literals if they exist in the AST
        # elif isinstance(node, ast.BooleanLiteral):
        #     return node.value
        elif isinstance(node, ast.Identifier):
            # Basic lookup - doesn't handle complex expressions or non-constant vars
            # We might need to store values along with types in symbol table for constants
            # Cannot resolve variable identifiers to static values here.
            # If a static value is required (like for layer units), this should fail.
            self.type_error(f"Expected a static literal value (number, string) but got identifier '{node.name}'. Required for configuration.", node)
            return None # Indicate failure
        # Add other literal types (BoolLiteral?) if they exist
        # Cannot resolve complex expressions statically here
        self.type_error(f"Cannot statically resolve value for AST node type {type(node).__name__}. Required for layer configuration.", node)
        return None # Indicate failure

    # --- Visitor Methods for Specific AST Nodes ---

    def visit_Program(self, node: ast.Program) -> NeuroType:
        # Type checking logic handled in the main 'check' method for Program
        # Here we just ensure child statements are visited
        for stmt in node.body:
            self.visit(stmt)
        node.type_info = NEURO_VOID
        return node.type_info

    def visit_NumberLiteral(self, node: ast.NumberLiteral) -> NeuroType:
        if isinstance(node.value, int):
            node.type_info = NEURO_INT
        elif isinstance(node.value, float):
            node.type_info = NEURO_FLOAT
        else:
            # Should not happen if lexer/parser are correct
            return self.type_error(f"Unknown numeric literal type: {type(node.value)}", node)
        return node.type_info

    def visit_StringLiteral(self, node: ast.StringLiteral) -> NeuroType:
        node.type_info = NEURO_STRING
        return node.type_info

    def visit_Identifier(self, node: ast.Identifier) -> NeuroType:
        var_type = self.current_scope.lookup(node.name)
        if var_type is None:
            return self.type_error(f"Variable '{node.name}' not defined.", node)
        node.type_info = var_type
        return node.type_info

    def visit_AssignmentStatement(self, node: ast.AssignmentStatement) -> NeuroType:
        value_type = self.visit(node.value)

        # Simple assignment: target = value
        if isinstance(node.target, ast.Identifier):
            target_name = node.target.name
            target_node = node.target
            # Allow redeclaration for now, just update type
            self.current_scope.declare(target_name, value_type)
            # Assign the inferred type to the identifier node as well
            target_node.type_info = value_type
        # TODO: Handle tuple assignment: target1, target2 = value
        # Requires value_type to be some kind of tuple/sequence type
        elif isinstance(node.target, list): # Assuming target is list of Identifiers
             # Placeholder for tuple assignment logic
             # Need to check if value_type is a sequence type and unpack it
             self.type_error("Tuple assignment not yet supported by type checker.", node)
             # Type the target identifiers as Any
             for target_node in node.target:
                 if isinstance(target_node, ast.Identifier):
                     target_node.type_info = NEURO_ANY
                     self.current_scope.declare(target_node.name, NEURO_ANY)
        else:
             self.type_error(f"Invalid assignment target type: {type(node.target).__name__}", node)

        # Assignment statement itself has no value
        node.type_info = NEURO_VOID
        return node.type_info

    # --- Updated Visitors for Neuro-specific constructs ---

    def visit_ModelDefinition(self, node: ast.ModelDefinition) -> NeuroType:
        # 1. Determine initial expected input type (e.g., from params or Any)
        initial_input_type: NeuroType = NEURO_ANY
        input_shape_expr = node.params.get('input_shape') # Look for input_shape param
        if input_shape_expr:
            # TODO: Need a more robust way to evaluate expressions to tuples
            # For now, assume it might be a literal tuple (represented how in AST?)
            # or maybe a list of NumberLiterals?
            # Let's try resolving the value and expect a tuple of ints/None
            resolved_shape = self._resolve_value(input_shape_expr)
            if isinstance(resolved_shape, tuple) and all(isinstance(d, int) or d is None for d in resolved_shape):
                 # Assuming Float32 default for now if shape is given
                 initial_input_type = TensorType(shape=resolved_shape, dtype=NEURO_FLOAT)
            else:
                # If input_shape is defined but not a valid tuple literal, report error
                 self.type_error("Model 'input_shape' parameter must be a valid tuple literal (e.g., (None, 784)).", input_shape_expr or node)
                 initial_input_type = NEURO_ANY # Set to Any on error

        # Store previous expected input type to restore after visiting layers
        previous_expected_input_type = self.current_expected_input_type
        self.current_expected_input_type = initial_input_type

        model_input_type: NeuroType = NEURO_ANY
        model_output_type: NeuroType = NEURO_ANY

        for i, layer_node in enumerate(node.layers):
            # Visit the layer node. visit_Layer will use and update 
            # self.current_expected_input_type
            layer_instance_type = self.visit(layer_node) 

            if not isinstance(layer_instance_type, LayerType):
                 # Error already reported by visit_Layer or sub-visit
                 # Propagate AnyType to avoid cascade failures
                 self.current_expected_input_type = NEURO_ANY 
                 continue # Skip compatibility check if layer type is invalid

            if i == 0:
                # The actual input type processed by the first layer
                model_input_type = layer_instance_type.input_type
                # We might also want to check if initial_input_type (from input_shape)
                # is compatible with the first layer's expected input type here.
                # This is already implicitly checked in visit_Layer call.

            # Compatibility check is implicitly handled within visit_Layer now,
            # as it receives the expected input type.
            # Update the expected type for the *next* layer
            self.current_expected_input_type = layer_instance_type.output_type

        # The output type of the model is the output type of the last layer
        model_output_type = self.current_expected_input_type
        # If the initial input was Any, but the first layer defined its input, use that.
        if isinstance(model_input_type, AnyType) and not isinstance(initial_input_type, AnyType):
             model_input_type = initial_input_type


        # Restore expected input type for outer scope
        self.current_expected_input_type = previous_expected_input_type

        # Final model type
        final_model_type = ModelType(input_type=model_input_type, output_type=model_output_type)
        node.type_info = final_model_type

        return final_model_type

    def visit_Layer(self, node: ast.Layer) -> NeuroType:
        layer_class_name = node.layer_type
        layer_cls = LAYER_CLASS_MAP.get(layer_class_name)

        if layer_cls is None:
            return self.type_error(f"Unknown layer type: '{layer_class_name}'", node)

        # Resolve parameter values (needed for layer instantiation)
        resolved_params = {}
        param_types = {}
        has_param_errors = False
        for key, expr_node in node.params.items():
            # Visit the expression node to get its type
            param_types[key] = self.visit(expr_node)
            # Attempt to get the static value (needed for configuration)
            value = self._resolve_value(expr_node)
            if value is None: # _resolve_value reports error if resolution fails
                 has_param_errors = True
            resolved_params[key] = value
        
        if has_param_errors:
             # Cannot reliably instantiate layer if params have errors or aren't static
             node.type_info = NEURO_ANY # Mark layer type as unknown
             return NEURO_ANY

        # Instantiate the layer object (from src.layers) to get its signature
        try:
            # Need to map AST param names/values to the layer class __init__ args
            # This assumes param names in Neuro syntax match __init__ args
            layer_instance = layer_cls(**resolved_params)
        except TypeError as e:
            # Catch errors from incorrect parameters passed to __init__
            return self.type_error(f"Invalid parameters for layer '{layer_class_name}': {e}", node)
        except Exception as e:
             # Catch other potential instantiation errors
             return self.type_error(f"Error instantiating layer '{layer_class_name}': {e}", node)

        # Get the type signature using the *expected* input type for this layer
        # This expected type was set by the previous layer (or initial type) in visit_ModelDefinition
        layer_type_signature = layer_instance.get_type_signature(self.current_expected_input_type)

        # Perform compatibility check: Does the expected input match the layer's actual required input?
        # The get_type_signature might already incorporate some checks, but we can be explicit.
        # The LayerType returned by get_type_signature contains the input type the layer *processed*
        # (which might be AnyType if the actual input was incompatible). We compare the *original* expected type.
        if not self.current_expected_input_type.is_compatible(layer_type_signature.input_type):
             # Only raise error if the input wasn't already AnyType (error propagation)
             if not isinstance(self.current_expected_input_type, AnyType):
                 self.type_error(
                     f"Type mismatch for layer '{layer_class_name}'. "
                     f"Expected input compatible with {layer_type_signature.input_type!r}, "
                     f"but received {self.current_expected_input_type!r} from previous layer.",
                     node
                 )
                 # Return AnyType to signify error, but use the signature from the layer 
                 # assuming it handled the bad input somehow for further checks?
                 # Or just return AnyType? Let's stick with the calculated signature for now.
                 # If the layer sig is Any, Any -> Any, that's fine.

        # Store the calculated LayerType on the AST node
        node.type_info = layer_type_signature

        # Return the signature (which includes the output type for the next layer)
        return layer_type_signature

    def visit_LossDefinition(self, node: ast.LossDefinition) -> NeuroType:
        # Resolve parameters, similar to visit_Layer
        resolved_params = {}
        param_types = {}
        has_param_errors = False
        # Assume 'type' parameter specifies the loss function
        loss_type_name_node = node.params.get('type')
        if not loss_type_name_node:
            return self.type_error("Loss definition requires a 'type' parameter (e.g., type=\"BCE\").", node)
        
        loss_type_name = self._resolve_value(loss_type_name_node)
        if not isinstance(loss_type_name, str):
             return self.type_error("Loss 'type' parameter must be a string literal.", loss_type_name_node)

        loss_cls = LOSS_CLASS_MAP.get(loss_type_name)
        if loss_cls is None:
             return self.type_error(f"Unknown loss type: '{loss_type_name}'", loss_type_name_node)

        # Resolve remaining parameters for instantiation
        for key, expr_node in node.params.items():
            if key == 'type': continue # Already handled
            param_types[key] = self.visit(expr_node)
            value = self._resolve_value(expr_node)
            if value is None:
                 has_param_errors = True
            resolved_params[key] = value

        if has_param_errors:
             node.type_info = NEURO_ANY
             return NEURO_ANY

        # Instantiate loss to get signature
        try:
            loss_instance = loss_cls(**resolved_params)
        except TypeError as e:
            return self.type_error(f"Invalid parameters for loss '{loss_type_name}': {e}", node)
        except Exception as e:
            return self.type_error(f"Error instantiating loss '{loss_type_name}': {e}", node)

        loss_type_signature = loss_instance.get_type_signature()
        node.type_info = loss_type_signature
        
        # Store the loss configuration globally? Or require assignment?
        # For now, let's store it in the symbol table with a known name
        # This allows visit_TrainStatement to find it.
        self.current_scope.declare("__current_loss__", loss_type_signature)
        
        return loss_type_signature

    def visit_OptimizerDefinition(self, node: ast.OptimizerDefinition) -> NeuroType:
        # Similar logic to LossDefinition
        resolved_params = {}
        param_types = {}
        has_param_errors = False
        opt_type_name_node = node.params.get('type')
        if not opt_type_name_node:
            return self.type_error("Optimizer definition requires a 'type' parameter (e.g., type=\"Adam\").", node)
        
        opt_type_name = self._resolve_value(opt_type_name_node)
        if not isinstance(opt_type_name, str):
            return self.type_error("Optimizer 'type' parameter must be a string literal.", opt_type_name_node)

        opt_cls = OPTIMIZER_CLASS_MAP.get(opt_type_name)
        if opt_cls is None:
             return self.type_error(f"Unknown optimizer type: '{opt_type_name}'", opt_type_name_node)

        for key, expr_node in node.params.items():
            if key == 'type': continue
            param_types[key] = self.visit(expr_node)
            value = self._resolve_value(expr_node)
            if value is None: has_param_errors = True
            resolved_params[key] = value
        
        if has_param_errors:
            node.type_info = NEURO_ANY
            return NEURO_ANY

        # Instantiate optimizer
        try:
            # Optimizers need 'params' which aren't available statically here.
            # BaseOptimizer.__init__ handles params=None for type checking.
            optimizer_instance = opt_cls(**resolved_params)
        except TypeError as e:
            return self.type_error(f"Invalid parameters for optimizer '{opt_type_name}': {e}", node)
        except Exception as e:
            return self.type_error(f"Error instantiating optimizer '{opt_type_name}': {e}", node)

        optimizer_type_signature = optimizer_instance.get_type_signature()
        node.type_info = optimizer_type_signature
        
        # Store globally in symbol table
        self.current_scope.declare("__current_optimizer__", optimizer_type_signature)
        
        return optimizer_type_signature

    def visit_TrainStatement(self, node: ast.TrainStatement) -> NeuroType:
        # 1. Check model_name exists and is a ModelType
        model_var_type = self.visit(node.model_name) # Visit the Identifier node
        if not isinstance(model_var_type, ModelType):
             return self.type_error(f"'{node.model_name.name}' is not a trainable Model.", node.model_name)

        # 2. Check data_source resolves to a compatible DataType
        data_source_type = self.visit(node.data_source)
        if not isinstance(data_source_type, (DataType, AnyType)): # Allow AnyType to avoid cascade errors
            return self.type_error("Invalid data source for training, expected DataType.", node.data_source)

        # 3. Check training params are valid types
        # Example: Check 'epochs' parameter
        epochs_expr = node.params.get('epochs')
        if epochs_expr:
             epochs_type = self.visit(epochs_expr)
             if not isinstance(epochs_type, IntType):
                 self.type_error("'epochs' parameter in train statement must be an integer.", epochs_expr)
        # TODO: Check other params like batch_size etc.
        # Check 'batch_size' parameter
        batch_size_expr = node.params.get('batch_size')
        if batch_size_expr:
            batch_size_type = self.visit(batch_size_expr)
            if not isinstance(batch_size_type, IntType):
                 self.type_error("'batch_size' parameter in train statement must be an integer.", batch_size_expr)

        # 4. Check model's input type matches data source's element type
        if isinstance(data_source_type, DataType) and isinstance(model_var_type, ModelType):
             # Assuming data element is just the input features for train
             # A more complex DataType might have tuples (input, target)
             # We need a clearer definition of DataType.element_type structure
             element_input_type = data_source_type.element_type # Assuming element_type is the input feature type
             if not element_input_type.is_compatible(model_var_type.input_type):
                  # Avoid error if types are Any
                  if not isinstance(element_input_type, AnyType) and not isinstance(model_var_type.input_type, AnyType):
                      self.type_error(
                          f"Model input type {model_var_type.input_type!r} is not compatible "
                          f"with data source element type {element_input_type!r} for training.",
                          node.data_source
                      )

        # 5. Check if Loss and Optimizer have been defined and are compatible
        loss_type = self.current_scope.lookup("__current_loss__")
        optimizer_type = self.current_scope.lookup("__current_optimizer__")

        if not isinstance(loss_type, LossType):
            # Error if loss not defined before train, or if lookup fails
            self.type_error("Loss function must be defined using Loss(...) before training.", node)
        else:
             # Check compatibility: Loss prediction input vs Model output
             if not model_var_type.output_type.is_compatible(loss_type.pred_type):
                 # Allow AnyType compatibility to avoid cascade errors
                 if not isinstance(model_var_type.output_type, AnyType) and not isinstance(loss_type.pred_type, AnyType):
                     self.type_error(
                         f"Model output type {model_var_type.output_type!r} "
                         f"is not compatible with the loss function's expected prediction input type {loss_type.pred_type!r}.",
                         node # Error on the train statement node?
                     )
             # TODO: Check loss_type.target_type compatibility with data_source element type
             # Example (depends on DataType having element_type):
             # if isinstance(data_source_type, DataType) and not data_source_type.element_type.is_compatible(loss_type.target_type):
             #     # Check if element_type is a tuple (e.g., features, labels) and compare label part
             #     # This needs a more structured DataType.element_type, maybe TupleType
             #     self.type_error(f"Data source element type {data_source_type.element_type!r} incompatible with loss target type {loss_type.target_type!r}", node.data_source)
             pass # Skip detailed target check for now

        if not isinstance(optimizer_type, OptimizerType):
             self.type_error("Optimizer must be defined using Optimizer(...) before training.", node)

        node.type_info = NEURO_VOID
        return node.type_info

    # --- Visitors for Calls ---

    def visit_FunctionCall(self, node: ast.FunctionCall) -> NeuroType:
        # 1. Check if the function name is defined
        func_name_id = node.func_name
        if not isinstance(func_name_id, ast.Identifier):
            # Parser should ensure this, but check defensively
            return self.type_error("Function name must be an identifier.", node)
        
        func_name = func_name_id.name
        func_type = self.current_scope.lookup(func_name)

        if func_type is None:
            return self.type_error(f"Function '{func_name}' is not defined.", func_name_id)
        
        if not isinstance(func_type, FunctionType):
            return self.type_error(f"'{func_name}' is not callable (not a function).", func_name_id)

        # 2. Check number of arguments
        # TODO: Handle named args, varargs, default args if syntax supports them
        expected_arg_count = len(func_type.param_types)
        actual_arg_count = len(node.args)
        if expected_arg_count != actual_arg_count:
            return self.type_error(f"Function '{func_name}' expects {expected_arg_count} arguments, but received {actual_arg_count}.", node)

        # 3. Check argument types
        for i, arg_node in enumerate(node.args):
            arg_type = self.visit(arg_node) # Get the type of the argument expression
            expected_arg_type = func_type.param_types[i]
            if not arg_type.is_compatible(expected_arg_type):
                # Allow AnyType compatibility to avoid cascade errors
                 if not isinstance(arg_type, AnyType) and not isinstance(expected_arg_type, AnyType):
                     self.type_error(
                         f"Argument {i+1} for function '{func_name}' has incompatible type. "
                         f"Expected type compatible with {expected_arg_type!r}, but got {arg_type!r}.",
                         arg_node
                     )
                     # If an arg is wrong, the return type is likely compromised
                     # We could continue checking other args but mark overall call as Any
        
        # If all checks pass, assign the function's return type to the call node
        node.type_info = func_type.return_type
        return node.type_info

    def visit_MethodCall(self, node: ast.MethodCall) -> NeuroType:
        # 1. Determine the type of the object being called upon
        object_name_id = node.object_name
        if not isinstance(object_name_id, ast.Identifier):
             return self.type_error("Object in method call must be an identifier.", node)
        
        object_name = object_name_id.name
        object_type = self.current_scope.lookup(object_name)
        
        if object_type is None:
             return self.type_error(f"Variable '{object_name}' not defined.", object_name_id)

        # 2. Look up the method signature for that type
        method_name = node.method_name
        object_type_name = type(object_type).__name__ # Get name like "ModelType", "DataType"
        
        type_methods = METHOD_SIGNATURES.get(object_type_name)
        if not type_methods:
            return self.type_error(f"Type '{object_type_name}' has no defined methods.", node)
            
        method_sig = type_methods.get(method_name)
        if not method_sig:
             return self.type_error(f"Type '{object_type_name}' has no method named '{method_name}'.", node)
        
        if not isinstance(method_sig, FunctionType):
             # Should not happen if METHOD_SIGNATURES is defined correctly
             return self.type_error(f"Internal error: Method signature for '{object_name}.{method_name}' is not a FunctionType.", node)

        # 3. Check arguments (assuming named arguments for now as node.args is a dict)
        # This is simpler than positional args check in FunctionCall
        expected_params = method_sig.param_types # This assumes positional from FunctionType def
        # TODO: Refactor FunctionType to better support named parameters for method calls
        # For now, let's assume node.args keys match expected param names/order implicitly
        
        # Basic argument count check (crude for named args)
        # A proper check needs parameter names in FunctionType
        if len(expected_params) != len(node.args):
             return self.type_error(f"Method '{method_name}' expects {len(expected_params)} arguments, but received {len(node.args)}.", node)

        # Check argument types (assuming order matches for now)
        # This needs improvement based on actual parameter names
        i = 0
        for arg_name, arg_node in node.args.items():
            if i >= len(expected_params):
                # Should have been caught by length check, but be safe
                break 
            arg_type = self.visit(arg_node)
            expected_arg_type = expected_params[i]
            if not arg_type.is_compatible(expected_arg_type):
                 if not isinstance(arg_type, AnyType) and not isinstance(expected_arg_type, AnyType):
                     self.type_error(
                         f"Argument '{arg_name}' for method '{method_name}' has incompatible type. "
                         f"Expected {expected_arg_type!r}, but got {arg_type!r}.",
                         arg_node
                     )
            i += 1

        # Assign return type
        node.type_info = method_sig.return_type
        return node.type_info

    def visit_EvaluateStatement(self, node: ast.EvaluateStatement) -> NeuroType:
        # Similar checks to TrainStatement, but simpler
        # 1. Check model type
        model_var_type = self.visit(node.model_name)
        if not isinstance(model_var_type, ModelType):
             return self.type_error(f"'{node.model_name.name}' is not an evaluatable Model.", node.model_name)
        
        # 2. Check data source type
        data_source_type = self.visit(node.data_source)
        if not isinstance(data_source_type, (DataType, AnyType)): # Allow AnyType to avoid cascade errors
            return self.type_error("Invalid data source for evaluation, expected DataType.", node.data_source)

        # 3. Check model input vs data source element compatibility
        if isinstance(data_source_type, DataType) and isinstance(model_var_type, ModelType):
             # Assuming data element is just the input features for evaluate
             # A more complex DataType might have tuples (input, target)
             # We need a clearer definition of DataType.element_type structure
             element_input_type = data_source_type.element_type # Assuming element_type is the input feature type
             if not element_input_type.is_compatible(model_var_type.input_type):
                  # Avoid error if types are Any
                  if not isinstance(element_input_type, AnyType) and not isinstance(model_var_type.input_type, AnyType):
                      self.type_error(
                          f"Model input type {model_var_type.input_type!r} is not compatible "
                          f"with data source element type {element_input_type!r} for evaluation.",
                          node.data_source
                      )

        # Evaluate statement likely returns some results (e.g., loss, accuracy)
        # For now, return Void, as the exact return type isn't defined.
        # Could define a ResultsType later.
        node.type_info = NEURO_VOID # Placeholder
        return node.type_info

    def visit_SaveStatement(self, node: ast.SaveStatement) -> NeuroType:
        # 1. Check model type
        model_var_type = self.visit(node.model_name)
        if not isinstance(model_var_type, ModelType):
             return self.type_error(f"'{node.model_name.name}' is not a saveable Model.", node.model_name)

        # 2. Check filepath type
        filepath_type = self.visit(node.filepath)
        if not isinstance(filepath_type, StringType):
            return self.type_error("Filepath argument for save must be a string.", node.filepath)
            
        node.type_info = NEURO_VOID
        return node.type_info

# Example usage (would typically be called after parsing)
# parser = Parser(tokens)
# ast_tree = parser.parse()
# type_checker = TypeChecker()
# overall_type = type_checker.check(ast_tree)
# if type_checker.errors:
#     for error in type_checker.errors:
#         print(error)
# else:
#     print("Type checking passed.")
#     # Proceed to interpreter, potentially passing the typed AST 