"""
NEURO REPL Module
Provides interactive development environment for NEURO.
"""

import cmd
import sys
import torch
from typing import Optional, Dict, Any, List
from .interpreter import NeuroInterpreter
from .errors import NeuroError
from tabulate import tabulate
import torch.nn as nn
import graphviz
import os

class NeuroREPL(cmd.Cmd):
    intro = 'Welcome to NEURO REPL. Type help or ? to list commands.'
    prompt = '>>> '

    def __init__(self):
        super().__init__()
        self.interpreter = NeuroInterpreter()
        self.last_result = None
        self.debug_mode = False
        self.watches = set()
        self.break_conditions = []
        self.profiling = False
        self.variables = {}  # Store for user-defined variables
        self.history = []  # Command history
        self.completions = {
            'model.': ['train', 'evaluate', 'predict', 'save', 'load'],
            'Dense': ['units', 'activation', 'input_shape'],
            'Conv2D': ['filters', 'kernel_size', 'strides', 'padding'],
            'LSTM': ['units', 'return_sequences', 'activation'],
            'optimizer.': ['learning_rate', 'momentum', 'decay'],
            'loss.': ['categorical_crossentropy', 'binary_crossentropy', 'mse']
        }

    def get_completions(self, text: str) -> List[str]:
        """Get code completions for the given text."""
        if '.' in text:
            obj, _ = text.split('.')
            return self.completions.get(obj + '.', [])
        return [k[:-1] for k in self.completions.keys() if k.startswith(text)]

    def add_to_history(self, cmd: str) -> None:
        """Add a command to history."""
        if cmd.strip():
            self.history.append(cmd)

    def handle_input(self, code: str) -> Any:
        """
        Handle multi-line input and code execution.
        
        Args:
            code: The code to execute
            
        Returns:
            The result of execution
        """
        try:
            self.add_to_history(code)
            # Store result in variables if it's an assignment
            if '=' in code and not code.strip().startswith('if') and not code.strip().startswith('for'):
                var_name = code.split('=')[0].strip()
                result = self.interpreter.interpret(code)
                self.variables[var_name] = result
                if isinstance(result, nn.Module):
                    print("Model created successfully")
                return result
            else:
                result = self.interpreter.interpret(code)
                if result is not None:
                    print(result)
                return result
        except NeuroError as e:
            print(f"Error:\n{str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return None

    def handle_command(self, cmd: str) -> None:
        """
        Handle REPL commands.
        
        Args:
            cmd: The command to execute
        """
        cmd = cmd.strip()
        if cmd == 'help':
            self.print_help()
            return
        elif cmd == 'clear':
            self.clear()
            self.variables.clear()
            return
        elif cmd == 'reset':
            self.__init__()
            print("Environment reset")
            return
        elif cmd == 'vars':
            self.print_variables()
            return
        elif cmd.startswith('help '):
            self.print_topic_help(cmd[5:])
            return
        elif cmd == 'debug':
            self.debug_mode = not self.debug_mode
            print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
            return
        elif cmd.startswith('watch '):
            var = cmd[6:].strip()
            self.watches.add(var)
            print(f"Watching variable: {var}")
            return
        elif cmd == 'profile':
            self.profiling = not self.profiling
            print(f"Profiling {'enabled' if self.profiling else 'disabled'}")
            return
        elif cmd.startswith('summary'):
            parts = cmd.split()
            model_name = parts[1] if len(parts) > 1 else None
            self.do_summary(model_name)
            return
        
        print(f"Unknown command: {cmd}")

    def print_help(self) -> None:
        """Print general help information."""
        help_text = """
Available commands:
------------------
help          : Show this help message
help <topic>  : Show help for a specific topic
clear         : Clear the screen and reset environment
vars          : List all variables
reset         : Reset the environment
exit          : Exit the REPL

Code Execution:
--------------
- Type NEURO code directly to execute
- Multi-line code blocks are supported
- Use '=' for assignment
- Access previous results with '_'

Debugging:
---------
debug        : Toggle debug mode
watch        : Add variables to watch
profile     : Toggle profiling
"""
        print(help_text)

    def print_variables(self) -> None:
        """Print all defined variables."""
        if not self.variables:
            print("No variables defined")
            return
        
        rows = []
        for name, value in self.variables.items():
            type_name = type(value).__name__
            if isinstance(value, (int, float, str, bool)):
                summary = str(value)
            else:
                summary = str(type(value))
            rows.append([name, type_name, summary])
        
        print(tabulate(rows, headers=['Name', 'Type', 'Value'], tablefmt='grid'))

    def print_topic_help(self, topic: str) -> None:
        """
        Print help for a specific topic.
        
        Args:
            topic: The topic to get help for
        """
        topics = {
            'Dense': """
Dense:
Dense layer for fully connected neural networks

Parameters:
- units: Number of output units
- activation: Activation function (relu, sigmoid, tanh)
- input_size: Size of input (optional)
""",
            'Conv2D': """
Conv2D:
2D Convolutional layer

Parameters:
- filters: Number of output filters
- kernel_size: Size of convolution kernel
- stride: Stride of convolution
- padding: Padding type
""",
            'custom_layer': """
Custom Layer:
Define custom neural network layers

Usage:
@custom_layer
def LayerName(x) {
    // Layer implementation
}
"""
        }
        
        if topic in topics:
            print(topics[topic])
        else:
            print(f"No help available for: {topic}")

    def clear(self) -> None:
        """Clear the screen."""
        if sys.platform == 'win32':
            os.system('cls')
        else:
            os.system('clear')

    def default(self, line: str) -> bool:
        """Handle NEURO code execution."""
        try:
            self.last_result = self.interpreter.interpret(line)
            if self.last_result is not None:
                print(self.last_result)
            return False
        except NeuroError as e:
            print(f"Error: {str(e)}")
            return False

    def do_summary(self, arg: str) -> bool:
        """Display model summary."""
        try:
            model = self._get_model(arg)
            if model:
                summary = self._get_model_summary(model)
                print(summary)
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
        return False

    def do_visualize(self, arg: str) -> bool:
        """Generate and display model architecture diagram."""
        try:
            model = self._get_model(arg)
            if model:
                dot = self._create_model_graph(model)
                dot.render("model_architecture", view=True, format="png")
                print("Model visualization saved as 'model_architecture.png'")
        except Exception as e:
            print(f"Error visualizing model: {str(e)}")
        return False

    def do_debug(self, arg: str) -> bool:
        """Toggle debug mode with specified options."""
        try:
            if not arg:
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                return False

            options = eval(f"dict({arg})")
            self.watches = options.get('watch', [])
            self.break_conditions = options.get('break_on', [])
            self.profiling = options.get('profile', False)
            self.debug_mode = True
            
            print(f"Debug mode ON with options:")
            print(f"Watching: {', '.join(self.watches)}")
            print(f"Break conditions: {', '.join(self.break_conditions)}")
            print(f"Profiling: {'ON' if self.profiling else 'OFF'}")
        except Exception as e:
            print(f"Error setting debug options: {str(e)}")
        return False

    def do_watch(self, arg: str) -> bool:
        """Add or remove watch variables."""
        if not arg:
            print(f"Currently watching: {', '.join(self.watches)}")
            return False
        
        try:
            items = [item.strip() for item in arg.split(',')]
            self.watches.extend(items)
            print(f"Added watches for: {', '.join(items)}")
        except Exception as e:
            print(f"Error setting watches: {str(e)}")
        return False

    def do_profile(self, arg: str) -> bool:
        """Toggle profiling mode."""
        self.profiling = not self.profiling
        print(f"Profiling: {'ON' if self.profiling else 'OFF'}")
        return False

    def do_exit(self, arg: str) -> bool:
        """Exit the REPL."""
        print("Goodbye!")
        return True

    def _get_model(self, name: str) -> Optional[nn.Module]:
        """Get model by name from interpreter variables."""
        if not name:
            # Try to find the last created model
            for var_name, var in reversed(list(self.interpreter.variables.items())):
                if isinstance(var, nn.Module):
                    return var
            print("No model found. Please specify a model name.")
            return None
        
        model = self.interpreter.variables.get(name)
        if not isinstance(model, nn.Module):
            print(f"'{name}' is not a model.")
            return None
        return model

    def _get_model_summary(self, model: nn.Module) -> str:
        """Generate a formatted model summary."""
        def count_parameters(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        rows = []
        total_params = 0
        
        for name, layer in model.named_children():
            params = count_parameters(layer)
            total_params += params
            
            # Get output shape if possible
            try:
                if hasattr(layer, 'out_features'):
                    output_shape = f"({layer.out_features},)"
                elif hasattr(layer, 'out_channels'):
                    output_shape = f"(N, {layer.out_channels}, H, W)"
                else:
                    output_shape = "?"
            except:
                output_shape = "?"
            
            rows.append([name, layer.__class__.__name__, output_shape, params])

        summary = "Model Summary\n" + "=" * 50 + "\n"
        summary += tabulate(rows, headers=["Layer", "Type", "Output Shape", "Param #"], tablefmt="grid")
        summary += f"\nTotal parameters: {total_params:,}"
        return summary

    def _create_model_graph(self, model: nn.Module) -> graphviz.Digraph:
        """Create a visual representation of the model architecture."""
        dot = graphviz.Digraph(comment='Model Architecture')
        dot.attr(rankdir='TB')
        
        def add_layer(layer: nn.Module, parent: str = None) -> str:
            layer_id = str(id(layer))
            label = f"{layer.__class__.__name__}"
            
            if hasattr(layer, 'out_features'):
                label += f"\n({layer.out_features})"
            elif hasattr(layer, 'out_channels'):
                label += f"\n({layer.out_channels})"
            
            dot.node(layer_id, label)
            if parent:
                dot.edge(parent, layer_id)
            return layer_id
        
        def build_graph(module: nn.Module, parent: str = None):
            for name, layer in module.named_children():
                layer_id = add_layer(layer, parent)
                build_graph(layer, layer_id)
        
        build_graph(model)
        return dot

    def _print_watched_vars(self) -> None:
        """Print watched variables."""
        for var in self.watches:
            if var in self.variables:
                print(f"{var} = {self.variables[var]}")

    def run(self) -> None:
        """Run the REPL."""
        print("NEURO REPL v1.0")
        print('Type "help" for more information.')
        
        while True:
            try:
                code = input(">>> ")
                if code.strip() == "exit":
                    break
                elif code.startswith(":"):
                    self.handle_command(code[1:].strip())
                else:
                    result = self.handle_input(code)
                    if result is not None:
                        print(result)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
            except EOFError:
                break
        
        print("\nGoodbye!")

def main():
    """Start the NEURO REPL."""
    try:
        NeuroREPL().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == '__main__':
    main() 