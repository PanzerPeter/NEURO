"""
NEURO REPL Module
Provides interactive development environment for NEURO.
"""

import cmd
import sys
import torch
from typing import Optional, Dict, Any
from .interpreter import NeuroInterpreter
from .errors import NeuroError
from tabulate import tabulate
import torch.nn as nn
import graphviz

class NeuroREPL(cmd.Cmd):
    intro = 'Welcome to NEURO REPL. Type help or ? to list commands.'
    prompt = '>>> '

    def __init__(self):
        super().__init__()
        self.interpreter = NeuroInterpreter()
        self.last_result = None
        self.debug_mode = False
        self.watches = []
        self.break_conditions = []
        self.profiling = False

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

        headers = ["Layer", "Type", "Output Shape", "Param #"]
        summary = tabulate(rows, headers=headers, tablefmt="grid")
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

def main():
    """Start the NEURO REPL."""
    try:
        NeuroREPL().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == '__main__':
    main() 