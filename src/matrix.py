# NEURO Matrix: Handles the .nrm data format

import yaml
from src.errors import NeuroDataError # Assuming a relevant error type exists or needs creation
import os # Needed for file existence check

class NeuroMatrix:
    """Handles the .nrm data format, expected to be YAML."""
    def __init__(self):
        self.metadata = {}
        self.data = [] # List of dictionaries, e.g., [{'input': [...], 'output': [...]}]
        # self.model_config = {} # Kept for potential future use

    def load(self, filepath):
        """Loads data from a .nrm (YAML) file."""
        print(f"Attempting to load NeuroMatrix from: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NeuroMatrix file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                loaded_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise NeuroDataError(f"Error parsing NeuroMatrix YAML file '{filepath}': {e}")
        except Exception as e:
            raise NeuroDataError(f"An unexpected error occurred while reading '{filepath}': {e}")

        if not isinstance(loaded_data, dict):
            raise NeuroDataError(f"Invalid NeuroMatrix format in '{filepath}': Top level must be a dictionary.")

        # Basic validation of structure
        if 'metadata' not in loaded_data or not isinstance(loaded_data['metadata'], dict):
            raise NeuroDataError(f"Invalid NeuroMatrix format in '{filepath}': Missing or invalid 'metadata' section.")
        if 'data' not in loaded_data or not isinstance(loaded_data['data'], list):
            raise NeuroDataError(f"Invalid NeuroMatrix format in '{filepath}': Missing or invalid 'data' section (must be a list).")

        # Validate individual data points (simple check for input/output keys)
        for i, item in enumerate(loaded_data['data']):
            if not isinstance(item, dict) or 'input' not in item or 'output' not in item:
                raise NeuroDataError(f"Invalid data point format at index {i} in '{filepath}': Each item must be a dict with 'input' and 'output' keys.")

        self.metadata = loaded_data['metadata']
        self.data = loaded_data['data']

        print(f"NeuroMatrix loaded successfully from '{filepath}'. Metadata: {self.metadata}, Data points: {len(self.data)}")

    def add_data_point(self, input_data, output_data):
        """Adds a single data point."""
        # Basic type validation could be added here
        self.data.append({'input': input_data, 'output': output_data})
        print(f"Added data point: input={input_data}, output={output_data}")

    def __len__(self):
        return len(self.data)

    def __str__(self):
        # Limit the string representation if data is large?
        return f"NeuroMatrix(metadata={self.metadata}, data_points={len(self.data)})"

# Potential: Add NeuroDataError to src/errors.py if it doesn't exist
# from src.errors import NeuroError
# class NeuroDataError(NeuroError):
#     """Custom exception for NeuroMatrix data loading/validation errors."""
#     pass 