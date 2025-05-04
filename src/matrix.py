# NEURO Matrix: Handles the .nrm data format

import yaml
from src.errors import NeuroDataError # Assuming a relevant error type exists or needs creation
import os # Needed for file existence check
import random # For shuffling data
import math # For calculating split indices

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

    def split(self, train_frac=0.7, val_frac=0.15, test_frac=0.15, shuffle=True, random_state=None):
        """Splits the data into train, validation, and test sets.

        Args:
            train_frac (float): Fraction of data for the training set.
            val_frac (float): Fraction of data for the validation set.
            test_frac (float): Fraction of data for the test set.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (int, optional): Seed for the random number generator for reproducible shuffles.

        Returns:
            tuple[NeuroMatrix, NeuroMatrix, NeuroMatrix]: A tuple containing the 
                train, validation, and test NeuroMatrix objects. If a fraction is 0, 
                the corresponding object will be None.

        Raises:
            ValueError: If the fractions do not sum to 1.0 or are invalid.
        """
        if not math.isclose(train_frac + val_frac + test_frac, 1.0):
            raise ValueError("Fractions for train, validation, and test sets must sum to 1.0")
        if not (0 <= train_frac <= 1 and 0 <= val_frac <= 1 and 0 <= test_frac <= 1):
            raise ValueError("Fractions must be between 0 and 1")

        num_samples = len(self.data)
        indices = list(range(num_samples))

        if shuffle:
            if random_state is not None:
                random.seed(random_state)
            random.shuffle(indices)

        train_end = math.ceil(num_samples * train_frac)
        val_end = train_end + math.ceil(num_samples * val_frac)
        # test_end is implicitly num_samples

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        print(f"Splitting {num_samples} samples: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        def create_subset_matrix(subset_indices):
            if not subset_indices:
                return None # Return None if the subset is empty
            subset_matrix = NeuroMatrix()
            subset_matrix.metadata = self.metadata.copy() # Copy metadata
            subset_matrix.metadata['split_from'] = self.metadata.get('name', 'original') # Add info about origin
            subset_matrix.data = [self.data[i] for i in subset_indices]
            return subset_matrix

        train_set = create_subset_matrix(train_indices)
        val_set = create_subset_matrix(val_indices)
        test_set = create_subset_matrix(test_indices)

        return train_set, val_set, test_set

# Potential: Add NeuroDataError to src/errors.py if it doesn't exist
# from src.errors import NeuroError
# class NeuroDataError(NeuroError):
#     """Custom exception for NeuroMatrix data loading/validation errors."""
#     pass 