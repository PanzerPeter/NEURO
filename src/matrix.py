# NEURO Matrix: Handles the .nrm data format

import yaml
from src.errors import NeuroDataError # Assuming a relevant error type exists or needs creation
import os # Needed for file existence check
import random # For shuffling data
import math # For calculating split indices
import numpy as np # Import numpy for numerical operations
import statistics # For median calculation

class NeuroMatrix:
    """Handles the .nrm data format, expected to be YAML."""
    def __init__(self):
        self.metadata = {}
        self.data = [] # List of dictionaries, e.g., [{'input': [...], 'output': [...]}]
        self.feature_info = {} # Store parsed feature info (names, types)
        self.input_features = [] # List of input feature names
        self.output_features = [] # List of output feature names
        self.missing_value_token = None # Store the token used for missing values

    def load(self, filepath):
        """Loads data from a .nrm (YAML) file, parsing the new structure.

        Standard YAML NaN representations (.nan, .NaN, .NAN) will be loaded
        as float('nan') and handled separately from the user-defined
        missing_value_token, which is converted to None internally.
        """
        print(f"Attempting to load NeuroMatrix from: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NeuroMatrix file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f: # Specify encoding
                loaded_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise NeuroDataError(f"Error parsing NeuroMatrix YAML file '{filepath}': {e}")
        except Exception as e:
            raise NeuroDataError(f"An unexpected error occurred while reading '{filepath}': {e}")

        if not isinstance(loaded_data, dict):
            raise NeuroDataError(f"Invalid NeuroMatrix format in '{filepath}': Top level must be a dictionary.")

        # --- Enhanced Validation and Metadata Parsing ---
        if 'metadata' not in loaded_data or not isinstance(loaded_data['metadata'], dict):
            raise NeuroDataError(f"Invalid NeuroMatrix format in '{filepath}': Missing or invalid 'metadata' section.")
        if 'data' not in loaded_data or not isinstance(loaded_data['data'], list):
            raise NeuroDataError(f"Invalid NeuroMatrix format in '{filepath}': Missing or invalid 'data' section (must be a list).")

        self.metadata = loaded_data['metadata']
        self.missing_value_token = self.metadata.get('missing_value_token', None) # Default to None

        # Parse feature information
        self.feature_info = self.metadata.get('features', {}) # Get features dict, default to empty
        self.input_features = []
        self.output_features = []
        expected_input_len = 0
        expected_output_len = 0

        if isinstance(self.feature_info.get('input'), list):
            for i, feature in enumerate(self.feature_info['input']):
                if not isinstance(feature, dict) or 'name' not in feature or 'type' not in feature:
                    raise NeuroDataError(f"Invalid feature definition in metadata.features.input[{i}] in '{filepath}'. Expected dict with 'name' and 'type'.")
                self.input_features.append(feature['name'])
            expected_input_len = len(self.input_features)
        # else: Consider warning or error if features.input is missing/invalid?

        if isinstance(self.feature_info.get('output'), list):
            for i, feature in enumerate(self.feature_info['output']):
                if not isinstance(feature, dict) or 'name' not in feature or 'type' not in feature:
                    raise NeuroDataError(f"Invalid feature definition in metadata.features.output[{i}] in '{filepath}'. Expected dict with 'name' and 'type'.")
                self.output_features.append(feature['name'])
            expected_output_len = len(self.output_features)
        # else: Consider warning or error if features.output is missing/invalid?

        # --- Data Validation Against Metadata ---
        raw_data = loaded_data['data']
        processed_data = []
        for i, item in enumerate(raw_data):
            if not isinstance(item, dict) or 'input' not in item or 'output' not in item:
                raise NeuroDataError(f"Invalid data point format at index {i} in '{filepath}': Each item must be a dict with 'input' and 'output' keys.")

            input_list = item['input']
            output_list = item['output']

            # Validate lengths if features are defined
            if expected_input_len > 0 and len(input_list) != expected_input_len:
                raise NeuroDataError(f"Data point {i} in '{filepath}': Input length {len(input_list)} does not match expected input features length {expected_input_len}.")
            if expected_output_len > 0 and len(output_list) != expected_output_len:
                 raise NeuroDataError(f"Data point {i} in '{filepath}': Output length {len(output_list)} does not match expected output features length {expected_output_len}.")

            # Convert missing tokens to None internally for consistent handling
            processed_input = [None if x == self.missing_value_token else x for x in input_list]
            processed_output = [None if x == self.missing_value_token else x for x in output_list]
            processed_data.append({'input': processed_input, 'output': processed_output})

        self.data = processed_data

        print(f"NeuroMatrix loaded successfully from '{filepath}'. Features: Input={len(self.input_features)}, Output={len(self.output_features)}. Data points: {len(self.data)}")
        # Optional: Convert self.data to NumPy arrays here for efficiency if needed
        # self._convert_to_numpy()

    def add_data_point(self, input_data, output_data):
        """Adds a single data point. Validates against existing feature lengths if defined."""
        if self.input_features and len(input_data) != len(self.input_features):
            raise ValueError(f"Input data length {len(input_data)} does not match defined input features length {len(self.input_features)}.")
        if self.output_features and len(output_data) != len(self.output_features):
            raise ValueError(f"Output data length {len(output_data)} does not match defined output features length {len(self.output_features)}.")

        # Convert missing tokens if they match the one defined in metadata
        processed_input = [None if x == self.missing_value_token else x for x in input_data]
        processed_output = [None if x == self.missing_value_token else x for x in output_data]

        self.data.append({'input': processed_input, 'output': processed_output})
        # Note: Adding data points after potential numpy conversion would require reconversion or appending differently
        print(f"Added data point: input={input_data}, output={output_data}")

    def __len__(self):
        return len(self.data)

    def __str__(self):
        # Improved string representation
        metadata_str = f"Name: {self.metadata.get('name', 'N/A')}, Input Features: {len(self.input_features)}, Output Features: {len(self.output_features)}"
        return f"NeuroMatrix({metadata_str}, Data Points: {len(self.data)})"

    def get_column(self, column_name, data_type='input', return_type=list):
        """Extracts a single column (feature) from the data.

        Args:
            column_name (str): The name of the feature to extract.
            data_type (str): 'input' or 'output' to specify where the feature resides.
            return_type (type): The type to return (list or np.ndarray). 
                              For np.ndarray, both None and float('nan') in the
                              original data are converted to np.nan, and the dtype
                              is appropriately set (usually float if NaNs are present).

        Returns:
            list | np.ndarray: A list or NumPy array containing the values.

        Raises:
            ValueError: If the column_name is not found or return_type is invalid.
        """
        if data_type not in ['input', 'output']:
            raise ValueError("data_type must be 'input' or 'output'")
        if return_type not in [list, np.ndarray]:
             raise ValueError("return_type must be list or np.ndarray")

        feature_list = self.input_features if data_type == 'input' else self.output_features
        try:
            col_index = feature_list.index(column_name)
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found in {data_type} features.")

        column_data = [point[data_type][col_index] for point in self.data]

        if return_type is np.ndarray:
            # Determine appropriate dtype, considering None and float('nan')
            valid_numeric_values = [x for x in column_data if x is not None and not (isinstance(x, float) and math.isnan(x))]
            has_none = any(x is None for x in column_data)
            has_nan = any(isinstance(x, float) and math.isnan(x) for x in column_data)

            dtype = object # Default to object

            if valid_numeric_values: # If there are any actual numbers
                 try:
                     result_dtype = np.result_type(*valid_numeric_values)
                     if np.issubdtype(result_dtype, np.number):
                         if has_nan or (has_none and np.issubdtype(result_dtype, np.integer)):
                             dtype = np.float64 # Promote to float if NaN present or if None + integer
                         else:
                              dtype = result_dtype
                 except TypeError:
                      pass # Stick with object if result_type fails

            elif has_nan or has_none: # Only None or NaN present
                 dtype = np.float64 # Use float for np.nan representation

            # Convert None and float('nan') to np.nan for numeric arrays
            if np.issubdtype(dtype, np.number):
                 processed_column = [np.nan if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in column_data]
                 return np.array(processed_column, dtype=dtype)
            else:
                 # For object arrays, keep None and original float('nan') as is
                 return np.array(column_data, dtype=object)
        else:
            return column_data # Return as plain list

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
                return None
            subset_matrix = NeuroMatrix()
            subset_matrix.metadata = self.metadata.copy()
            subset_matrix.metadata['split_from'] = self.metadata.get('name', 'original')
            subset_matrix.data = [self.data[i] for i in subset_indices]
            # --- Copy other relevant attributes --- 
            subset_matrix.feature_info = self.feature_info # Shallow copy ok?
            subset_matrix.input_features = self.input_features
            subset_matrix.output_features = self.output_features
            subset_matrix.missing_value_token = self.missing_value_token
            # -------------------------------------
            return subset_matrix

        train_set = create_subset_matrix(train_indices)
        val_set = create_subset_matrix(val_indices)
        test_set = create_subset_matrix(test_indices)

        return train_set, val_set, test_set

    def normalize(self, method='minmax', columns=None, data_type='input'):
        """Normalizes numerical data columns.

        Modifies data in-place.

        Args:
            method (str): 'minmax' or 'zscore'.
            columns (list[str], optional): List of column names to normalize. 
                                          Defaults to all numeric columns in data_type.
            data_type (str): 'input' or 'output'. Defaults to 'input'.
        """
        print(f"Applying {method} normalization to {data_type} data...")
        if not self.data:
            print("Warning: No data to normalize.")
            return

        feature_defs = self.feature_info.get(data_type, [])
        feature_names = self.input_features if data_type == 'input' else self.output_features

        if not feature_names:
             print(f"Warning: No {data_type} features defined to normalize.")
             return

        cols_to_normalize = []
        if columns is None:
            # Default: find all numeric columns
            for i, name in enumerate(feature_names):
                # Find corresponding feature definition
                f_def = next((f for f in feature_defs if f.get('name') == name), None)
                if f_def and f_def.get('type') == 'numeric':
                    cols_to_normalize.append(name)
        else:
            # Use provided list, check if they exist and are numeric
            for name in columns:
                 if name not in feature_names:
                     raise ValueError(f"Column '{name}' not found in {data_type} features.")
                 f_def = next((f for f in feature_defs if f.get('name') == name), None)
                 if not f_def or f_def.get('type') != 'numeric':
                      raise ValueError(f"Column '{name}' is not defined as numeric in {data_type} features.")
                 cols_to_normalize.append(name)

        if not cols_to_normalize:
            print(f"Warning: No numeric columns found or specified to normalize in {data_type}.")
            return

        print(f"Normalizing columns: {cols_to_normalize}")

        for col_name in cols_to_normalize:
            col_index = feature_names.index(col_name)
            # Extract column as numpy array with nan
            col_data_np = self.get_column(col_name, data_type, return_type=np.ndarray)

            # Check if data is numeric before proceeding
            if not np.issubdtype(col_data_np.dtype, np.number):
                 print(f"Warning: Column '{col_name}' is not numeric. Skipping normalization.")
                 continue

            # Check for all NaNs
            if np.all(np.isnan(col_data_np)):
                 print(f"Warning: Column '{col_name}' contains only missing values. Skipping normalization.")
                 continue

            if method == 'minmax':
                min_val = np.nanmin(col_data_np)
                max_val = np.nanmax(col_data_np)
                range_val = max_val - min_val
                if range_val == 0 or np.isnan(range_val):
                    print(f"Warning: Column '{col_name}' has zero range or contains only NaNs after filtering. Skipping minmax normalization.")
                    continue
                # Apply normalization using numpy vectorized ops where possible
                normalized_col = (col_data_np - min_val) / range_val

            elif method == 'zscore':
                mean_val = np.nanmean(col_data_np)
                std_dev = np.nanstd(col_data_np)
                if std_dev == 0 or np.isnan(std_dev):
                     print(f"Warning: Column '{col_name}' has zero standard deviation or contains only NaNs. Setting z-score to 0 for non-NaN values.")
                     # Set non-NaN values to 0.0, keep NaNs as NaN
                     normalized_col = np.where(np.isnan(col_data_np), np.nan, 0.0)
                else:
                    normalized_col = (col_data_np - mean_val) / std_dev
            else:
                raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'.")

            # Update the original self.data list
            # Overwrite original values unless they were the missing token (None)
            # NaN values in the original data will be updated with NaN results from normalization
            for i, point in enumerate(self.data):
                original_value = point[data_type][col_index]
                if original_value is not None: # Skip if it was the missing token
                     # Assign the value from the normalized numpy array
                     point[data_type][col_index] = normalized_col[i]
            # ---------------------------------------------

        print(f"Normalization ({method}) applied successfully to {data_type} columns: {cols_to_normalize}.")
        # Update metadata more specifically
        norm_info = self.metadata.get('normalization', [])
        if not isinstance(norm_info, list): norm_info = [norm_info] # Ensure list
        norm_info.append({ 'method': method, 'columns': cols_to_normalize, 'type': data_type })
        self.metadata['normalization'] = norm_info

    def handle_missing(self, strategy='mean', columns=None, data_type='input', value=None):
        """Handles missing values (None) in specified columns.

        Modifies data in-place. Handles values that are internally represented as None
        (originating from the 'missing_value_token' in the metadata). It does not
        automatically process float('nan') values unless 'nan' (or similar) was
        explicitly set as the 'missing_value_token'.

        Args:
            strategy (str): 'mean', 'median', 'remove' (row), 'constant'.
            columns (list[str], optional): List of column names to process.
                                          Defaults to all columns in data_type.
            data_type (str): 'input' or 'output'. Defaults to 'input'.
            value (float/int/str, optional): Value for 'constant' strategy.
        """
        print(f"Handling missing values ({strategy}) in {data_type} data...")
        if not self.data:
            print("Warning: No data for missing value handling.")
            return

        if strategy == 'constant' and value is None:
             raise ValueError("A 'value' must be provided for 'constant' strategy.")
        if strategy not in ['mean', 'median', 'remove', 'constant']:
             raise ValueError(f"Unknown missing value strategy: {strategy}. Use 'mean', 'median', 'remove', or 'constant'.")

        feature_names = self.input_features if data_type == 'input' else self.output_features
        if not feature_names:
             print(f"Warning: No {data_type} features defined to handle missing values.")
             return

        cols_to_process = []
        if columns is None:
            cols_to_process = feature_names # Default to all columns
        else:
            # Use provided list, check if they exist
            for name in columns:
                 if name not in feature_names:
                     raise ValueError(f"Column '{name}' not found in {data_type} features.")
                 cols_to_process.append(name)

        if not cols_to_process:
            print(f"Warning: No columns specified or found to process for missing values in {data_type}.")
            return
            
        print(f"Processing missing values in columns: {cols_to_process}")

        if strategy == 'remove':
            original_count = len(self.data)
            new_data = []
            indices_to_check = [feature_names.index(name) for name in cols_to_process]
            for point in self.data:
                has_missing = False
                for idx in indices_to_check:
                    if point[data_type][idx] is None: # Still check for None here
                        has_missing = True
                        break
                if not has_missing:
                    new_data.append(point)
            removed_count = original_count - len(new_data)
            if removed_count > 0:
                print(f"Removed {removed_count} rows containing missing values in specified {data_type} columns.")
                self.data = new_data
            else:
                 print(f"No rows removed. No missing values found in specified {data_type} columns.")

        else: # Handle 'mean', 'median', 'constant'
            for col_name in cols_to_process:
                col_index = feature_names.index(col_name)
                replacement_value = None

                if strategy == 'constant':
                    replacement_value = value
                else: # 'mean' or 'median'
                    feature_defs = self.feature_info.get(data_type, [])
                    f_def = next((f for f in feature_defs if f.get('name') == col_name), None)
                    if not f_def or f_def.get('type') != 'numeric':
                        print(f"Warning: Skipping '{strategy}' imputation for non-numeric column '{col_name}'.")
                        continue

                    # Extract column as numpy array with nan
                    col_data_np = self.get_column(col_name, data_type, return_type=np.ndarray)
                    
                    if np.all(np.isnan(col_data_np)):
                        print(f"Warning: Column '{col_name}' contains only missing values. Cannot calculate {strategy}. Skipping imputation.")
                        continue

                    if strategy == 'mean':
                        replacement_value = np.nanmean(col_data_np)
                    elif strategy == 'median':
                        replacement_value = np.nanmedian(col_data_np) # Use nanmedian

                # Apply the replacement value (where original was None)
                if replacement_value is not None: # Check replacement was calculated/provided
                    filled_count = 0
                    for point in self.data:
                        if point[data_type][col_index] is None: # Find original Nones
                            point[data_type][col_index] = replacement_value
                            filled_count += 1
                    if filled_count > 0:
                         rep_val_str = f"{replacement_value:.4f}" if isinstance(replacement_value, (int, float)) else str(replacement_value)
                         print(f"Filled {filled_count} missing values in column '{col_name}' using {strategy} (value={rep_val_str}).")
                elif strategy in ['mean', 'median']:
                     print(f"No missing values found or filled in column '{col_name}'.")

        print(f"Missing values handled ({strategy}) successfully for {data_type} columns: {cols_to_process}.")
        # Update metadata more specifically
        missing_info = self.metadata.get('missing_handled', [])
        if not isinstance(missing_info, list): missing_info = [missing_info] # Ensure list
        missing_info.append({ 'strategy': strategy, 'columns': cols_to_process, 'type': data_type })
        self.metadata['missing_handled'] = missing_info