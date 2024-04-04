# Imports
import numpy as np
import pandas as pd

from types import Optional

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Constants
VALID_PREPROCESS_MODES = ('naive', 'one-hot', 'one-hot-pca', 'extended')
DIFFICULTY_PARAMETERS = {
    'easy':None,
    'medium':None,
    'hard':None
}

class BenchmarkMethods:
    def __init__(self,
                 simulate_data: bool,           # Whether to simulate or not
                 k_clusters: int,               # Number of clusters
                 approach: str,
                 # For a given dataset:
                 given_data: Optional[pd.DataFrame],    # A given pd.df
                 gd_num_indices: Optional[list[int]],   # Num idxs for df
                 gd_cat_indices: Optional[list[int]],   # Cat idxs for df
                 gd_multi_indices: Optional[list[int]], # Multi idxs for df
                 # For a simulation:
                 sim_n_observations: Optional[int],
                 simulation_difficulty: Optional[str]) -> None:
        
        self.k_clusters = k_clusters
        
        # Check arg value before assignment
        if approach not in VALID_PREPROCESS_MODES:
            raise ValueError(
                f"Argument approach must be one of {VALID_PREPROCESS_MODES}")
        self.approach = approach
        
        # This logic below helps enforce certain parameters according to the
        # simulate_data argument.
        if not isinstance(simulate_data, bool):
            raise ValueError(f'Argument simulate_data must be a boolean.')

        if simulate_data:
            self.simulate_data = True
            self.data = None
            
            if not isinstance(sim_n_observations, int):
                raise ValueError("Number of observations for simulation" 
                                 "must be an integer.")
            self.sim_n_observations = sim_n_observations

            if simulation_difficulty not in DIFFICULTY_PARAMETERS.keys():
                raise ValueError(
                    f"Argument simulation_difficulty must be one of"
                    f"{list(DIFFICULTY_PARAMETERS.keys())}."
                )
            self.sim_difficulty = simulation_difficulty
            self.num_indices = None,
            self.cat_indices = None,
            self.multi_indices = None
        
        else:   # If given data:
            self.simulate_data = False
            if given_data is None:
                raise ValueError(
                    "If simulate_data is False, a dataset must be provided")
            
            if k_clusters >= given_data.shape[0]:
                raise ValueError("Argument k_clusters must be less than"
                                 "the number of observations in the given data")

            self.data = given_data
            self.sim_n_observations = None
            self.sim_difficulty = None
            
            for arg in [gd_num_indices, gd_cat_indices, gd_multi_indices]:
                if not all(isinstance(n, int) for n in arg):
                    raise TypeError(f"Argument {arg} contains non-int values.")
            self.num_indices = gd_num_indices,
            self.cat_indices = gd_cat_indices,
            self.multi_indices = gd_multi_indices

        self.numerical_encoder = None
        self.categorical_encoders = dict()
        self.dummy_indices = None

    def benchmark(self,
                  num_cols: list[str],
                  cat_cols: list[str],
                  multi_cols: list[str],
                  target: str):
        if self.simulate_data:
            # Generate a dataset
            pass
        # Apply preprocessing

        # Apply encoding

        # Join matrix and define column indexes for each attribute type

        # Call clustering according to approach 
    
    def preprocess(self):
        
        dataset = self.data
        num_cols = self.num_indices
        cat_cols = self.cat_indices
        multi_cols: self.multi_indices
        approach = self.approach
        
        # Check arguments
        for arg in [num_cols, cat_cols, multi_cols]:
            if isinstance(arg, list):
                if not all(isinstance(n, int) for n in arg):
                    raise TypeError(f"Argument {arg} contains non-int values.")
            else:
                raise TypeError(f"Argument {arg} must be a list of integers.")
        
        # Segregate the datasets
        X_num = dataset.iloc[:, num_cols].copy()
        X_cat = dataset.iloc[:, cat_cols].copy()
        X_multi = dataset.iloc[:, multi_cols].copy()

        # Numerical variables receive MinMax scaling
        num_scaler = MinMaxScaler()
        X_num.iloc[:,:] = num_scaler.fit_transform(X_num.values)
        
        self.numerical_encoder = num_scaler
        
        # Categorical variables are labeled as integers
        for i in range(X_cat.shape[1]):
            idx = cat_cols[i]
            
            cat_encoder = LabelEncoder()
            X_cat.iloc[:,i] = cat_encoder.fit_transform(X_cat.iloc[:,i])
            
            self.categorical_encoders[idx] = cat_encoder

        # Process multi-valued columns according to approach
        if approach == 'naive':
            # Set multi-valued as text and apply categorical encoding
            for i in range(X_multi.shape[1]):
                idx = multi_cols[i]

                cat_encoder = LabelEncoder()
                X_multi.iloc[:,i] = cat_encoder.fit_transform(
                    X_multi.iloc[:,i].astype(str)
                )

                self.categorical_encoders[idx] = cat_encoder
            
            # Return matrices
            return X_num, X_cat, X_multi

        elif approach in ('one-hot', 'one-hot-pca'):
            # Steps
            
            if approach == 'one-hot-pca':
                pass # Extra steps
            
            # Consolidate all matrices into input_matrix X.
            X = pd.concat([X_num, X_cat, X_multi])
            num_index = 
            cat_index = 
        elif approach == 'extended':
            # Steps

            # Consolidate all matrices into input_matrix X.
            X = pd.concat([X_num, X_cat, X_multi])
            num_index = 
            cat_index = 
            multi_cols_index = 