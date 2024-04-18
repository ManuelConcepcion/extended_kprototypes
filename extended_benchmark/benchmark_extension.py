# Imports
import datetime as dt
import numpy as np
import pandas as pd

from types import Optional

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

# Constants
VALID_PREPROCESS_MODES = ('naive', 'one-hot', 'one-hot-pca', 'extended')
DIFFICULTY_PARAMETERS = {
    'easy': None,
    'medium': None,
    'hard': None
}
MINIMUM_N_TO_D_RATIO = 5

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

        self.preprocessing_time = 0.0
        self.clustering_time = 0.0
        self.evaluation_time = 0.0

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
    
    def categorical_encoding(self, df, cols):
        for i in range(df.shape[1]):
            idx = cols[i]
            
            cat_encoder = LabelEncoder()
            df.iloc[:,i] = cat_encoder.fit_transform(df.iloc[:,i])
            
            self.categorical_encoders[idx] = cat_encoder
        
        return df

    def preprocess(self):
        time_start = dt.datetime.now()

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
        x_num = dataset.iloc[:, num_cols].copy()
        x_cat = dataset.iloc[:, cat_cols].copy()
        x_multi = dataset.iloc[:, multi_cols].copy()

        # Numerical variables receive MinMax scaling
        num_scaler = MinMaxScaler()
        x_num.iloc[:,:] = num_scaler.fit_transform(x_num.values)
        
        self.numerical_encoder = num_scaler
        
        # Categorical variables are labeled as integers
        x_cat = self.categorical_encoding(x_cat, cat_cols)

        # Process multi-valued columns according to approach
        if approach == 'naive':
            # Set multi-valued as text and apply categorical encoding
            x_multi = self.categorical_encoding(x_multi, multi_cols)
            
            # Return matrices
            self.preprocessing_time = (
                dt.datetime.now() - time_start).total_seconds()
            return x_num, x_cat, x_multi

        elif approach in ('one-hot', 'one-hot-pca'):
            # Unpack the multi-valued dataset
            columns_to_concat = []
            for i in range(x_multi.shape[1]):
                dummy_df = pd.get_dummies(
                        x_multi.iloc[:,i].apply(pd.Series).stack()
                    ).groupby(
                        'index', level=0
                    ).sum()
                columns_to_concat.append(dummy_df)
            x_dummies = pd.concat(columns_to_concat, axis=1)

            # If there are too many dummies, cut down least frequent
            if x_dummies.shape[0]/x_dummies.shape[1] <= MINIMUM_N_TO_D_RATIO:
                frequencies = pd.DataFrame(
                        x_dummies.mean(), columns=['freq']
                    ).sort_values('freq', ascending=False)
                
                saved_dummies = list(
                    frequencies.iloc[:int(x_dummies.shape[0]/5)].index)
                x_dummies = x_dummies.loc[:,saved_dummies]

            if approach == 'one-hot-pca':
                # Apply MLE to set number of components automatically
                
                pca = PCA(
                    n_components='mle',
                    whiten=True
                    # Vectors are scaled to unit variance so that we can use 
                    # them as numeric attributes.    
                )  

                x_dummies = pca.fit_transform(x_dummies)
               
                pca_cols = [
                    "component_"+str(col) for col in range(
                        x_dummies.shape[1]
                    )]
                x_dummies = pd.DataFrame(x_dummies,
                                         columns=pca_cols)
            
            # Consolidate all matrices into input_matrix X.
            self.preprocessing_time = (
                dt.datetime.now() - time_start).total_seconds()
            return x_num, x_cat, x_dummies
        
        elif approach == 'extended':
            # Ensure the elements in x_multi are list_like
            type_check = x_multi.map(
                                lambda x: pd.api.types.is_list_like(
                                    x, allow_sets=True)
                            ).any()
            if not type_check.any():
                # If any of the columns in type_check is not list_like
                for item in enumerate(type_check):
                    if not item[1]:
                        raise TypeError(f"Column {type_check.index[item[0]]}"
                                         "is not list-like. Check column" 
                                         "indices or re-encode the column.")
            # Turn x_multi into sets
            try:
                x_multi = x_multi.map(set)
            except Exception as exc:
                raise TypeError("Something went wrong while encoding"
                                "multi-valued attributes as sets. Ensure"
                                "all values are list-like.") from exc

            # Consolidate all matrices into input_matrix X.ç
            self.preprocessing_time = (
                dt.datetime.now() - time_start).total_seconds()
            return x_num, x_cat, x_multi

    def cluster(self):
        pass