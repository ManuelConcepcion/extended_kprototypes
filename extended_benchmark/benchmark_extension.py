"""Provide the necessary tools for benchmarking Extended K-Prototypes."""
# Imports
import time
from typing import Any, Optional
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import \
    adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

from kmodes.kprototypes import KPrototypes
from kmodes.extended_kprototypes import ExtendedKPrototypes
from kmodes.util.dissim import jaccard_dissim_sets

# Constants
VALID_PREPROCESS_MODES = ('naive', 'one-hot', 'one-hot-pca', 'extended')
PARAM_GUIDE = {
    'n_samples': int,
    'n_clusters': int,
    # Numeric features
    'n_numeric_features': int,
    # Categorical features
    'n_categorical_features': int,
    'categorical_cardinalities': list,
    # Multi-valued features
    'n_multival_features': int,
    'multival_vocab_lens': list,
    # Difficulty params
    'separability': float,
    'multival_intersections': int,
    'noise': float,
    'class_weights': list,
    # Approach settings
    'approach_settings': dict
}
# MINIMUM_N_TO_D_RATIO = 5


class Preprocessor:
    """Preprocess the raw data as necessary for a given method."""
    def __init__(self,
                 raw_data: pd.DataFrame,
                 approach_settings: dict,
                 categorical_indexes: list[int],
                 multival_indexes: list[int]) -> None:
        self.raw_data = raw_data
        self.approach_settings = approach_settings
        self.categorical_indexes = categorical_indexes
        self.multival_indexes = multival_indexes

    def _naive_preprocessing(self,
                             raw_data: pd.DataFrame) -> pd.DataFrame:
        naive_data = raw_data.copy()

        for icol in self.multival_indexes:
            naive_data.iloc[:, icol] = naive_data.iloc[:, icol].astype(str)
        return naive_data, (self.categorical_indexes+self.multival_indexes)

    def _cut_dummies(self,
                     df_with_dummies: pd.DataFrame,
                     new_categorical_indexes: list) -> pd.DataFrame:
        dummy_indexes = list(set(new_categorical_indexes).difference(
            set(self.categorical_indexes)))

        frequencies = pd.DataFrame(
            df_with_dummies.iloc[:, dummy_indexes].mean(),
            columns=['freq']).sort_values('freq', ascending=False)

        saved_dummies = frequencies.index[
            :self.approach_settings['one-hot']['max_dummies']]
        saved_cols = [col for icol, col in enumerate(df_with_dummies.columns)
                      if icol not in dummy_indexes] + list(saved_dummies)

        return df_with_dummies[saved_cols]

    def _one_hot_preprocessing(self,
                               raw_data: pd.DataFrame,
                               cut_dummies: bool) -> pd.DataFrame:
        processed_df = raw_data.copy()

        for col in self.multival_indexes:
            processed_df.iloc[:, col] = (processed_df.iloc[:, col]
                                         .apply(lambda x: list(x)))
        df_as_is = processed_df.iloc[:, [i for i in
                                         range(processed_df.shape[1]) if i not
                                         in self.multival_indexes]]
        multi_val_df = processed_df.iloc[:, self.multival_indexes]
        multi_val_df.index.rename('index', inplace=True)

        columns_to_concat = [df_as_is]
        out_cols = list(df_as_is.columns)

        for icol in range(multi_val_df.shape[1]):
            dummy_df = pd.get_dummies(multi_val_df.iloc[:, 0].apply(pd.Series)
                                      .stack()).groupby('index', level=0).sum()
            out_cols += [f'multi_{icol}_{col}' for col in dummy_df.columns]
            columns_to_concat.append(dummy_df)

        out_df = pd.concat(columns_to_concat, axis=1)
        out_df.columns = out_cols

        num_indexes = [icol for icol in range(raw_data.shape[1]) if (
            icol not in self.categorical_indexes
            or icol not in self.multival_indexes)]
        new_categorical_indexes = [icol for icol in range(out_df.shape[1])
                                   if icol not in num_indexes]

        # Preserve only the top-k frequent dummies
        if cut_dummies:
            out_df = self._cut_dummies(out_df, new_categorical_indexes)

        return out_df, new_categorical_indexes

    def _apply_pca(self,
                   data: pd.DataFrame,
                   new_categorical_indexes: list) -> pd.DataFrame:
        """Apply PCA to a previously dummified dataset."""
        processing_df = data.copy()
        dummy_icols = list(
            set(new_categorical_indexes).difference(
                set(self.categorical_indexes))
            )
        pca = PCA(n_components=round(len(dummy_icols)*0.25))
        pca_df = pd.DataFrame(pca.fit_transform(processing_df
                                                .iloc[:, dummy_icols]))
        pca_df.columns = [f'pca_{col}' for col in pca_df.columns]

        other_icols = [icol for icol in range(processing_df.shape[1])
                       if icol not in dummy_icols]

        return (pd.concat(
            [processing_df.iloc[:, other_icols], pca_df], axis=1),
                self.categorical_indexes)

    def preprocess_data(self,
                        approach: str,
                        raw_data: pd.DataFrame = None):
        """
        Preprocess the data for clustering according to one of the valid
        approaches.
        """
        # Verify inputs
        if approach not in VALID_PREPROCESS_MODES:
            raise ValueError("Approach must be one of "
                             f"{VALID_PREPROCESS_MODES}.")

        if raw_data is None:
            raw_data = self.raw_data
        if approach == 'naive':
            # Turn the categorical and multi-valued into dummies
            approach_data, new_categorical_indexes = \
                self._naive_preprocessing(raw_data)
        elif approach == 'one-hot':
            # Turn all items across all dicts into dummies
            approach_data, new_categorical_indexes = \
                self._one_hot_preprocessing(raw_data, cut_dummies=True)
        elif approach == 'one-hot-pca':
            # Same as one-hot but apply PCA to reduce dummies
            approach_data, new_categorical_indexes = \
                self._one_hot_preprocessing(raw_data, cut_dummies=False)
            # Apply pca
            approach_data, new_categorical_indexes = \
                self._apply_pca(approach_data, new_categorical_indexes)
        elif approach == 'extended':
            # Data needs no more processing :)
            approach_data = raw_data.copy()
            new_categorical_indexes = self.categorical_indexes
        else:
            raise ValueError(f"Invalid approach: '{approach}'.")

        return approach_data, new_categorical_indexes


class Experiment:
    """Handle the benchmarking process for a given configuration"""
    def __init__(self,
                 benchmarking_config: dict[str, Any],
                 approaches: tuple[str] = ('extended',),
                 random_state: int = 42) -> None:
        self.benchmarking_config = self._validate_config(benchmarking_config,
                                                         PARAM_GUIDE)
        self.approaches = approaches
        self.random_state = random_state

        self.data = None
        self.true_labels = None
        self.categorical_indexes = None
        self.multival_indexes = None

    @staticmethod
    def _validate_config(config_dict, param_guide):
        """Validate structure and types of provided configuration."""
        try:
            for parm in param_guide.keys():
                if not isinstance(config_dict[parm], param_guide[parm]):
                    raise ValueError(f"Parameter {parm} should be an integer.")
        except KeyError as e:
            print(f"Missing key {parm} in the configuration dict.")
            raise e

        # Length checks
        if len(config_dict['class_weights']) != config_dict['n_clusters']:
            if (len(config_dict['class_weights']) !=
                    config_dict['n_clusters']-1):
                raise ValueError("A number of class_weights equal to the "
                                 "number of clusters or the number of "
                                 "clusters minus one must be provided. List "
                                 "had size "
                                 f"{len(config_dict['class_weights'])}.")

        if (len(config_dict['categorical_cardinalities']) !=
                config_dict['n_categorical_features']):

            raise ValueError("A cardinality must be provided for every "
                             "categorical attribute. "
                             f"{len(config_dict['categorical_cardinalities'])}"
                             " cardinalities were provided for "
                             f"{config_dict['n_categorical_features']}.")
        if (len(config_dict['multival_vocab_lens']) !=
                config_dict['n_multival_features']):

            raise ValueError("A vocabulary length must be provided for every "
                             f"multi-valued attribute. "
                             f"{len(config_dict['multival_subvocab_length'])} "
                             "vocabulary length list-likes were provided for "
                             f"{config_dict['n_multival_features']} "
                             "attributes.")

        for card in config_dict['categorical_cardinalities']:
            if card < config_dict['n_clusters']:
                raise ValueError("Categorical attribute cardinalities cannot "
                                 "be lower than n_clusters.")

        for card_tuple in config_dict['multival_vocab_lens']:
            if len(card_tuple) != config_dict['n_clusters']:
                raise ValueError("A sub-vocabulary length must be provided for"
                                 f" each cluster. {card_tuple} was provided "
                                 "without the required length "
                                 f"{config_dict['n_clusters']}")

        if (config_dict['approach_settings']['extended']['gamma_m'] +
                config_dict['approach_settings']['extended']['gamma_c'] >=
                1.0):
            actual_sum = (config_dict['approach_settings']['extended']
                          ['gamma_m'] + config_dict['approach_settings']
                          ['extended']['gamma_c'])
            raise ValueError("Extended K-Prototypes gamma values should sum "
                             "up to less than 1.0. The values provided sum up "
                             f"to {actual_sum}.")

        return config_dict

    @staticmethod
    def _assign_categorical_features(class_labels: np.ndarray,
                                     cardinalities: list,
                                     random_state):
        """Create the categorical features for an existing partition."""
        n_clusters = len(np.unique(class_labels))
        random_generator = np.random.default_rng(seed=random_state)
        cluster_classes_keys = []
        categorical_attribute_arrays = []

        for card in cardinalities:
            cluster_to_class = dict()

            extra_levels = card % n_clusters
            levels_per_cluster = (card - extra_levels) / n_clusters
            curr_level = 0

            for clust in np.unique(class_labels):
                cluster_to_class[clust] = []

                for level in range(int(levels_per_cluster)):
                    cluster_to_class[clust].append(curr_level+level)
                curr_level += 2

            # Assign the extra levels at random
            if extra_levels:
                for extra_level in range(extra_levels):
                    clust = random_generator.choice(class_labels)
                    cluster_to_class[clust].append(curr_level+extra_level)

            # For items belonging to a label, assign it a level from the dict
            attribute = np.zeros(class_labels.shape[0], dtype=np.int32)
            for i_label, label in enumerate(class_labels):
                attribute[i_label] = \
                    random_generator.choice(cluster_to_class[label])

            categorical_attribute_arrays.append(attribute)
            cluster_classes_keys.append(cluster_to_class)

        return (np.stack(categorical_attribute_arrays, axis=1),
                cluster_classes_keys)

    @staticmethod
    def _assign_multival_features(class_labels: np.ndarray,
                                  subvocab_lengths: list,
                                  intersection_lvl: int) -> np.ndarray:
        """
        Create multi-valued attributes from class labels, the lengths of the
        vocabulary subsets that are assigned to each label, and the degree to
        which pairwise clusters should have intersections.

        Arguments
        ---------
        intersection_lvl:int
            The level of intersection refers to the number of items in the
            vocabulary that are common to cluster pairs. The higher it is
            relative to the sub-vocabulary lenghts, the lower the distance
            between clusters in relation to their multi-valued attributes
            will be.
        """
        clusters = np.unique(class_labels)
        attribute_label_dicts = []
        multi_valued_attribute_arrays = []

        for subvocab in subvocab_lengths:   # Iterate over multivalued attrs
            total_attribute_vocabulary = {-1}
            label_vocab_dict = dict().fromkeys(clusters)

            for clust in clusters:
                label_vocab_dict[clust] = set()
                subvocab_clust_l = subvocab[clust]

                for item in range(max(total_attribute_vocabulary) + 1,
                                  max(total_attribute_vocabulary) +
                                  subvocab_clust_l + 1):
                    label_vocab_dict[clust].add(item)
                    total_attribute_vocabulary.add(item)

            total_attribute_vocabulary.remove(-1)

            # Add the pairwise intersections
            if intersection_lvl > 0:
                for cluster_pair in [clust_comb for clust_comb in
                                     combinations_with_replacement(clusters, 2)
                                     if clust_comb[0] != clust_comb[1]]:
                    for item in range(max(total_attribute_vocabulary) + 1,
                                      max(total_attribute_vocabulary) +
                                      intersection_lvl + 1):

                        label_vocab_dict[cluster_pair[0]].add(item)
                        label_vocab_dict[cluster_pair[1]].add(item)
                        total_attribute_vocabulary.add(item)

            attribute_label_dicts.append(label_vocab_dict)

            # Build, for each attribute, the array containing the observations
            attribute = np.zeros(class_labels.shape[0], dtype=np.object_)
            for i_label, label in enumerate(class_labels):
                attribute[i_label] = label_vocab_dict[label]

            multi_valued_attribute_arrays.append(attribute)

        return (np.stack(multi_valued_attribute_arrays, axis=1),
                attribute_label_dicts)

    @staticmethod
    def _consolidate_attribute_arrays(xlist: list[tuple[str, np.ndarray]],
                                      target: np.ndarray):
        column_dictionary = dict()
        index_dict = dict()

        global_index = 0
        for attr_type, attr_vals in xlist:
            index_dict[attr_type] = []
            local_index = 0

            for i_attr in range(attr_vals.shape[1]):
                column_dictionary[f'{attr_type}_{local_index}'] = \
                    attr_vals[:, i_attr]
                index_dict[attr_type].append(global_index)

                local_index += 1
                global_index += 1

        return target, pd.DataFrame(column_dictionary), index_dict

    def generate_data(self):
        """Generate synthetic data for benchmarking."""
        xnum, y_true = make_classification(
            n_samples=self.benchmarking_config['n_samples'],
            n_features=self.benchmarking_config['n_numeric_features'],
            n_informative=self.benchmarking_config['n_numeric_features'],
            n_redundant=0,
            n_repeated=0,
            n_classes=self.benchmarking_config['n_clusters'],
            n_clusters_per_class=1,
            weights=self.benchmarking_config['class_weights'],
            flip_y=self.benchmarking_config['noise'],
            class_sep=self.benchmarking_config['separability'],
            random_state=self.random_state
            )
        xcat, _ = \
            self._assign_categorical_features(class_labels=y_true,
                                              cardinalities=self
                                              .benchmarking_config
                                              ['categorical_cardinalities'],
                                              random_state=self.random_state
                                              )
        xmulti, _ = self._assign_multival_features(class_labels=y_true,
                                                   subvocab_lengths=self
                                                   .benchmarking_config
                                                   ['multival_vocab_lens'],
                                                   intersection_lvl=self
                                                   .benchmarking_config
                                                   ['multival_intersections']
                                                   )
        # Put everything together
        all_attributes = [
            ('num', xnum),
            ('cat', xcat),
            ('multi', xmulti)
        ]

        return self._consolidate_attribute_arrays(all_attributes,
                                                  target=y_true)

    def average_silhouette_score(self,
                                 data: pd.DataFrame,
                                 labels: np.ndarray,
                                 categorical: list[int],
                                 multi_valued: Optional[list[int]] = None,
                                 kp_gamma: Optional[float] = None,
                                 gamma_c: Optional[float] = None,
                                 gamma_m: Optional[float] = None):
        """Compute the average silhouette score for given data and partition"""
        # Construct the joint distance matrix
        xcat = data.iloc[:, categorical].values
        distance_matrix_cat = pdist(xcat, custom_hamming_dist)

        if multi_valued is None:
            num_idxs = [idx for idx in range(data.shape[1])
                        if idx not in categorical]
            xnum = data.iloc[:, num_idxs].values

            distance_matrix_num = pdist(xnum)
            distance_matrix = \
                squareform(distance_matrix_cat*kp_gamma + distance_matrix_num)
        else:
            gamma_n = 1 - gamma_c - gamma_m

            num_idxs = [idx for idx in range(data.shape[1])
                        if idx not in categorical+multi_valued]
            xnum = data.iloc[:, num_idxs].values
            xmulti = data.iloc[:, multi_valued].values

            distance_matrix_num = pdist(xnum)
            distance_matrix_multi = pdist(xmulti, jaccard_dissim_sets)
            distance_matrix = squareform(distance_matrix_cat*gamma_c +
                                         distance_matrix_num*gamma_n +
                                         distance_matrix_multi*gamma_m)

        return silhouette_score(distance_matrix, labels, metric="precomputed")

    def run_experiment(self):
        """Run an experiment configuration over the approaches provided."""
        if self.data is None:
            self.true_labels, self.data, index_reference = self.generate_data()
            self.categorical_indexes = index_reference['cat']
            self.multival_indexes = index_reference['multi']

        approaches_data_dict = dict.fromkeys(self.approaches)
        approaches_cat_idxs_dict = dict.fromkeys(self.approaches)
        approaches_results = dict.fromkeys(self.approaches)

        data_preprocessor = Preprocessor(
            raw_data=self.data,
            approach_settings=self.benchmarking_config['approach_settings'],
            categorical_indexes=self.categorical_indexes,
            multival_indexes=self.multival_indexes
            )

        for approach in self.approaches:
            approaches_results[approach] = {}
            start = time.perf_counter()

            (approaches_data_dict[approach],
             approaches_cat_idxs_dict[approach]) = \
                data_preprocessor.preprocess_data(approach=approach)

            stop = time.perf_counter()
            approaches_results[approach]['preprocess_time'] = stop-start

        # Now that the data for each approach is prepared, run the experiment
        # with the provided configuration.

        for approach in self.approaches:
            if approach == 'extended':
                gamma_c = (self.benchmarking_config['approach_settings']
                           [approach]['gamma_c'])
                gamma_m = (self.benchmarking_config['approach_settings']
                           [approach]['gamma_m'])
                theta = (self.benchmarking_config['approach_settings']
                         [approach]['theta'])

                kp = ExtendedKPrototypes(
                    n_clusters=self.benchmarking_config['n_clusters'],
                    gamma_c=gamma_c,
                    gamma_m=gamma_m,
                    theta=theta,
                    random_state=self.random_state
                )
                start = time.perf_counter()
                kp.fit(approaches_data_dict[approach],
                       categorical=self.categorical_indexes,
                       multi_valued=self.multival_indexes)
                stop = time.perf_counter()

                # Calc the silhouette score for the resulting clustering
                silhouette_result = self.average_silhouette_score(
                    data=approaches_data_dict[approach],
                    labels=kp.labels_,
                    categorical=self.categorical_indexes,
                    multi_valued=self.multival_indexes,
                    gamma_c=kp.gamma_c,
                    gamma_m=kp.gamma_m
                )
            else:
                gamma = (self.benchmarking_config['approach_settings']
                         [approach]['gamma'])
                if gamma:
                    approaches_results[approach]['gamma'] = gamma

                kp = KPrototypes(
                    n_clusters=self.benchmarking_config['n_clusters'],
                    gamma=gamma, random_state=self.random_state
                    )
                start = time.perf_counter()
                kp.fit(approaches_data_dict[approach],
                       categorical=approaches_cat_idxs_dict[approach])
                stop = time.perf_counter()

                silhouette_result = self.average_silhouette_score(
                    data=approaches_data_dict[approach],
                    labels=kp.labels_,
                    categorical=approaches_cat_idxs_dict[approach],
                    kp_gamma=kp.gamma
                )

            approaches_results[approach]['clustering_time'] = stop-start
            approaches_results[approach]['sum_of_times'] = (
                approaches_results[approach]['preprocess_time'] + (stop-start))
            approaches_results[approach]['n_iter'] = kp.n_iter_

            predicted_labels = kp.labels_
            approaches_results[approach]['MIS'] = \
                adjusted_mutual_info_score(labels_true=self.true_labels,
                                           labels_pred=predicted_labels)
            approaches_results[approach]['ARI'] = \
                adjusted_rand_score(labels_true=self.true_labels,
                                    labels_pred=predicted_labels)
            approaches_results[approach]['Silhouette Index'] = silhouette_result

            approaches_results[approach]['centroids'] = kp.cluster_centroids_

        return approaches_results

    def experiment_across_values(self,
                                 base_config: dict,
                                 random_states: list[int],
                                 **kwargs):
        """
        Define a number of keys to change iteratively across an otherwise
        static experiment configuration. The base configuration must be valid.
        This function works linearly, such that all arguments[i] will be tried
        at iteration i and no more.
        """
        config = self._validate_config(base_config, PARAM_GUIDE)

        # Check that kwargs have the same length.
        value_lens = []
        for value in kwargs.values():
            value_lens.append(len(value))
        if len(set(value_lens)) > 1:
            raise ValueError("One or more of the provided arguments has a "
                             "mismatched length with the rest.")
        # Check there are enough random_states.
        if len(random_states) != value_lens[0]:
            raise ValueError("A random_state int must be provided for every "
                             f"new parameter configuration. {value_lens[0]} "
                             "configurations were provided with "
                             f"{len(random_states)} random states.")

        # Iterate over each provided configuration
        n_iter = value_lens[0]
        results = []

        for i in range(n_iter):
            # Repopulate the config dict as needed.
            for key, value in kwargs.items():
                if key not in config.keys():
                    raise KeyError(f"Argument {key} is not present in the "
                                   "base_config dictionary.")
                config[key] = value[i]

                self.data = None
                self.random_state = random_states[i]
                self.benchmarking_config = config

                results.append(self.run_experiment())

        return results


def custom_hamming_dist(a, b):
    """
    Scipy's pdist does not support dtype object and I am not encoding the sets
    for the naive approach.
    """
    return np.sum(np.vectorize(lambda x, y: x == y)(a, b)) / a.shape[0]
