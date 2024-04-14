"""
K-prototypes clustering for mixed categorical and numerical data
"""

# pylint: disable=unused-argument,attribute-defined-outside-init

from collections import defaultdict
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from . import kmodes
from .util import get_max_value_key, encode_features, get_unique_rows, \
    decode_centroids, pandas_to_numpy, convert_listlike_to_sets
from .util.dissim import matching_dissim, euclidean_dissim, jaccard_dissim_sets
from .util.init_methods import init_cao, init_huang, init_cao_multi

# Number of tries we give the initialization methods to find non-empty
# clusters before we switch to random initialization.
MAX_INIT_TRIES = 20
# Number of tries we give the initialization before we raise an
# initialization error.
RAISE_INIT_TRIES = 100


class ExtendedKPrototypes(kmodes.KModes):
    """k-protoypes clustering algorithm for mixed numerical/categorical data.

    Parameters
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 100
        Maximum number of iterations of the k-modes algorithm for a
        single run.

    num_dissim : func, default: euclidian_dissim
        Dissimilarity function used by the algorithm for numerical variables.
        Defaults to the Euclidian dissimilarity function.

    cat_dissim : func, default: matching_dissim
        Dissimilarity function used by the kmodes algorithm for categorical
        variables. Defaults to the matching dissimilarity function.

    n_init : int, default: 10
        Number of time the k-modes algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of cost.

    init : {'Huang', 'Cao', 'random' or a list of ndarrays}, default: 'Cao'
        Method for initialization:
        'Huang': Method in Huang [1997, 1998]
        'Cao': Method in Cao et al. [2009]
        'random': choose 'n_clusters' observations (rows) at random from
        data for the initial centroids.
        If a list of ndarrays is passed, it should be of length 2, with
        shapes (n_clusters, n_features) for numerical and categorical
        data respectively. These are the initial encoded centroids.

    gamma : float, default: None
        Weighing factor that determines relative importance of numerical vs.
        categorical attributes (see discussion in Huang [1997]). By default,
        automatically calculated from data.

    verbose : integer, optional
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    cluster_centroids_ : array, [n_clusters, n_features]
        Categories of cluster centroids

    labels_ :
        Labels of each point

    cost_ : float
        Clustering cost, defined as the sum distance of all points to
        their respective cluster centroids.

    n_iter_ : int
        The number of iterations the algorithm ran for.

    epoch_costs_ :
        The cost of the algorithm at each epoch from start to completion.

    gamma : float
        The (potentially calculated) weighing factor.

    Notes
    -----
    See:
    Huang, Z.: Extensions to the k-modes algorithm for clustering large
    data sets with categorical values, Data Mining and Knowledge
    Discovery 2(3), 1998.

    """

    def __init__(self, n_clusters=8, max_iter=100, num_dissim=euclidean_dissim,
                 cat_dissim=matching_dissim, multi_dissim=jaccard_dissim_sets,
                 init='Cao', n_init=10, gamma_c=0.33, gamma_m=0.33, verbose=0,
                 random_state=None, n_jobs=1):

        super(ExtendedKPrototypes, self).__init__(n_clusters, max_iter,
                                                  cat_dissim,
                                                  init, verbose=verbose,
                                                  random_state=random_state,
                                                  n_jobs=n_jobs)
        self.num_dissim = num_dissim
        self.multi_dissim = multi_dissim
        self.gamma_c = gamma_c
        self.gamma_m = gamma_m
        self.n_init = n_init
        if isinstance(self.init, list) and self.n_init > 1:
            if self.verbose:
                print("Initialization method is deterministic. "
                      "Setting n_init to 1.")
            self.n_init = 1

    def fit(self, X, y=None, categorical=None, multi_valued=None,
            sample_weight=None):
        """Compute k-prototypes clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        categorical : Index of columns that contain categorical data

        sample_weight : sequence, default: None
        The weight that is assigned to each individual data point when
        updating the centroids.

        """
        if categorical is not None:
            if not isinstance(categorical, (int, list, tuple)):
                raise ValueError("The 'categorical' argument needs to be an "
                                 "integer with the index of the categorical "
                                 "column in your data, or a list or tuple of "
                                 "several of them, but it is a "
                                 f"{type(categorical)}.")
        if multi_valued is not None:
            if not isinstance(multi_valued, (int, list, tuple)):
                raise ValueError("The 'categorical' argument needs to be an "
                                 "integer with the index of the categorical "
                                 "column in your data, or a list or tuple of "
                                 "several of them, but it is a "
                                 f"{type(multi_valued)}.")

        X = pandas_to_numpy(X)

        random_state = check_random_state(self.random_state)
        kmodes._validate_sample_weight(sample_weight, n_samples=X.shape[0],
                                       n_clusters=self.n_clusters)

        # If self.gamma is None, gamma will be automatically determined from
        # the data. The function below returns its value.
        self._enc_cluster_centroids, self._enc_map, self.labels_, self.cost_, \
            self.n_iter_, self.epoch_costs_, self.gamma_c, self.gamma_m = \
            k_prototypes(
                X=X,
                categorical=categorical,
                multi_valued=multi_valued,
                n_clusters=self.n_clusters,
                max_iter=self.max_iter,
                num_dissim=self.num_dissim,
                cat_dissim=self.cat_dissim,
                multi_dissim=self.multi_dissim,
                gamma_c=self.gamma_c,
                gamma_m=self.gamma_m,
                init=self.init,
                n_init=self.n_init,
                verbose=self.verbose,
                random_state=random_state,
                n_jobs=self.n_jobs,
                sample_weight=sample_weight,
            )

        return self

    def predict(self, X, categorical=None, multi_valued=None, **kwargs):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        categorical : Indices of columns that contain categorical data

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        assert hasattr(self, '_enc_cluster_centroids'), "Model not yet fitted."

        if categorical is not None:
            if not isinstance(categorical, (int, list, tuple)):
                raise ValueError("The 'categorical' argument needs to be an "
                                 "integer with the index of the categorical "
                                 "column in your data, or a list or tuple of "
                                 "several of them, but it is a "
                                 f"{type(categorical)}.")

        if multi_valued is not None:
            if not isinstance(multi_valued, (int, list, tuple)):
                raise ValueError("The 'categorical' argument needs to be an "
                                 "integer with the index of the categorical "
                                 "column in your data, or a list or tuple of "
                                 "several of them, but it is a "
                                 f"{type(multi_valued)}.")
        if self.verbose:
            print('Preprocessing')
        X = pandas_to_numpy(X)
        Xnum, Xcat, Xmulti = _split_num_cat_multi(X, categorical, multi_valued)
        Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)
        Xcat, _ = encode_features(Xcat, enc_map=self._enc_map)
        Xmulti = convert_listlike_to_sets(Xmulti)
        if self.verbose:
            print("Done preprocessing.")
        return labels_cost(Xnum=Xnum, Xcat=Xcat, Xmulti=Xmulti,
                           centroids=self._enc_cluster_centroids,
                           num_dissim=self.num_dissim,
                           cat_dissim=self.cat_dissim,
                           multi_dissim=self.multi_dissim,
                           gamma_c=self.gamma_c, gamma_m=self.gamma_m)[0]

    @property
    def cluster_centroids_(self):
        if hasattr(self, '_enc_cluster_centroids'):
            return np.hstack((
                self._enc_cluster_centroids[0],
                decode_centroids(self._enc_cluster_centroids[1], self._enc_map),
                self._enc_cluster_centroids[2]
            ))
        raise AttributeError("'{}' object has no attribute "
                             "'cluster_centroids_' because the model is not "
                             "yet fitted.")


def labels_cost(Xnum, Xcat, Xmulti,
                centroids, num_dissim, cat_dissim, multi_dissim,
                gamma_c, gamma_m,
                membship=None, sample_weight=None):
    """
    Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """
    gamma_n = 1 - (gamma_c + gamma_m)
    n_points = Xnum.shape[0]
    Xnum = check_array(Xnum)

    cost = 0.
    labels = np.empty(n_points, dtype=np.uint16)
    for ipoint in range(n_points):
        # Numerical cost = sum of Euclidean distances
        num_costs = num_dissim(centroids[0], Xnum[ipoint])
        cat_costs = cat_dissim(centroids[1], Xcat[ipoint], X=Xcat,
                               membship=membship)
        multi_costs = multi_dissim(centroids[2], Xmulti[ipoint])
        # Gamma relates the categorical cost to the numerical cost.
        tot_costs = (gamma_n * num_costs +
                     gamma_c * cat_costs +
                     gamma_m * multi_costs)
        clust = np.argmin(tot_costs)
        labels[ipoint] = clust
        if sample_weight is not None:
            cost += tot_costs[clust] * sample_weight[ipoint]
        else:
            cost += tot_costs[clust]

    return labels, cost


def k_prototypes(X, categorical, multi_valued, n_clusters, max_iter,
                 num_dissim, cat_dissim, multi_dissim, gamma_c, gamma_m,
                 init, n_init, verbose, random_state, n_jobs,
                 sample_weight=None):
    """k-prototypes algorithm"""
    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")

    if categorical is None or not categorical:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical, int):
        categorical = [categorical]
    if len(categorical) == X.shape[1]:
        raise ValueError("All columns are categorical, use k-modes instead "
                         "of k-prototypes.")
    if max(categorical) > X.shape[1]:
        raise ValueError("Categorical index larger than number of columns.")

    # ncatattrs = len(categorical)
    # nnumattrs = X.shape[1] - ncatattrs
    n_points = X.shape[0]
    if n_clusters > n_points:
        raise ValueError(f"Cannot have more clusters ({n_clusters}) "
                         f"than data points ({n_points}).")

    Xnum, Xcat, Xmulti = _split_num_cat_multi(x=X, cat_idxs=categorical,
                                              multi_val_idxs=multi_valued)
    Xnum, Xcat = check_array(Xnum), check_array(Xcat, dtype=None)

    # Convert the categorical values in Xcat to integers for speed.
    # Based on the unique values in Xcat, we can make a mapping to achieve it.
    Xcat, enc_map = encode_features(Xcat)

    # Make sure that the contents of Xmulti are sets
    Xmulti = convert_listlike_to_sets(Xmulti)

    # Are there more n_clusters than unique rows? Then set the unique
    # rows as initial values and skip iteration.
    unique = get_unique_rows(X, source='extendedkproto')
    n_unique = unique.shape[0]
    if n_unique <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = n_unique
        init = list(_split_num_cat_multi(unique, categorical, multi_valued))
        init[1], _ = encode_features(init[1], enc_map)

    # Estimate a good value for gamma, which determines the weighing of
    # categorical values in clusters (see Huang [1997]).
    # if gamma is None:
    #     gamma = 0.5 * np.mean(Xnum.std(axis=0))

    results = []
    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    if n_jobs == 1:
        for init_no in range(n_init):
            results.append(_extn_k_proto_single(Xnum, Xcat, Xmulti,
                                                n_clusters, n_points, max_iter,
                                                num_dissim, cat_dissim,
                                                multi_dissim, gamma_c, gamma_m,
                                                init, init_no, verbose,
                                                seeds[init_no], sample_weight))
    else:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_extn_k_proto_single)(Xnum, Xcat, Xmulti,
                                          n_clusters, n_points, max_iter,
                                          num_dissim, cat_dissim,
                                          multi_dissim, gamma_c, gamma_m,
                                          init, init_no, verbose,
                                          seeds[init_no], sample_weight)
            for init_no, seed in enumerate(seeds))
    (all_centroids, all_labels, all_costs,
     all_n_iters, all_epoch_costs) = zip(*results)

    best = np.argmin(all_costs)
    if n_init > 1 and verbose:
        print(f"Best run was number {best + 1}")

    # Note: return gamma in case it was automatically determined.
    return all_centroids[best], enc_map, all_labels[best], all_costs[best], \
        all_n_iters[best], all_epoch_costs[best], gamma_c, gamma_m


def _extn_k_proto_single(Xnum, Xcat, Xmulti,
                         n_clusters, n_points,
                         max_iter, num_dissim, cat_dissim, multi_dissim,
                         gamma_c, gamma_m, init, init_no,
                         verbose, random_state, sample_weight=None):
    # For numerical part of initialization, we don't have a guarantee
    # that there is not an empty cluster, so we need to retry until
    # there is none.
    gamma_n = 1 - (gamma_c + gamma_m)
    nnumattrs = Xnum.shape[1]
    ncatattrs = Xcat.shape[1]
    nmultiattrs = Xmulti.shape[1]

    random_state = check_random_state(random_state)
    init_tries = 0

    # CENTROID INIT AND FIRST CLUSTER ASSINGMENT LOOP --------
    while True:
        init_tries += 1
        # _____ INIT _____
        if verbose:
            print("Init: initializing centroids")
        if isinstance(init, str) and init.lower() == 'huang':
            centroids = init_huang(Xcat, n_clusters, cat_dissim, random_state)
        elif isinstance(init, str) and init.lower() == 'cao':
            centroids = init_cao(Xcat, n_clusters, cat_dissim)
        elif isinstance(init, str) and init.lower() == 'random':
            seeds = random_state.choice(range(n_points), n_clusters)
            centroids = Xcat[seeds]
        elif isinstance(init, list):
            # Make sure inits are 2D arrays.
            init = [np.atleast_2d(cur_init).T if len(cur_init.shape) == 1
                    else cur_init
                    for cur_init in init]
            assert init[0].shape[0] == n_clusters, \
                "Wrong number of initial numerical centroids in init " \
                f"({init[0].shape[0]}, should be {n_clusters})."
            assert init[0].shape[1] == nnumattrs, \
                "Wrong number of numerical attributes in init " \
                f"({init[0].shape[1]}, should be {nnumattrs})."
            assert init[1].shape[0] == n_clusters, \
                "Wrong number of initial categorical centroids in init " \
                f"({init[1].shape[0]}, should be {n_clusters})."
            assert init[1].shape[1] == ncatattrs, \
                "Wrong number of categorical attributes in init " \
                f"({init[1].shape[1]}, should be {ncatattrs})."
            assert init[2].shape[0] == n_clusters, \
                "Wrong number of initial categorical centroids in init " \
                f"({init[2].shape[0]}, should be {n_clusters})."
            assert init[2].shape[1] == ncatattrs, \
                "Wrong number of categorical attributes in init " \
                f"({init[2].shape[1]}, should be {ncatattrs})."

            centroids = [np.asarray(init[0], dtype=np.float64),
                         np.asarray(init[1], dtype=np.uint16)]

        else:
            raise NotImplementedError("Initialization method not supported.")

        if not isinstance(init, list):
            # Numerical is initialized by drawing from normal distribution,
            # categorical following the k-modes methods, and multi following
            # the proposed extension to the Cao initialization logic.
            meanx = np.mean(Xnum, axis=0)
            stdx = np.std(Xnum, axis=0)
            centroids = [   # 0:num, 1:cat, 2:multi
                meanx + random_state.randn(n_clusters, nnumattrs) * stdx,
                centroids,
                # Add multi-categorical part
                init_cao_multi(Xmulti, n_clusters=n_clusters,
                               dissim=multi_dissim)
            ]

        # INITIAL CLUSTER ASSIGNMENT ------
        if verbose:
            print("Init: initializing clusters")
        membship = np.zeros((n_clusters, n_points), dtype=np.bool_)
        # Keep track of the sum of attribute values per cluster so that we
        # can do k-means on the numerical attributes.
        cl_attr_sum = np.zeros((n_clusters, nnumattrs), dtype=np.float64)
        # Same for the membership sum per cluster
        cl_memb_sum = np.zeros(n_clusters, dtype=np.float64)
        # cl_attr_freq is a list of lists with dictionaries that contain
        # the frequencies of values per cluster and attribute.
        cl_attr_freq = [[defaultdict(float) for _ in range(ncatattrs)]
                        for _ in range(n_clusters)]

        cl_multi_attr_freq = [[defaultdict(float)
                               for _ in range(Xmulti.shape[1])]
                              for _ in range(n_clusters)]
        gl_multi_attr_freq = [defaultdict(float)
                              for _ in range(Xmulti.shape[1])]
        # Go point by point to fill these "accounting" matrices and dicts
        for ipoint in range(n_points):
            weight = sample_weight[ipoint] if sample_weight is not None else 1
            # Initial assignment to clusters
            clust = np.argmin(
                gamma_n * num_dissim(centroids[0], Xnum[ipoint]) +
                gamma_c * cat_dissim(
                    centroids[1], Xcat[ipoint], X=Xcat, membship=membship) +
                gamma_m * multi_dissim(centroids[2], Xmulti[ipoint])
            )
            membship[clust, ipoint] = 1
            cl_memb_sum[clust] += weight
            # Count attribute values per cluster.
            for iattr, curattr in enumerate(Xnum[ipoint]):
                cl_attr_sum[clust, iattr] += curattr * weight
            for iattr, curattr in enumerate(Xcat[ipoint]):
                cl_attr_freq[clust][iattr][curattr] += weight
            for iattr, curattr in enumerate(Xmulti[ipoint]):
                # curattr is a set in this case
                for item in curattr:
                    gl_multi_attr_freq[iattr][item] += weight
                    cl_multi_attr_freq[clust][iattr][item] += weight

        # If no empty clusters, then consider initialization finalized.
        if membship.sum(axis=1).min() > 0:
            break

        if init_tries == MAX_INIT_TRIES:
            # Could not get rid of empty clusters. Randomly
            # initialize instead.
            init = 'random'
        elif init_tries == RAISE_INIT_TRIES:
            raise ValueError(
                "Clustering algorithm could not initialize. "
                "Consider assigning the initial clusters manually."
            )

    # INITIAL CENTROID UPDATE
    for ik in range(n_clusters):
        for iattr in range(nnumattrs):
            # Numerical update: mean of attributes
            centroids[0][ik, iattr] = cl_attr_sum[ik, iattr] / cl_memb_sum[ik]
        for iattr in range(ncatattrs):
            # Categorical update: mode of attributes
            centroids[1][ik, iattr] = \
                get_max_value_key(cl_attr_freq[ik][iattr])
        for iattr in range(nmultiattrs):
            # Multi-valued update: proposed technique
            centroids[2][ik, iattr] = _compare_cl_to_gl_attribute_frequency(
                cl_dict=cl_multi_attr_freq[ik][iattr],
                gl_dict=gl_multi_attr_freq[iattr],
                n_clust=cl_memb_sum[ik],
                n_points=n_points
            )

    # _____ ITERATION _____
    if verbose:
        print("Starting iterations...")
    itr = 0
    labels = None
    converged = False

    _, cost = labels_cost(Xnum=Xnum, Xcat=Xcat, Xmulti=Xmulti,
                          centroids=centroids, num_dissim=num_dissim,
                          cat_dissim=cat_dissim, multi_dissim=multi_dissim,
                          gamma_c=gamma_c, gamma_m=gamma_m, membship=membship,
                          sample_weight=sample_weight)

    epoch_costs = [cost]
    while itr < max_iter and not converged:
        itr += 1
        (centroids, cl_attr_sum, cl_memb_sum, cl_attr_freq,
         cl_multi_attr_freq, membship, moves) = \
            _extn_k_proto_iter(Xnum=Xnum, Xcat=Xcat, Xmulti=Xmulti,
                               centroids=centroids, cl_attr_sum=cl_attr_sum,
                               cl_memb_sum=cl_memb_sum,
                               cl_attr_freq=cl_attr_freq,
                               cl_multi_attr_freq=cl_multi_attr_freq,
                               gl_multi_attr_freq=gl_multi_attr_freq,
                               membship=membship, num_dissim=num_dissim,
                               cat_dissim=cat_dissim,
                               multi_dissim=multi_dissim,
                               gamma_c=gamma_c, gamma_m=gamma_m,
                               random_state=random_state,
                               sample_weight=sample_weight)

        # All points seen in this iteration
        labels, ncost = labels_cost(Xnum=Xnum, Xcat=Xcat, Xmulti=Xmulti,
                                    centroids=centroids, num_dissim=num_dissim,
                                    cat_dissim=cat_dissim,
                                    multi_dissim=multi_dissim,
                                    gamma_c=gamma_c, gamma_m=gamma_m,
                                    membship=membship,
                                    sample_weight=sample_weight)
        converged = (moves == 0) or (ncost >= cost)
        epoch_costs.append(ncost)
        cost = ncost
        if verbose:
            print(f"Run: {init_no + 1}, iteration: {itr}/{max_iter}, "
                  f"moves: {moves}, ncost: {ncost}")

    return centroids, labels, cost, itr, epoch_costs


def _extn_k_proto_iter(# Pre-Separated Attribute Arrays
                       Xnum: np.ndarray,
                       Xcat: np.ndarray,
                       Xmulti: np.ndarray,
                       # "Accounting" matrices
                       centroids: list[np.ndarray],
                       cl_attr_sum: np.ndarray[np.float64],
                       cl_memb_sum: np.ndarray[np.float64],
                       cl_attr_freq: list[list[defaultdict]],
                       cl_multi_attr_freq: list[list[defaultdict]],
                       gl_multi_attr_freq: list[list[defaultdict]],
                       membship: np.ndarray[np.bool_],
                       # Dissimilarity Functions
                       num_dissim: Callable,
                       cat_dissim: Callable,
                       multi_dissim: Callable,
                       # Gamma parameters
                       gamma_c: float,
                       gamma_m: float,
                       # Miscellaneous Arguments
                       random_state,
                       sample_weight):
    """Single iteration of the k-prototypes algorithm"""
    n_points = Xnum.shape[0]
    if n_points != Xcat.shape[0] or n_points != Xmulti.shape[0]:
        raise RuntimeError("Something has gone terribly wrong. Attribute "
                           "matrices have different numbers of rows.")

    gamma_n = 1 - (gamma_c + gamma_m)
    moves = 0
    for ipoint in range(Xnum.shape[0]):
        weight = sample_weight[ipoint] if sample_weight is not None else 1
        clust = np.argmin(
            gamma_n * num_dissim(centroids[0], Xnum[ipoint]) +
            gamma_c * cat_dissim(
                centroids[1], Xcat[ipoint], X=Xcat, membship=membship) +
            gamma_m * multi_dissim(centroids[2], Xmulti[ipoint])
        )
        if membship[clust, ipoint]:
            # Point is already in its right place.
            continue

        # Move point, and update old/new cluster frequencies and centroids.
        moves += 1
        old_clust = np.argwhere(membship[:, ipoint])[0][0]

        # Note that membship gets updated by kmodes.move_point_cat.
        # move_point_num only updates things specific to the k-means part.
        cl_attr_sum, cl_memb_sum = _move_point_num(
            Xnum[ipoint], clust, old_clust, cl_attr_sum, cl_memb_sum, weight
        )
        cl_multi_attr_freq = _move_point_multi(
            Xmulti[ipoint], clust, old_clust, cl_multi_attr_freq, weight
        )
        # Update the actual centroids and the categorical "accounting" vars
        # noinspection PyProtectedMember
        cl_attr_freq, membship, centroids[1] = kmodes._move_point_cat(
            Xcat[ipoint], ipoint, clust, old_clust,
            cl_attr_freq, membship, centroids[1], weight
        )

        # Update old and new centroids for numerical attributes using
        # the means and sums of all values
        for iattr in range(len(Xnum[ipoint])):
            for curc in (clust, old_clust):
                if cl_memb_sum[curc]:
                    centroids[0][curc, iattr] = (cl_attr_sum[curc, iattr] /
                                                 cl_memb_sum[curc])
                else:
                    centroids[0][curc, iattr] = 0.

        # Could be integrated in the previous loop, but is left here
        # for modularity and compartimentalisation.
        for iattr in range(len(Xmulti[ipoint])):
            # Based on the updated accounting variable, cl_multi_attr_freq
            for curc in (clust, old_clust):
                centroids[2][curc, iattr] = (
                    _compare_cl_to_gl_attribute_frequency(
                        cl_dict=cl_multi_attr_freq[curc][iattr],
                        gl_dict=gl_multi_attr_freq[iattr],
                        n_clust=cl_memb_sum[curc],
                        n_points=n_points
                        )
                    )

        # In case of an empty cluster, reinitialize with a random point
        # from largest cluster.
        if not cl_memb_sum[old_clust]:
            from_clust = membship.sum(axis=1).argmax()
            choices = [ii for ii, ch in enumerate(
                membship[from_clust, :]) if ch]
            rindx = random_state.choice(choices)

            cl_attr_sum, cl_memb_sum = _move_point_num(
                Xnum[rindx], old_clust, from_clust, cl_attr_sum, cl_memb_sum,
                weight
            )
            cl_attr_freq, membship, centroids[1] = kmodes._move_point_cat(
                Xcat[rindx], rindx, old_clust, from_clust,
                cl_attr_freq, membship, centroids[1], weight
            )
            cl_multi_attr_freq = _move_point_multi(
                Xmulti[rindx], old_clust, from_clust,
                cl_multi_attr_freq, weight
            )

    return (centroids, cl_attr_sum, cl_memb_sum, cl_attr_freq,
            cl_multi_attr_freq, membship, moves)


def _compare_cl_to_gl_attribute_frequency(cl_dict, gl_dict, n_clust, n_points):
    modal_set = set()
    for item in cl_dict.keys():
        if item in gl_dict:
            freq_delta = (cl_dict[item])/n_clust - (gl_dict[item])/n_points
            if freq_delta >= 0.001:
                modal_set.add(item)

    return modal_set


def _move_point_num(point, to_clust, from_clust, cl_attr_sum, cl_memb_sum,
                    sample_weight):
    """Move point between clusters, numerical attributes."""
    # Update sum of attributes in cluster.
    for iattr, curattr in enumerate(point):
        cl_attr_sum[to_clust][iattr] += curattr * sample_weight
        cl_attr_sum[from_clust][iattr] -= curattr * sample_weight
    # Update sums of memberships in cluster
    cl_memb_sum[to_clust] += 1
    cl_memb_sum[from_clust] -= 1
    return cl_attr_sum, cl_memb_sum


def _move_point_multi(point, to_clust, from_clust,
                      cl_multi_attr_freq, sample_weight):
    """Move point between clusters, multi-valued attributes."""
    # Update the frequency count for the clusters involved.
    for iattr, curattr in enumerate(point):
        to_attr_counts = cl_multi_attr_freq[to_clust][iattr]
        from_attr_counts = cl_multi_attr_freq[from_clust][iattr]

        for item in curattr:
            to_attr_counts[item] += sample_weight
            from_attr_counts[item] -= sample_weight

    return cl_multi_attr_freq


def _split_num_cat_multi(x, cat_idxs, multi_val_idxs):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param x: Feature matrix
    :param categorical: Indices of categorical columns
    """
    num_idxs = [i for i in range(x.shape[1])
                if i not in (cat_idxs + multi_val_idxs)]
    Xnum = np.asanyarray(x[:, num_idxs]).astype(np.float64)

    Xcat = np.asanyarray(x[:, cat_idxs])
    Xmulti = np.asanyarray(x[:, multi_val_idxs])
    return Xnum, Xcat, Xmulti
