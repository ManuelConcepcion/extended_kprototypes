"""
Functions for initializing clusters for categorical and multi-valued
attributes.
"""

from collections import defaultdict
from typing import Callable
from numpy.typing import ArrayLike

import numpy as np


def init_huang(x, n_clusters, dissim, random_state):
    """Initialize centroids according to method by Huang [1997]."""
    n_attrs = x.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # determine frequencies of attributes
    for iattr in range(n_attrs):
        # Sample centroids using the probabilities of attributes.
        # (I assume that's what's meant in the Huang [1998] paper; it works,
        # at least)
        # Note: sampling using population in static list with as many choices
        # as frequency counts. Since the counts are small integers,
        # memory consumption is low.
        choices = x[:, iattr]
        # So that we are consistent between Python versions,
        # each with different dict ordering.
        choices = sorted(choices)
        centroids[:, iattr] = random_state.choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in x.
    for ik in range(n_clusters):
        ndx = np.argsort(dissim(x, centroids[ik]))
        # We want the centroid to be unique, if possible.
        while (np.all(x[ndx[0]] == centroids, axis=1).any()
               and ndx.shape[0] > 1):
            ndx = np.delete(ndx, 0)
        centroids[ik] = x[ndx[0]]

    return centroids


def init_cao(x, n_clusters, dissim):
    """Initialize centroids according to method by Cao et al. [2009].

    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters
    """
    n_points, n_attrs = x.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for val in x[:, iattr]:
            freq[val] += 1
        for ipoint in range(n_points):
            dens[ipoint] += (freq[x[ipoint, iattr]]
                             / float(n_points)
                             / float(n_attrs))

    # Choose initial centroids based on distance and density.
    return cao_logic(x, centroids, dens, n_clusters, dissim)


def init_cao_multi(x_multi: ArrayLike,
                   n_clusters: int,
                   dissim: Callable):
    """Initialize centroids according to method by Cao et al. [2009].

    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters

    Parameters:
    x (numpy.array): Observations x Attributes matrix containing the
                     multi-valued attributes each x[i,j] should be a set.
    n_clusters (int): Number of desired clusters.
    dissim (function): Pairwise dissimilarity function. Unused if
                       distance_matrix is provided.
    """
    # Set up
    n_points, n_attrs = x_multi.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')

    attr_densities = count_dictionary_value_frequency(x_multi, n_points)

    # Create N x D densities matrix to sum all attributes and get the 1d 'dens'
    # Choose initial centroids based on distance and density.
    return cao_logic(x=x_multi,
                     centroids=centroids,
                     dens=np.sum(
                         np.column_stack(attr_densities), axis=1) / n_attrs,
                     n_clusters=n_clusters,
                     dissim=dissim)


def count_dictionary_value_frequency(x_multi, n_points):
    """
    Find the frequency of each individual item of each attribute's
    dictionary.
    """
    # Create a dictionary of frequencies for each attribute
    flat_x = np.array(list(
        map(lambda x: np.array(x, dtype='object'), x_multi)
        )).flatten('F')
    attr_densities = []
    attr_dict = defaultdict(default_factory=0)
    start_idx = 0

    # Iterate to extract each attribute separately
    for end_idx in range(n_points, flat_x.shape[0]+1, n_points):
        # Clear the dictionary from the previous iteration
        attr_dict.clear()
        # Select the relevant indices
        attr_values = flat_x[start_idx:end_idx]
        start_idx = end_idx
        # Turn the sets into lists for everything else to work
        attr_values = list(map(list, attr_values))

        # Extract a dictionary for each attribute
        keys, values = np.unique(np.concatenate(attr_values),
                                 return_counts=True)
        attr_dict.update(zip(keys, values))

        # For each observation, reference the dict to get the total hits
        totals = np.array(
            [np.sum(observation) for observation in list(
                map(np.vectorize(lambda x: attr_dict[x]), attr_values)
            )])
        attr_densities.append(totals / len(attr_dict.keys()))

    return attr_densities


def cao_logic(x, centroids, dens, n_clusters, dissim):
    """Perform the algorithm in Cao [2009] after preliminary calculations."""
    n_points = x.shape[0]
    centroids[0] = x[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(x, centroids[ikk]) * dens
            centroids[ik] = x[np.argmax(np.min(dd, axis=0))]

    return centroids
