"""
Generic utilities for clustering
"""
from hashlib import sha256
from typing import Optional

import numpy as np


def pandas_to_numpy(x):
    """Turn pandas to numpy if x is a pandas dataframe"""
    return x.values if 'pandas' in str(x.__class__) else x


def get_max_value_key(dic):
    """Gets the key for the maximum value in a dict."""
    v = np.array(list(dic.values()))
    k = np.array(list(dic.keys()))

    maxima = np.where(v == np.max(v))[0]
    if len(maxima) == 1:
        return k[maxima[0]]

    # In order to be consistent, always selects the minimum key
    # (guaranteed to be unique) when there are multiple maximum values.
    return k[maxima[np.argmin(k[maxima])]]


def encode_features(X, enc_map=None):
    """Converts categorical values in each column of X to integers in the range
    [0, n_unique_values_in_column - 1].

    If mapping is not provided, it is calculated based on the values in X.

    Unknown values during prediction get a value of -1.
    """
    if enc_map is None:
        fit = True
        # We will calculate enc_map, so initialize the list of column mappings.
        enc_map = []
    else:
        fit = False

    Xenc = np.zeros(X.shape, dtype='int32')
    for ii in range(X.shape[1]):
        if fit:
            col_enc = {val: jj for jj, val in enumerate(np.unique(X[:, ii]))}
            enc_map.append(col_enc)
        # Unknown categories all get a value of -1.
        Xenc[:, ii] = np.array([enc_map[ii].get(x, -1) for x in X[:, ii]])

    return Xenc, enc_map


def convert_listlike_to_sets(Xmulti):
    """
    Take multi-valued attributes encoded as lists and turn them into sets.
    """
    try:
        out = np.zeros(shape=Xmulti.shape, dtype='object')
        for ipoint in range(Xmulti.shape[0]):
            for iattr in range(Xmulti.shape[1]):
                if not isinstance(Xmulti[ipoint][iattr], set):
                    print(f"{type(Xmulti[ipoint][iattr])}")
                    out[ipoint][iattr] = set(Xmulti[ipoint][iattr])
                else:
                    out[ipoint][iattr] = Xmulti[ipoint][iattr]

    except TypeError as exc:
        raise TypeError("There was a problem converting the multi-valued "
                        "attributes to sets.") from exc

    return out


def decode_centroids(encoded, mapping):
    """Decodes the encoded centroids array back to the original data
    labels using a list of mappings.
    """
    decoded = []
    for ii in range(encoded.shape[1]):
        # Invert the mapping so that we can decode.
        inv_mapping = {v: k for k, v in mapping[ii].items()}
        decoded.append(np.vectorize(inv_mapping.__getitem__)(encoded[:, ii]))
    return np.atleast_2d(np.array(decoded)).T


def get_unique_rows(a, source: Optional[str] = None):
    """Gets the unique rows in a numpy array."""
    if source is None:
        return np.vstack(list({tuple(row) for row in a}))

    elif source == 'extendedkproto':
        unique_seen = {'hello'}
        unique_rows = []

        for row in a:
            hexdigest = sha256(
                str(row).encode('utf-8'), usedforsecurity=False).hexdigest()
            if hexdigest not in unique_seen:
                unique_seen.add(hexdigest)
                unique_rows.append(row)

        return np.vstack(unique_rows)
