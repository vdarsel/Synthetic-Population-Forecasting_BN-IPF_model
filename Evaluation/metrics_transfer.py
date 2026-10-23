import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


_CAT_TYPES = ["binary", "category", "bool"]
_NUM_TYPES = ["float", "int"]

_TOL = 1e-9    # relative/absolute tolerance to declare two distances tied
_ZERO = 1e-12  # distance considered to be 0
_DEFAULT = 2




def preprocess_numerical_data(value_test, combi_test, proportion_test, col_num):
    """Compute the cumulative probabilities for numerical columns."""
    repartition_function_num = []
    values_num = []
    
    for id in col_num.index:
        mask = (combi_test == id)
        values_variable = value_test[mask].astype(float)
        order = np.argsort(values_variable)
        cum_prob = np.cumsum(proportion_test[mask][order])
        temp = np.concatenate([[0.0], cum_prob])
        centered_cum_prob = (temp[:-1] + temp[1:]) / 2.0
        
        repartition_function_num.append(centered_cum_prob)
        values_num.append(values_variable[order])
    
    return repartition_function_num, values_num


def _transform_continuous(x: np.ndarray, values: np.ndarray, cdf: np.ndarray,
                          interpolation: str) -> np.ndarray:
    """Transform continuous values using the cumulative distribution. Vectorized function

    interpolation = "linear": proper piecewise-linear interpolation of the CDF between
                              the observed values (clamped outside their range)
    interpolation = "legacy": the exact formula of transform_continuous, in which the
                              interpolation weight is (v - inf_v) / (sup_v - inf_v + 1e-3)
                              with inf_v, sup_v the *positions* (not the values) in `values`
    """
    pos = np.searchsorted(values, x, side="right")   # number of observed values <= x
    inf = np.maximum(pos - 1, 0)                      # last index with value <= x (0 if none)
    sup = np.minimum(pos, len(values) - 1)            # first index with value > x (last if none)
    if interpolation == "legacy":
        w = (x - inf) / (sup - inf + 1e-3)
    else:
        lo, hi = values[inf], values[sup]
        denom = hi - lo
        w = np.divide(x - lo, denom, out=np.zeros_like(x), where=denom > 0)
        w = np.clip(w, 0.0, 1.0)
    out = cdf[inf] + w * (cdf[sup] - cdf[inf])
    return np.where(np.isnan(x), 0.5, out)


def _encode(
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    cdfs: list[np.ndarray],
    values_num: list[np.ndarray],
    num_cols: list[str],
    cat_cols: list[str],
    cat_ref: pd.DataFrame,
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode (generated, train, test) as float matrices; Euclidean distance on them is d."""
    parts: list[list[np.ndarray]] = [[], [], []]

    for col, values, cdf in zip(num_cols, values_num, cdfs):
        for k, df in enumerate(frames):
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            parts[k].append(_transform_continuous(x, values, cdf, interpolation)[:, None])

    for c in cat_cols:
        cats = pd.Index(cat_ref[c].astype(str))
        for k, df in enumerate(frames):
            codes = pd.Categorical(df[c].astype(str), categories=cats).codes  # -1 = unknown
            onehot = np.zeros((len(df), len(cats)))
            known = codes >= 0
            onehot[np.flatnonzero(known), codes[known]] = 1.0
            parts[k].append(onehot / np.sqrt(2.0))
    return tuple(np.hstack(p) if p else np.zeros((len(f), 0)) for p, f in zip(parts, frames))



# --------------------------------------------------------------------------- #
# Nearest neighbours with exact distances
# --------------------------------------------------------------------------- #
def _knn(Xq: np.ndarray, Xref: np.ndarray, k: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """k nearest rows of Xref for each row of Xq: (exact distances, indices), sorted ascending."""
    k = min(k, len(Xref))
    nn = NearestNeighbors(n_neighbors=k).fit(Xref)
    dist = np.empty((len(Xq), k))
    idx = np.empty((len(Xq), k), dtype=np.int64)
    for s in range(0, len(Xq), chunk):
        q = Xq[s:s + chunk]
        _, i = nn.kneighbors(q)
        d = np.linalg.norm(Xref[i] - q[:, None, :], axis=2)  # exact re-measure of the candidates
        o = np.argsort(d, axis=1, kind="stable")
        dist[s:s + chunk] = np.take_along_axis(d, o, axis=1)
        idx[s:s + chunk] = np.take_along_axis(i, o, axis=1)
    return dist, idx


def _tied_indices(i: int, q: np.ndarray, Xref: np.ndarray, idx: np.ndarray,
                  mask: np.ndarray, saturated: bool) -> np.ndarray:
    """Indices of every row of Xref at the minimal distance from q."""
    if not saturated:
        return idx[i][mask[i]]
    full = np.linalg.norm(Xref - q, axis=1)  # all k candidates tied: scan the whole set
    return np.flatnonzero(full <= full.min() + _TOL * (1.0 + full.min()))


# --------------------------------------------------------------------------- #
# Distances and score
# --------------------------------------------------------------------------- #
def compute_distances(
    generated_data: pd.DataFrame,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    df_info: pd.DataFrame,
    proportion_test_files: tuple(np.ndarray, np.ndarray, np.ndarray),
    dict_unique_values: dict[np.ndarray],
    interpolation: str = "linear",
    n_candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per generated row: (d_train, d_test, d_data). See compute_score for the parameters."""
    if interpolation not in ("linear", "legacy"):
        raise ValueError("interpolation must be 'linear' or 'legacy'")
    
    print("Computing learning transfer...")

    col_cat = list(df_info.loc[df_info["Type"].isin(_CAT_TYPES), "Variable_name"])
    col_num = df_info.loc[df_info["Type"].isin(_NUM_TYPES)]["Variable_name"]
    num_info = df_info.loc[df_info["Type"].isin(_NUM_TYPES), "Variable_name"]  # index = variable id

    for name, frame in (("generated_data", generated_data), ("training_data", training_data),
                        ("test_data", test_data)):
        missing = [c for c in col_cat + list(num_info) if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} is missing columns described in df_info: {missing}")

    for file in proportion_test_files:
        assert len(file.shape)==1

    proportion_test, value_test, combis_list = proportion_test_files
    
    repartition_function_num, values_num = preprocess_numerical_data(value_test, combis_list, proportion_test, col_num)

    Xg, Xtr, Xte = _encode((generated_data, training_data, test_data), repartition_function_num, values_num, col_num, col_cat,
                           dict_unique_values, interpolation)

    print(f"Computation of {n_candidates} closest neighbours..")
    d_tr, i_tr = _knn(Xg, Xtr, n_candidates)
    d_te, i_te = _knn(Xg, Xte, n_candidates)
    print("Found !")
    
    d_train, d_test = d_tr[:, 0], d_te[:, 0]
    
    keep_idx = ~((d_tr[:,0]<_ZERO)&((d_te[:,0]<_ZERO)))

    print("Compute datasets distance")
    d_data_list = []
    for i in np.arange(len(generated_data))[keep_idx]:
        id_tr = i_tr[i,d_tr[i]<(d_tr[i,0]+_TOL)]
        id_te = i_te[i,d_te[i]<(d_te[i,0]+_TOL)]
        res = []
        for id1 in id_tr:
            res.append([np.linalg.norm(Xtr[id1] - Xte[id2]) for id2 in id_te])
        d_data_list.append(np.min(res))    
    d_data = np.array(d_data_list)
    
    assert (((d_train[keep_idx] - d_test[keep_idx])==0)[d_data==0]).all()
    
    d_data[d_data==0] = _DEFAULT


    return d_train[keep_idx], d_test[keep_idx], d_data

def _generate_histogram_distance(array_values, path):
    """Generate and save a histogram of DCR values."""
    plt.figure(figsize=[7,7])
    plt.hist(array_values,20,label=set)
    plt.title(f'Distribution Distances')
    plt.legend()
    plt.savefig(f"{path}/hist_Distances.png")
    plt.close()

def compute_score(
    generated_data: pd.DataFrame,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    df_info: pd.DataFrame,
    proportion_test_files: tuple(np.ndarray, np.ndarray, np.ndarray),
    dict_unique_values: dict[np.ndarray],
    path_saving:str,
    interpolation: str = "linear",
    n_candidates: int = 8,
) -> float:
    """Mean over generated rows of (d_test - d_train + d_data) / (2 * d_data).

    Parameters
    ----------
    df_info : one row per variable, columns "Variable_name" and "Type"
        (binary / cat / bool -> categorical, cont / int -> numerical). The index of df_info is
        the variable id used in the proportion files.
    proportion_test_file : path prefix of the files read by load_proportion_data
        (<prefix>_1.npy, <prefix>_1_comb.npy, <prefix>_1_values.npy). If None, the same
        distribution is computed from the full `test_data` instead.
    balance_train_test : crop train and test to the same size (first rows), as in
        Distance_to_Closest_Records, so a larger set is not closer merely because it is larger.
    interpolation : "linear" (proper interpolation of the CDF between observed values) or
        "legacy" (exact formula of transform_continuous, which interpolates with array
        positions instead of values; reproduces the original DCR numbers).
    n_candidates : neighbours retrieved per row to detect ties; rows whose candidates are all
        tied fall back to an exact full scan, so the result does not depend on this value.

    Generated rows present in both train and test (d_train = d_test = 0) are omitted and the
    mean is taken over the remaining rows. If the nearest train and test records coincide
    (d_data = 0) the row is equidistant and scores 0.5.

    Train/test proximity score for a generated tabular dataset.

    For every generated row x:

        d_train(x) = min_{y in X_train} d(x, y)
        d_test(x)  = min_{z in X_test}  d(x, z)
        d_data(x)  = min d(y, z)  over y in argmin_train(x), z in argmin_test(x)

        score(x) = (d_test - d_train + d_data) / (2 * d_data)

    The returned value is the mean of score(x) over the retained generated rows
    (rows that appear in both the training and the test set are omitted).

    By the triangle inequality d_data >= |d_test - d_train|, so each score lies in [0, 1]:
        * ~1   -> generated row sits much closer to the training data (memorisation)
        * 0.5  -> equidistant from train and test
        * ~0   -> generated row sits much closer to the test data

    Representation (same as Distance_to_Closest_Records):
        * numerical columns (df_info Type "float" / "int"): value -> mid-point CDF of the test
        distribution, F_j - p_j / 2, linearly interpolated between observed values
        * categorical columns (Type "binary" / "category" / "bool"): one-hot (categories of the full test
        set, unknown category -> zeros), divided by sqrt(2) so a mismatch costs 1
        * d = Euclidean distance on the concatenation

    Efficiency: nearest neighbours come from scikit-learn (one fit per reference set, queries in
    chunks); the k closest candidates are re-measured exactly (the BLAS distance trick is not exact
    enough to detect distance 0), and everything else is vectorised. Only rows whose nearest
    neighbour is tied are resolved one by one.
    """
    d_train, d_test, d_data = compute_distances(
        generated_data, training_data, test_data, df_info, proportion_test_files,
        dict_unique_values, interpolation, n_candidates)

    res = (d_test-d_train+d_data) / (2*d_data)
    
    _generate_histogram_distance(res, path_saving)

    return res.mean(), d_train, d_test, d_data