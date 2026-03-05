import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.helpers.helpers import (
    filter_lifetime_missingness,
    filter_large_gaps,
    enforce_time_t_eligibility,
    fn_freeze_universe_monthly,
)


def calc_resid_pca(
    daily_rets,
    n_components=3,
    window_size=60,
    min_obs_ratio=0.8,
    max_gap=150,
    freeze_universe_monthly=True,
    lifetime_threshold=2000,
    refit_every=None,
):
    """
    Compute rolling PCA-based market-neutral residual returns.

    daily_rets (df):                Raw return matrix 
    n_components (int):             Number of PCA components to remove
    window_size (int):              Rolling window length in bars
    min_obs_ratio (float):          Min fraction of window required for a column to be included
    max_gap (int):                  Max consecutive NaN gap allowed per asset
    freeze_universe_monthly (bool): If True, freezes tradeable universe monthly
    lifetime_threshold (int):       Min total observations required per asset
    refit_every (int/None):         Refit PCA every N bars. None = every bar (original behaviour).
                                    E.g. refit_every=6 refits once per day on 4h data.

    """

    ret = filter_lifetime_missingness(daily_rets, lifetime_threshold=lifetime_threshold)
    ret = filter_large_gaps(ret, max_gap=max_gap)

    min_obs = int(window_size * min_obs_ratio)
    valid_cols_mask = ret.notna().sum(axis=0) >= min_obs
    ret = ret.loc[:, valid_cols_mask]

    ret_np = ret.to_numpy().astype(float)         # (T, N)
    was_nan = np.isnan(ret_np)                    # track original missingness

    # Fill NaN with 0 only for computation; we will re-mask at the end
    ret_filled = np.where(was_nan, 0.0, ret_np)   # (T, N)

    n_bars, n_assets = ret_filled.shape

    resid_list = []
    rolling_pca_list = []

    # Cache for refitting
    _pca_cache = None
    _mean_cache = None
    _std_cache = None

    for i in range(window_size, n_bars):

        should_refit = (
            _pca_cache is None
            or refit_every is None
            or ((i - window_size) % refit_every == 0)
        )

        if should_refit:
            window_data = ret_filled[i - window_size: i]          # (W, N)

            mean = window_data.mean(axis=0)
            std = window_data.std(axis=0, ddof=1)
            std[std == 0] = 1.0                        

            window_scaled = (window_data - mean) / std

            pca = PCA(n_components=n_components)
            pca.fit(window_scaled)

            for k in range(n_components):
                if pca.components_[k, np.argmax(np.abs(pca.components_[k]))] < 0:
                    pca.components_[k] *= -1

            _pca_cache = pca
            _mean_cache = mean
            _std_cache = std

        # Apply cached (or fresh) PCA to current bar
        curr = ret_filled[i: i + 1]                               # (1, N)
        curr_scaled = (curr - _mean_cache) / _std_cache

        scores = _pca_cache.transform(curr_scaled)                # (1, n_components)
        common = np.dot(scores, _pca_cache.components_)           # (1, N)
        residual = curr_scaled - common                            # (1, N) — standardised space

        resid_list.append(residual)
        rolling_pca_list.append(_pca_cache.explained_variance_ratio_[:n_components].copy())

    resid_array = np.vstack(resid_list)                           # (T - W, N)

    # Re-mask positions where original data was NaN
    original_nan_mask = was_nan[window_size:]
    resid_array[original_nan_mask] = np.nan

    resid_df = pd.DataFrame(
        resid_array,
        index=ret.index[window_size:],
        columns=ret.columns,
    )

    resid_df = enforce_time_t_eligibility(resid_df, ret)

    if freeze_universe_monthly:
        resid_df = fn_freeze_universe_monthly(resid_df)

    return resid_df


def plot_explained_variance(daily_rets, n_comps=3):
    """
    Full-sample explained variance plot.
    """

    ret_clean = daily_rets.dropna(how="all", axis=1).fillna(0)
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(ret_clean)

    pca = PCA()
    pca.fit(returns_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        "bo-",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Scree Plot (full-sample — diagnostic only)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Explained variance for first {n_comps} components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:n_comps]):
        print(f"  PC{i+1}: {ratio:.3f}")


def calc_pca_loadings(
    rets,
    window_size,
    n_components = 1,
):
    """
    Compute rolling PCA loadings over time.
    Returns DataFrame of shape (n_bars, n_assets) for each component.
    """

    loadings = []

    for i in range(window_size, len(rets)):
        window = rets.iloc[i - window_size:i].fillna(0)
        
        scaler     = StandardScaler()
        scaled     = scaler.fit_transform(window)
        pca        = PCA(n_components=n_components)
        pca.fit(scaled)

        # PC1 loadings — flip sign so BTC loading is always positive
        pc1 = pca.components_[0]
        if pc1[rets.columns.get_loc("BTCUSDT")] < 0:
            pc1 = -pc1

        loadings.append(pc1)

    return pd.DataFrame(
        loadings,
        index=rets.index[window_size:],
        columns=rets.columns,
    )