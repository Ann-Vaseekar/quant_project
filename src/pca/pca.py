import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.helpers.helpers import (
    filter_lifetime_missingness, filter_large_gaps, enforce_time_t_eligibility, fn_freeze_universe_monthly
)


def calc_resid_pca(
    daily_rets, n_components=3, window_size=60, plot_variance=False,
    min_obs_ratio=0.8,
    max_gap=150,
    freeze_universe_monthly=True,
    lifetime_threshold=2000
):
    """
    function that runs pca to calculate residual returns
    daily_rets (DataFrame): daily returns for securities
    n_components (int): number of pca components to estimate market/factor component
    window_size (int): number of days for rolling window pca calculation
    """

    ret = filter_lifetime_missingness(daily_rets, lifetime_threshold=lifetime_threshold)

    ret = filter_large_gaps(ret, max_gap=max_gap)

    threshold = int(window_size * 0.9)
    valid_cols = (np.sum(~daily_rets.isna(), axis=0) >= threshold)
    valid_idx = np.where(valid_cols.to_numpy())[0]

    # Convert to NumPy and fill NaNs
    daily_rets_clean = daily_rets.iloc[:, valid_idx].to_numpy()
    daily_rets_clean = np.nan_to_num(daily_rets_clean, nan=0.0)

    resid_list = []
    rolling_pca = []

    for i in range(window_size, len(daily_rets_clean)):
        # Slice window
        window_data = daily_rets_clean[i-window_size:i]
        curr = daily_rets_clean[i:i+1]

        # Standardize
        mean = window_data.mean(axis=0)
        std = window_data.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        window_scaled = (window_data - mean) / std
        curr_scaled = (curr - mean) / std

        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(window_scaled)

        scores_curr = pca.transform(curr_scaled)
        common_curr = np.dot(scores_curr, pca.components_)
        residual_today = curr_scaled - common_curr

        resid_list.append(residual_today)
        rolling_pca.append(pca.explained_variance_ratio_[:3])

    # Stack residuals
    resid_array = np.vstack(resid_list)

    # Create DataFrame
    resid_df = pd.DataFrame(
        resid_array,
        index=daily_rets.index[window_size:],
        columns=daily_rets.columns[valid_idx]
    )


    resid_df = enforce_time_t_eligibility(resid_df, ret)
    
    if freeze_universe_monthly:
        resid_df = fn_freeze_universe_monthly(resid_df)

    if plot_variance:

        cols = [f"PC{i}" for i in range(1, n_components+1)]

        rolling_pca_df = pd.DataFrame(rolling_pca, columns=cols)

        plt.figure(figsize=(12, 6))
        for col in rolling_pca_df.columns:
            plt.plot(rolling_pca_df.index, rolling_pca_df[col], label=col)
        plt.xlabel('Time')
        plt.ylabel('Explained Variance Ratio')
        plt.title(f"Rolling Explained Variance Ratio ({window_size}-day window)")
        plt.legend()
        plt.grid(True)
        plt.show()

    return resid_df



def plot_explained_variance(daily_rets, n_comps=3):

    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(daily_rets)

    pca = PCA()

    pca.fit_transform(returns_scaled)

    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
            np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Scree Plot: Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

    # Print explained variance for first 3 components
    print(f"Explained variance ratio for first {n_comps} components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:n_comps]):
        print(f"PC{i+1}: {ratio:.3f}")



def plot_loadings(daily_rets):

    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(daily_rets)

    pca = PCA()
    pca.fit_transform(returns_scaled)

    # Get component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(daily_rets.columns))],
        index=daily_rets.columns
    )

    # Plot loadings for first 3 PCs on a single plot
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.plot(loadings.index, loadings[f'PC{i+1}'], 
                marker='o', 
                linewidth=2, 
                markersize=8,
                label=f'PC{i+1}',
                )

    plt.title('Component Loadings for First 3 Principal Components')
    plt.xlabel('Stocks')
    plt.ylabel('Loading Value')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Print top contributors for each PC
    print("Top contributors to each principal component:")
    for i in range(3):
        print(f"\nPC{i+1}:")
        top_contributors = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(3)
        for stock, loading in top_contributors.items():
            print(f"{stock}: {loading:.3f}")