

import numpy as np
import pandas as pd
from src.helpers.helpers import (
    filter_lifetime_missingness, filter_large_gaps, enforce_time_t_eligibility, fn_freeze_universe_monthly
)

import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def compute_alpha_adj_resid(
    ret,
    window_size=400,
    mkt_ticker="BTCUSDT",
    min_obs_ratio=0.8
):
    """
    Compute rolling residuals using rolling OLS regression per asset vs market.
    Returns a DataFrame of residuals (alpha-adjusted).
    """
    min_obs = int(window_size * min_obs_ratio)
    resid = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)

    # Clean market returns
    market_ret = ret[mkt_ticker].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()

    asset_cols = [c for c in ret.columns if c != mkt_ticker]
    asset_rets = ret[asset_cols].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()

    n_rows, n_assets = asset_rets.shape

    # Rolling regression per asset
    for col_idx in range(n_assets):
        y = asset_rets[:, col_idx]

        # Only iterate over time once per asset
        for i in range(window_size-1, n_rows):
            y_window = y[i-window_size+1:i+1]
            x_window = market_ret[i-window_size+1:i+1]

            # Stack constant for intercept
            X_window = np.column_stack([np.ones(len(x_window)), x_window])

            # Drop rows with NaN
            mask = ~np.isnan(y_window) & ~np.isnan(x_window)
            if mask.sum() < min_obs:
                continue

            X_clean = X_window[mask]
            y_clean = y_window[mask]

            # Solve OLS via numpy least squares (fast)
            beta_hat = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]

            # Residual = last observation minus predicted
            resid_value = y_clean[-1] - (beta_hat[0] + beta_hat[1]*x_window[mask][-1])
            resid.iloc[i, ret.columns.get_loc(asset_cols[col_idx])] = resid_value

    return resid

from statsmodels.regression.rolling import RollingOLS

def compute_beta_sm(ret,
                 window_size=400,
                 mkt_ticker="BTCUSDT",
                 min_obs_ratio=0.8,
                 hac_lags=5):
    """
    Rolling regression beta of each asset vs market using RollingOLS.
    Uses HAC (Newey-West) covariance for crypto returns.

    Returns
    -------
    DataFrame of rolling betas
    """

    min_obs = int(window_size * min_obs_ratio)

    # Optional but recommended: convert to log returns
    # ret = np.log1p(ret)

    assets = [c for c in ret.columns if c != mkt_ticker]

    beta_df = pd.DataFrame(index=ret.index, columns=assets, dtype=float)

    for asset in assets:

        y = ret[asset]

        X = ret[[mkt_ticker]].copy()
        X = sm.add_constant(X)

        model = RollingOLS(
            y,
            X,
            window=window_size,
            min_nobs=min_obs
        )

        results = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags}
        )

        beta_df[asset] = results.params[mkt_ticker]

    return beta_df

def compute_beta(ret, window_size=400, mkt_ticker="BTCUSDT", min_obs_ratio=0.8):
    """
    Compute rolling beta using rolling covariance with market divided by rolling variance.
    
    Returns:
        DataFrame with rolling beta values.
    """
    min_obs = int(window_size * min_obs_ratio)
    
    cov = ret.rolling(window_size, min_periods=min_obs).cov(ret[mkt_ticker])
    var_m = ret[mkt_ticker].rolling(window_size, min_periods=min_obs).var()
    
    beta = cov.divide(var_m, axis=0)
    return beta


def compute_resid(
    ret,
    window_size=400,
    mkt_ticker="BTCUSDT",
    min_obs_ratio=0.8,
    max_gap=50,
    freeze_universe_monthly=True,
    lifetime_threshold=150,
    alpha_adjusted=False
):
    """
    Compute market-neutral residuals using rolling OLS regression.
    
    Steps:
    1. Pre-filter assets with extreme lifetime missingness
    2. Remove assets with large consecutive gaps
    3. Compute rolling beta
    4. Calculate residuals and enforce time-t eligibility
    5. Optionally freeze universe monthly
    """

    ret = filter_lifetime_missingness(ret, lifetime_threshold=lifetime_threshold)
    
    ret = filter_large_gaps(ret, max_gap=max_gap)

    # linear interpolation to fill small gaps for beta calculation
    #ret.interpolate(method='linear', inplace=True)

    if alpha_adjusted:
        resid = compute_alpha_adj_resid(
            ret,
            window_size=window_size,
            mkt_ticker=mkt_ticker,
            min_obs_ratio=min_obs_ratio
        )  
    else:
        
        beta = compute_beta_sm(ret, window_size=window_size, mkt_ticker=mkt_ticker, min_obs_ratio=min_obs_ratio)
    
        resid = ret - beta.multiply(ret[mkt_ticker], axis=0)
    
    # resid = enforce_time_t_eligibility(resid, ret)
    
    # if freeze_universe_monthly:
    #     resid = fn_freeze_universe_monthly(resid)
    
    # print("After computing residuals, final shape is:", resid.shape)
    
    return resid



# def compute_resid(ret, window_size=90, mkt_ticker="BTCUSDT"):
    
#     # Rolling covariance with market
#     cov = ret.rolling(window_size, min_periods=int(window_size * 0.8)).cov(ret[mkt_ticker])
    
#     # Rolling market variance
#     var_m = ret[mkt_ticker].rolling(window_size, min_periods=int(window_size * 0.8)).var()
    
#     # Beta
#     beta = cov.divide(var_m, axis=0)
    
#     # Market-neutral returns
#     resid = ret - beta.multiply(ret[mkt_ticker], axis=0)
    
#     return resid