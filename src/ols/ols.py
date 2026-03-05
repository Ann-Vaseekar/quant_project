

import numpy as np
from src.helpers.helpers import (
    filter_lifetime_missingness, filter_large_gaps, enforce_time_t_eligibility, fn_freeze_universe_monthly
)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_residual_fast(
    ret,
    window_size=400,
    mkt_ticker="BTCUSDT",
    min_obs_ratio=0.8,
):
    """
    Fast rolling market-neutral residuals using:
        beta = rolling_cov(asset, market) / rolling_var(market)

    ret (df): Return matrix
    window_size (int): Rolling window length
    mkt_ticker (str): Market column name
    min_obs_ratio (float): Minimum fraction of window required

    """

    ret = ret.copy().astype(float)
    min_obs = int(window_size * min_obs_ratio)

    if mkt_ticker not in ret.columns:
        raise ValueError(f"{mkt_ticker} not in return columns")

    mkt = ret[mkt_ticker]

    # Rolling variance of market
    var_m = (
        mkt.rolling(window_size, min_periods=min_obs)
        .var()
        .replace(0, np.nan)
    )
    cov = ret.rolling(window_size, min_periods=min_obs).cov(mkt)
    beta = cov.divide(var_m, axis=0)
    resid = ret - beta.multiply(mkt, axis=0)

    resid = resid.drop(columns=[mkt_ticker])

    return resid


def calc_resid_ols(
    ret,
    window_size=400,
    mkt_ticker="BTCUSDT",
    min_obs_ratio=0.8,
    max_gap=150,
    freeze_universe_monthly=True,
    lifetime_threshold=150,
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

    resid = compute_residual_fast(
        ret,
        window_size=window_size,
        mkt_ticker=mkt_ticker,
        min_obs_ratio=min_obs_ratio)
    
    resid = enforce_time_t_eligibility(resid, ret)
    
    if freeze_universe_monthly:
        resid = fn_freeze_universe_monthly(resid)

    
    return resid
