import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def max_consecutive_nans(series):
    is_nan = series.isna().astype(int)
    groups = (is_nan != is_nan.shift()).cumsum()
    return is_nan.groupby(groups).sum().max()


def filter_lifetime_missingness(ret, lifetime_threshold=2000):
    """
    Pre-filter assets with extreme lifetime missingness.
    
    Keeps assets with at least (total_obs - lifetime_threshold) observations.
    """
    total_obs = len(ret)
    lifetime_valid = ret.count()
    keep_assets = (lifetime_valid >= (total_obs - lifetime_threshold))
    return ret.loc[:, keep_assets]


def filter_large_gaps(ret, max_gap=120):
    """
    Remove assets with large consecutive gaps of NaN values.
    """
    gap_filter = []
    for col in ret.columns:
        if max_consecutive_nans(ret[col]) <= max_gap:
            gap_filter.append(col)
    return ret[gap_filter]


def enforce_time_t_eligibility(resid, ret):
    """
    Enforce that residuals are only valid when the underlying return is valid.
    """
    valid_today = ret.notna()
    return resid.where(valid_today)


def fn_freeze_universe_monthly(resid):
    """
    Freeze the universe monthly: assets are eligible for an entire month 
    if they were eligible on the first day of that month.
    """
    month = resid.index.to_period("M")
    frozen_mask = pd.DataFrame(False, index=resid.index, columns=resid.columns)
    
    for m in month.unique():
        month_idx = month == m
        first_day = resid.index[month_idx][0]
        eligible = resid.loc[first_day].notna()
        frozen_mask.loc[month_idx, eligible.index] = eligible.values
    
    return resid.where(frozen_mask)

def kill_flat_tails(ret, window=50, tol=1e-8):
    rolling_var = ret.rolling(window).var()
    dead_mask = rolling_var < tol
    
    ret_clean = ret.copy()
    
    for col in ret.columns:
        dead_idx = dead_mask[col]
        if dead_idx.any():
            first_dead = dead_idx.idxmax()  # first True
            if dead_idx.loc[first_dead]:
                ret_clean.loc[first_dead:, col] = np.nan
    
    return ret_clean