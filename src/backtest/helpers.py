import numpy as np
import pandas as pd
import logging
import src.analysis.stats as stats
from src.ols.ols import calc_resid_ols
from src.pca.pca import calc_resid_pca
from src.signal_construction.transform_signal import transform_signal, dollar_neutral_weights
from src.signal_construction.reduce_trading import partial_adjustment_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MKT_TICKER   = "BTCUSDT"
N_COMPS      = [1, 2, 3]
TCOST_BPS    = 20
RANK_THRESH  = 0.2
RHO          = 0.0


def sharpe(ret: pd.Series, ann_factor: float) -> float:
    """
    Computes sharpe ratio.

    Parameters:
        rets (df): Return matrix at bar frequency.
        ann_factor (float): Annualisation factor (bars per year).
    """
    if ret.std() == 0:
        return np.nan
    return ret.mean() / ret.std() * np.sqrt(ann_factor)


def resample_to_freq(px: pd.DataFrame, freq: str = "1d") -> pd.DataFrame:
    """
    Resample close prices to a given frequency and compute returns.
    Uses the last bar of each period as the close.

    Parameters:
        px (df): Prices matrix.
        freq (str): Frequency to resample to.
    """
    px_resampled = px.resample(freq).last()
    px_resampled = px_resampled.dropna(how="all")
    logger.info(f"Resampled to {freq}")
    rets = px_resampled.pct_change().dropna(how="all")
    return rets


def freq_config(
        px: pd.DataFrame, freq: str, window_sizes: list | int = None
        ) -> tuple[pd.DataFrame, float, list, callable]:
    """
    Get rets, annualisation factor, window sizes, and window label
    function from the input price DataFrame and frequency string.

    Parameters:
        px (df): Prices matrix.
        freq (str): Frequency to resample to.
        window_sizes (list/int): Length of rolling window calculation (no. of bars).
    """
    rets         = resample_to_freq(px, freq)
    if freq[-1] == "d":
        num_days     = int(freq.replace("d", ""))
        bar_hours    = 24 * num_days
        bars_pd      = 365 // num_days
        if not window_sizes:
            window_sizes = [m * 30 for m in range(1, 5)]  
        window_label = lambda w: w // 30
    else:
        bar_hours    = int(freq.replace("h", ""))
        bars_per_day = 24 // bar_hours
        bars_pd      = 365 * bars_per_day
        if not window_sizes:
            window_sizes = [int(m * 15 * bars_per_day) for m in range(1, 5)] 
        window_label = lambda w: (w / bars_per_day / 30)

    return rets, bar_hours, bars_pd, window_sizes, window_label


def avg_holding_period(w: pd.DataFrame, bar_hours: int = 4) -> dict:
    """
    Compute average holding period from weight matrix.
    Returns holding period in bars, hours, and days.
    """
    # One-way turnover per bar
    to = (w.fillna(0) - w.shift().fillna(0)).abs().sum(axis=1) / 2
    to = to[to > 0]  # exclude bars with no trading

    avg_to        = to.mean()
    hp_bars       = 1 / avg_to
    hp_hours      = hp_bars * bar_hours
    hp_days       = hp_hours / 24

    return {"bars": hp_bars, "hours": hp_hours, "days": hp_days}


def compute_resid_method(
    rets: pd.DataFrame,
    method: str,
    window_size: int,
    n_comp: int,
    mkt_ticker: str
):
    """
    Compute residuals using method specified.

    Parameters:
        rets (df): Return matrix at bar frequency.
        method (str): Method to compute residual (pca/ols)
        n_comp (int): Number of PCA components. Required for method="pca".
        window_size (int): Rolling window in bars.
        mkt_ticker (str or None): Benchmark ticker. Required for method="ols".
    """

    if method == "pca":
        return calc_resid_pca(rets, n_components=n_comp, window_size=window_size)
    elif method == "ols":
        return calc_resid_ols(rets, window_size=window_size, mkt_ticker=mkt_ticker)
    else:
        raise ValueError("residual computing method invalid")


def run_one(
    rets: pd.DataFrame,
    method: str,
    window_size: int,
    ann_factor: float,
    oos_start: str = None,
    n_comp: int = None,
    mkt_ticker: str = None,   
    bar_hours: int = 4,
    alpha: float = 1.0,
    rho: float = RHO,
    rank_thresh: float = RANK_THRESH,
    resid_df: pd.DataFrame = None,
) -> dict:
    """
    Run a single (n_comp, window_size) backtest.

    Parameters:
        rets (df): Return matrix at bar frequency.
        method (str): Method to compute residual (pca/ols)
        n_comp (int): Number of PCA components. Required for method="pca".
        window_size (int): Rolling window in bars.
        ann_factor (float): Annualisation factor (bars per year).
        oos_start (str or None): If set, stats computed on OOS slice only.
            Signal always computed on full history to avoid cold-start.
        mkt_ticker (str or None): Benchmark ticker. Required for method="ols".
        resid_df (df or none): Pre-computed residuals. If None, computed internally.
    """
    if resid_df is None:
        resid_df = compute_resid_method(
		      rets=rets, method=method, window_size=window_size,
		      n_comp=n_comp, mkt_ticker=mkt_ticker
		   )
    resid_smoothed = resid_df.ewm(alpha=alpha).mean()
    signal   = transform_signal(-resid_smoothed, how="rank", rank_thresh=rank_thresh)
    w        = dollar_neutral_weights(signal)
    w        = partial_adjustment_weights(w, rho=rho)
    w        = w.sub(w.mean(axis=1), axis=0)

    w_lag     = w.shift(1)
    gross_ret = (w_lag * rets).sum(axis=1)
    to        = (w.fillna(0) - w.shift().fillna(0)).abs().sum(axis=1) / 2
    net_ret   = gross_ret - to * TCOST_BPS * 1e-4
    to        = to.iloc[1:]  # drop cold start bar
    net_ret   = net_ret.iloc[1:]  # drop cold start bar

    if oos_start is not None:
        gross_ret = gross_ret.loc[oos_start:]
        net_ret   = net_ret.loc[oos_start:]
        to        = to.loc[oos_start:]
        if len(gross_ret) == 0:
            raise ValueError(f"No data after oos_start={oos_start}")
        bench = rets.loc[oos_start:, MKT_TICKER]
    else:
        bench = rets[MKT_TICKER]

    full_stats = stats.compute_full_stats(gross_ret, bench)
    holding_info = avg_holding_period(w, bar_hours=bar_hours)

    return {
        # Series
        "gross_cum":            (1 + gross_ret).cumprod() - 1,
        "net_cum":              (1 + net_ret).cumprod() - 1,
        "gross_ret":            gross_ret,
        "net_ret":              net_ret,
        "avg_holding_days":     holding_info["days"],
        "to":                   to,
        "ic":                   {
                                      h: resid_df.rank(axis=1).corrwith(rets.shift(-h), axis=1, method="spearman").dropna().mean()
                                      for h in [1, 2, 4, 6, 12, 24]
                                },
        # Scalars
        "gross_sharpe":         sharpe(gross_ret, ann_factor),
        "net_sharpe":           sharpe(net_ret,   ann_factor),
        "avg_ann":              full_stats["avg_ann"].iloc[0],
        "vol_ann":              full_stats["vol_ann"].iloc[0],
        "alpha_ann":            full_stats["alpha_ann"].iloc[0],
        "alpha_tstat":          full_stats["alpha_tstat"].iloc[0],
        "beta":                 full_stats["beta"].iloc[0],
        "hit_rate":             full_stats["hit_rate"].iloc[0],
        "r_squared":            full_stats["r_squared"].iloc[0],
        "avg_turnover":         to.mean(),
        "cost_drag_ann":        (to * TCOST_BPS * 1e-4).mean() * ann_factor,
        "n_obs":                len(gross_ret),
    }


def results_to_scalar_df(results: dict, stat: str) -> pd.DataFrame:
    """
    Extract a scalar stat: rows=window months, columns=n_comp or mkt_ticker.
    """
    return pd.DataFrame(
        {col_key: {w: v[stat] for w, v in wins.items()}
         for col_key, wins in results.items()}
    )


def results_to_series(results: dict, stat: str, col_key) -> pd.DataFrame:
    """
    Extract a time-series stat for one n_comp or mkt_ticker:
    rows=dates, columns=window months.
    """
    return pd.DataFrame(
        {w: v[stat] for w, v in results[col_key].items()}
    )

