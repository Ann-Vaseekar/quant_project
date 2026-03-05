import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import logging
from src.backtest.helpers import (
    resample_to_freq, compute_resid_method, run_one, freq_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

N_COMPS      = [1, 2, 3]
MKT_TICKERS  = ["BTCUSDT", "ETHUSDT"]
RANK_THRESH  = 0.2
RHO          = 0.0
ALPHA        = 1.0


def run_backtest(
    px: pd.DataFrame,
    method: str,
    freq: str = "4h",
    rho: float = RHO,
    alpha: float = ALPHA,
    rank_thresh: float = RANK_THRESH,
    n_comps: list = N_COMPS,
    mkt_tickers: list = MKT_TICKERS,
    window_sizes: list | int = None
) -> dict:
    """
    Run backtest.

    Parameters:
        px (df):                     Prices matrix at bar frequency.
        method (str):                Method to compute residual (pca/ols)
        n_comp (list):               Number of PCA components. Required for method="pca".
        window_sizes (list or int):  Rolling window in bars.
        ann_factor (float):          Annualisation factor (bars per year).
        oos_start (str or None):     If set, stats computed on OOS slice only.
                                     Signal always computed on full history to avoid cold-start.
        mkt_tickers (list):          Benchmark ticker. Required for method="ols".
    """
    rets, bar_hours, bars_pd, window_sizes, window_label = freq_config(px, freq, window_sizes)

    if method == "pca":
        method_param_name = "n_comp"
        method_params = n_comps
    else:  # ols
        method_param_name = "mkt_ticker"
        method_params = mkt_tickers

    results = {}
    for method_param in method_params:
        results[method_param] = {}
        for window_size in tqdm(window_sizes, desc=f"{method_param_name}={method_param}"):
            label = window_label(window_size)
            results[method_param][label] = run_one(
                    rets=rets,
                    method=method,
                    window_size=window_size,
                    ann_factor=bars_pd,
                    bar_hours=bar_hours,
                    alpha=alpha,
                    rho=rho,
                    rank_thresh=rank_thresh,
                    n_comp=method_param if method == "pca" else None,
                    mkt_ticker=method_param if method == "ols" else None,
                )
    return results


def run_backtest_opt(
    px: pd.DataFrame,
    param_grid: dict,
    method: str,
    freq: str = "4h",
    optimise_on: str = "net_sharpe",
) -> pd.DataFrame:
    """
    Sweeps all combinations of param_grid and returns a
    summary DataFrame sorted by optimise_on.
    """
    rets, bar_hours, bars_pd, _, _ = freq_config(px, freq)

    if method not in ["pca", "ols"]:
        raise ValueError("method must be 'pca' or 'ols'")

    window_sizes = param_grid["window_size"]
    rho_grid     = param_grid["rho"]
    alpha_grid   = param_grid["alpha"]
    thresh_grid  = param_grid["rank_thresh"]


    if method == "pca":
        method_param_name = "n_comp"
        method_param_grid = param_grid["n_comp"]
    else:  # ols
        method_param_name = "mkt_ticker"
        method_param_grid = param_grid["mkt_ticker"]

    print("Precomputing residuals...")
    resid_cache = {}
    for method_param, window_size in tqdm(
        list(itertools.product(method_param_grid, window_sizes)),
        desc="Residuals"
    ):
        resid_cache[(method_param, window_size)] = compute_resid_method(
            rets=rets,
            method=method,
            window_size=window_size,
            n_comp=method_param if method == "pca" else None,
            mkt_ticker=method_param if method == "ols" else None,
        )

    rows = []

    for (method_param, window_size), resid_df in tqdm(resid_cache.items(), desc="Sweep"):
        for alpha_, rho_, thresh_ in itertools.product(
            alpha_grid, rho_grid, thresh_grid
        ):
            try:
                results = run_one(
                    rets=rets,
                    method=method,
                    window_size=window_size,
                    ann_factor=bars_pd,
                    bar_hours=bar_hours,
                    alpha=alpha_,
                    rho=rho_,
                    rank_thresh=thresh_,
                    n_comp=method_param if method == "pca" else None,
                    mkt_ticker=method_param if method == "ols" else None,
                    resid_df=resid_df
                )

                rows.append({
                    method_param_name:        method_param,
                    "window_size":            window_size,
                    "rho":                    rho_,
                    "alpha_ewm":              alpha_,
                    "rank_thresh":            thresh_,
                    "gross_sharpe":           results["gross_sharpe"],
                    "net_sharpe":             results["net_sharpe"],
                    "avg_turnover":           results["avg_turnover"],
                    "avg_ann":                results["avg_ann"],
                    "vol_ann":                results["vol_ann"],
                    "alpha":                  results["alpha_ann"],
                    "beta":                   results["beta"],
                    "alpha_tstat":            results["alpha_tstat"],
                    "hit_rate":               results["hit_rate"],
                    "avg_holding_days":       results["avg_holding_days"],
                })

            except Exception as e:
                logger.warning(
                    f"Failed {method_param_name}={method_param}, "
                    f"window={window_size}, rho={rho_}, "
                    f"alpha={alpha_}, thresh={thresh_}: {e}"
                )


    summary = pd.DataFrame(rows)

    if summary.empty:
        raise ValueError("No valid parameter combinations produced results.")

    return summary.sort_values(optimise_on, ascending=False).reset_index(drop=True)


def run_backtest_oos(
    px: pd.DataFrame,
    method: str,
    freq: str,
    rho: float,
    rank_thresh: float,
    alpha: float,
    window_size: int,
    oos_start: str, 
    mkt_ticker: str=None,
    n_comp: int=None,
) -> dict:
    """
    Run oos backtest.

    Parameters:
        px (df):                     Prices matrix at bar frequency.
        method (str):                Method to compute residual (pca/ols)
        n_comp (list):               Number of PCA components. Required for method="pca".
        window_sizes (list or int):  Rolling window in bars.
        ann_factor (float):          Annualisation factor (bars per year).
        oos_start (str or None):     If set, stats computed on OOS slice only.
                                     Signal always computed on full history to avoid cold-start.
        mkt_tickers (list):          Benchmark ticker. Required for method="ols".
    """

    if method == "pca":
        if n_comp == None:
            raise ValueError("n_comps needs to be defined for PCA.")
    elif method == "ols":
        if mkt_ticker == None:
            raise ValueError("mkt_ticker needs to be defined for OLS.")
    else:
        raise ValueError("method invalid. set to either ols or pca")

    if freq == "1d":
        rets         = resample_to_freq(px, "1D")
        bar_hours    = 24
        bars_pd      = 365
    else:
        rets         = resample_to_freq(px, freq)
        bar_hours    = int(freq.replace("h", ""))
        bars_per_day = 24 // bar_hours
        bars_pd      = 365 * bars_per_day

    results = run_one(
        rets=rets,
        method=method,
        window_size=window_size,
        ann_factor=bars_pd,
        bar_hours=bar_hours,
        alpha=alpha,
        rho=rho,
        rank_thresh=rank_thresh,
        n_comp=n_comp,
        mkt_ticker=mkt_ticker,
        oos_start=oos_start
    )

    return results


