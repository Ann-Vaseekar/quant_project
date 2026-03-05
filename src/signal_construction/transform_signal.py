from scipy.stats import norm
import pandas as pd

def transform_signal(df, thresh=0.03, how="winsorize", rank_thresh=None):
    high = df.quantile(1 - thresh, axis=1)
    low = df.quantile(thresh, axis=1)

    valid_how = ["winsorize", "truncate", "rank", "inv_cdf"]

    if how not in valid_how:
        assert f"invalid how, must be in {valid_how}"

    if how == "winsorize":
        return df.clip(lower=low, upper=high, axis=0)

    elif how == "truncate":
        mask = (df.le(high, axis=0)) & (df.ge(low, axis=0))
        return df.where(mask, 0)
    
    elif how == "rank":
        # try 0.2, 0.3?

        ranked = df.rank(axis=1, pct=True)
        
        scaled = ranked.sub(0.5).mul(2)

        if rank_thresh is None:
            return scaled
        
        mask = (ranked <= rank_thresh) | (ranked >= 1 - rank_thresh)
        
        return scaled.where(mask, 0)
    
    elif how == "inv_cdf":

        ranked = df.rank(axis=1, pct=True)

        eps = 1e-6
        ranked = ranked.clip(eps, 1 - eps)

        return pd.DataFrame(norm.ppf(ranked), index=ranked.index, columns=ranked.columns)


def standardise(df, window, min_period=1):
    df_mean = df.rolling(window, min_periods=min_period).mean()
    df_std = df.rolling(window, min_periods=min_period).std()

    return (df-df_mean) / df_std


def dollar_neutral_weights(signal):
    signal = signal.sub(signal.mean(axis=1), axis=0)
    weights = signal.div(signal.abs().sum(axis=1), axis=0)
    return weights