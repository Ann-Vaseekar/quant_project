import pandas as pd
import numpy as np


def compute_stats(rets):
    stats = {}
    stats["avg"] = np.mean(rets)
    stats["hit_rate"] = sum([x>0 for x in rets]) / len(rets)
    stats["max_ret"] = max(rets)
    return pd.DataFrame(stats)


def compute_annualised_stats(rets):

    stats = {}
    stats["avg"] = rets.mean()*252
    stats["vol"] = rets.std()*np.sqrt(252)
    stats["sharpe"] = stats["avg"]/stats["vol"]
    stats["hit_rate"] = (rets>0).mean()

    stats = pd.DataFrame(stats)

    return stats


def analyze_signal(rets,signal):
    analysis = {}

    pos_rets = []
    neg_rets = []
    for i in range(len(rets)):
        if signal[i] > 1:
            pos_rets.append(rets[i])
        elif signal[i] < -1:
            neg_rets.append(rets[i])

    analysis["pos_ret"] = np.mean(pos_rets)
    analysis["neg_ret"] = np.mean(neg_rets)
    analysis["spread"] = analysis["pos_ret"] - analysis["neg_ret"]

    return pd.DataFrame(analysis)


def compute_alpha(ret, mkt_ticker="BTCUSDT")
    corr = ret.rolling(252).corr(ret[mkt_ticker])
    vol = ret.rolling(252).std()
    beta = (corr*vol).divide(vol[mkt_ticker],axis=0)
    resid = ret - beta.multiply(ret[mkt_ticker],0)

    return resid