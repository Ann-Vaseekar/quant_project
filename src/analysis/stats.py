import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

PERIODS_PER_YEAR = 2190  # 4H crypto


def compute_full_stats(rets, market_rets):

    # Force Series
    rets = pd.Series(rets).astype(float)
    market_rets = pd.Series(market_rets).astype(float)

    # Align & drop NaNs
    df = pd.concat([rets, market_rets], axis=1).dropna()
    df.columns = ["strategy", "market"]

    y = df["strategy"]
    X = sm.add_constant(df["market"])

    # model = sm.OLS(y, X).fit(
    #     cov_type="HAC",
    #     cov_kwds={"maxlags": 6}
    # )

    model = sm.OLS(y, X).fit()

    PERIODS_PER_YEAR = 2190  # 4H crypto

    stats = {}
    stats["avg_ann"] = y.mean() * PERIODS_PER_YEAR
    stats["vol_ann"] = y.std() * np.sqrt(PERIODS_PER_YEAR)
    stats["sharpe"] = stats["avg_ann"] / stats["vol_ann"]
    stats["hit_rate"] = (y > 0).mean()

    alpha_4h = model.params["const"]
    beta = model.params["market"]

    stats["alpha_ann"] = alpha_4h * PERIODS_PER_YEAR
    stats["alpha_tstat"] = model.tvalues["const"]
    stats["beta"] = beta
    stats["beta_tstat"] = model.tvalues["market"]
    stats["r_squared"] = model.rsquared

    return pd.DataFrame(stats, index=[0])




# def compute_annualised_stats(rets, market_rets):

#     stats = {}
#     stats["avg"] = rets.mean()*252
#     stats["vol"] = rets.std()*np.sqrt(252)
#     stats["sharpe"] = stats["avg"]/stats["vol"]
#     stats["hit_rate"] = (rets>0).mean()


#     # CAPM beta (same as daily)
#     cov = np.cov(rets, market_rets)[0, 1]
#     var_mkt = np.var(market_rets)
#     beta = cov / var_mkt

#     # Annualized alpha
#     daily_alpha = rets.mean() - beta * market_rets.mean()
#     stats["alpha"] = daily_alpha * 252
#     stats["beta"] = beta

#     return pd.DataFrame(stats, index=[0])


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


def compute_alpha(strategy_ret, market_ret, days=365):

    cov = strategy_ret.rolling(days).cov(market_ret)
    var_mkt = market_ret.rolling(days).var()

    beta = cov / var_mkt

    mean_strat = strategy_ret.rolling(days).mean()
    mean_mkt = market_ret.rolling(days).mean()

    alpha = mean_strat - beta * mean_mkt

    return beta, alpha



def drawdown(gross_ret):

    cumulative_returns = (1 + gross_ret).cumprod()

    dd = (cumulative_returns / cumulative_returns.cummax() - 1)

    dd.plot()
    plt.show()

    return dd



def drawdown_duration(gross_ret):

    cumulative_returns = (1 + gross_ret).cumprod()
    
    peak = cumulative_returns.cummax()

    duration = pd.Series(index=gross_ret.index, dtype=int)

    for i in range(1, len(cumulative_returns)):
        if cumulative_returns.iloc[i] < peak.iloc[i]:
            duration.iloc[i] = duration.iloc[i-1] + 1
        else:
            duration.iloc[i] = 0

    duration.plot()
    plt.show()

    return duration
