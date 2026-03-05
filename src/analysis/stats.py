import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def compute_full_stats(rets, market_rets, freq="4h"):

    rets = pd.Series(rets).astype(float)
    market_rets = pd.Series(market_rets).astype(float)

    df = pd.concat([rets, market_rets], axis=1).dropna()
    df.columns = ["strategy", "market"]

    y = df["strategy"]
    X = sm.add_constant(df["market"])

    model = sm.OLS(y, X).fit()

    period_hours = pd.Timedelta(freq).total_seconds() / 3600
    PERIODS_PER_YEAR = 365 * 24 / period_hours

    stats = {}
    stats["avg_ann"] = y.mean() * PERIODS_PER_YEAR
    stats["vol_ann"] = y.std() * np.sqrt(PERIODS_PER_YEAR)
    stats["sharpe"] = stats["avg_ann"] / stats["vol_ann"]
    stats["hit_rate"] = (y >= 0).mean()

    alpha_4h = model.params["const"]
    beta = model.params["market"]

    stats["alpha_ann"] = alpha_4h * PERIODS_PER_YEAR
    stats["alpha_tstat"] = model.tvalues["const"]
    stats["beta"] = beta
    stats["beta_tstat"] = model.tvalues["market"]
    stats["r_squared"] = model.rsquared

    return pd.DataFrame(stats, index=[0])


def rolling_sharpe(ret, days=90, freq="4h", plot=True):
    if "h" in freq:
        bars_per_day = 24 // int(freq.replace("h", ""))
    elif "d" in freq:
        bars_per_day = 1 // int(freq.replace("d", ""))
    else:
        raise ValueError(f"Invalid freq: {freq}")

    window   = days * bars_per_day
    periods  = 365 * bars_per_day

    roll_mean = ret.rolling(window).mean()
    roll_std  = ret.rolling(window).std()

    roll_sharpe = (roll_mean / roll_std) * np.sqrt(periods)

    if plot:
        plt.plot(roll_sharpe)
        plt.title(f"Rolling {days}-day Sharpe Ratio")
        plt.xticks(rotation=45, ha='right')
        plt.show()

    return roll_sharpe


def drawdown(ret):

    cum = (1 + ret).cumprod()
    dd  = cum / cum.cummax() - 1
    return dd


def drawdown_duration(ret):

    cum      = (1 + ret).cumprod()
    is_below = cum < cum.cummax()

    group    = (~is_below).cumsum()
    duration = is_below.groupby(group).cumsum().astype(int)

    return duration


def plot_drawdown(gross_ret, net_ret, title = "Drawdown"):

    dd_gross  = drawdown(gross_ret).astype(float).fillna(0)
    dur_gross = drawdown_duration(gross_ret).astype(float).fillna(0)
    dd_net    = drawdown(net_ret).astype(float).fillna(0)
    dur_net   = drawdown_duration(net_ret).astype(float).fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(dd_gross, color="red", label="Gross")
    axes[0].plot(dd_net, color="steelblue", label="Net")
    axes[0].fill_between(dd_gross.index, dd_gross, 0, alpha=0.3, color="red")
    axes[0].fill_between(dd_net.index,   dd_net,   0, alpha=0.2, color="steelblue")
    axes[0].set_title(title)
    axes[0].set_ylabel("Drawdown")
    axes[0].legend()

    axes[1].plot(dur_gross, color="red", label="Gross")
    axes[1].plot(dur_net, color="steelblue", label="Net")
    axes[1].set_ylabel("Bars below peak")
    axes[1].set_xlabel("Date")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plt.show()