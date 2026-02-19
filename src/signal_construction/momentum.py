import numpy as np

def compute_momentum(ret):
    # fill out the body here
    # return a DataFrame "momentum" containing a simple momentum indicator 

    avg = ret.rolling(252).mean() * 252
    vol = ret.rolling(252).std() * np.sqrt(252)

    momentum = avg / vol

    return momentum


def compute_portfolio(momentum):
    # fill out the body here
    # return a DataFrame "portfolio" containing portfolio weights

    weights = (momentum > 1).astype(int)

    return weights.div(weights.abs().sum(1),0)