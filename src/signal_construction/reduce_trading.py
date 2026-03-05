import pandas as pd


def partial_adjustment_weights(w_star: pd.DataFrame, rho: float):
    """
    Apply partial adjustment to target weights and returns smooth weights.
    rho: Persistence parameter (0 = full rebalance, 0.9 = slow trading)
    """
    w = w_star.copy().fillna(0)
    w_adj = pd.DataFrame(index=w.index, columns=w.columns)
    
    # Initialize first row
    w_adj.iloc[0] = w.iloc[0]
    
    for t in range(1, len(w)):
        w_adj.iloc[t] = (
            rho * w_adj.iloc[t-1] +
            (1 - rho) * w.iloc[t]
        )
    
    return w_adj
