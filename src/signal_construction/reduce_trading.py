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


def apply_dynamic_threshold_buffer(target_weights, threshold=0.002):
    """
    Dynamic threshold: scale threshold by typical daily weight change.
    """
    buffered_weights = target_weights.copy()

    # Compute typical daily change (median abs change)
    typical_change = (target_weights - target_weights.shift()).abs().median(axis=1)

    for t in range(1, len(target_weights)):
        prev = buffered_weights.iloc[t-1]
        target = target_weights.iloc[t]
        change = target - prev

        # dynamic threshold = threshold * typical change
        dyn_thresh = threshold * typical_change.iloc[t]
        new_weights = prev.where(change.abs() < dyn_thresh, target)

        buffered_weights.iloc[t] = new_weights

    return buffered_weights


def apply_partial_adjustment_buffer(target_weights, threshold=0.002):
    """
    Gradually adjust weights only for changes above threshold.
    Fully updates if threshold=0.
    """
    buffered_weights = target_weights.copy()

    for t in range(1, len(target_weights)):
        prev = buffered_weights.iloc[t-1].fillna(0)
        target = target_weights.iloc[t].fillna(0)

        change = target - prev
        excess = np.maximum(change.abs() - threshold, 0)
        new_weights = prev + np.sign(change) * excess

        buffered_weights.iloc[t] = new_weights

    return buffered_weights


def apply_no_trade_buffer(target_weights, threshold=0.002):
    """
    threshold = minimum absolute weight change required to trade
    e.g. 0.002 = 20bps weight change
    """
    buffered_weights = target_weights.copy()
    
    for t in range(1, len(target_weights)):
        prev = buffered_weights.iloc[t-1]
        target = target_weights.iloc[t]
        
        change = target - prev
        
        # Only update where change exceeds threshold
        new_weights = prev.where(change.abs() < threshold, target)
        
        buffered_weights.iloc[t] = new_weights
        
    return buffered_weights


def apply_significant_trade(w, threshold=0.01):
    """
    threshold = minimum absolute weight change required to trade
    """
    delta = w - w.shift()
    return w.shift() + delta.where(abs(delta) > threshold, 0)


def reduce_significant_swings(w, threshold=0.1):
    """
    threshold = maximum weight
    """
    return w.clip(-threshold, threshold)


def trade_non_consecutive_days(w, ts=2):
    """
    timestep = no of timesteps between trades
    e.g. if weights calculated every 4 hours, trades are made every 4*ts hours
    """
    block = ts
    master = w.iloc[0:len(w)//block*block:block]

    for i in range(1, block):
        w.iloc[i:len(master)*block:block] = master.values
    
    return w