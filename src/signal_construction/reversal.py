def reversal_signal(rets):
    # fill out the body here
    # return a DataFrame "signal"
    # signal has same index/columns as rets
    # the value in signal is  1 if the symbol had the worst return on
    # a particular day, -1 if it had the best, and 0 otherwise

    daily_max = rets.max(axis=1)
    daily_min = rets.min(axis=1)

    signal = rets.eq(daily_min, axis=0).astype(int) - rets.eq(daily_max, axis=0).astype(int)
    
    return signal