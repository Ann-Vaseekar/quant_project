# import pandas as pd
# import statsmodels.api as sm
# from statsmodels.regression.rolling import RollingOLS

# def compute_resid(ret, window_size=90, mkt_ticker="BTCUSDT"):

#     min_periods = int(window_size * 0.8)
#     assets = [c for c in ret.columns if c != mkt_ticker]

#     resid = pd.DataFrame(index=ret.index, columns=assets, dtype=float)
#     betas = pd.DataFrame(index=ret.index, columns=assets, dtype=float)
#     beta_pvals = pd.DataFrame(index=ret.index, columns=assets, dtype=float)
#     alphas = pd.DataFrame(index=ret.index, columns=assets, dtype=float)
#     alpha_pvals = pd.DataFrame(index=ret.index, columns=assets, dtype=float)

#     for asset in assets:

#         y = ret[asset]
#         X = ret[[mkt_ticker]].copy()
#         X = sm.add_constant(X)

#         model = RollingOLS(
#             y,
#             X,
#             window=window_size,
#             min_nobs=min_periods
#         )

#         results = model.fit(cov_type="HAC", cov_kwds={"maxlags":5})

#         pvalues = pd.DataFrame(
#             results.pvalues,
#             index=results.params.index,
#             columns=results.params.columns
#         )

#         # Parameters
#         alphas[asset] = results.params["const"]
#         betas[asset] = results.params[mkt_ticker]

#         # P-values
#         alpha_pvals[asset] = pvalues["const"]
#         beta_pvals[asset] = pvalues[mkt_ticker]

#         # Residuals: actual - fitted
#         fitted = (X * results.params).sum(axis=1)
#         resid[asset] = y - fitted

#     return resid, beta_pvals, alpha_pvals
