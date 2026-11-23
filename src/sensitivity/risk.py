"""Trailing covariance diagnostics, separate from portfolio construction."""
import numpy as np
import pandas as pd

RISK_FIELDS = ("risk_diversification_ratio", "risk_top_share", "risk_effective_contributors",
               "risk_estimated_vol", "risk_hedge_share")


def risk_metrics(weights, covariance):
    """Diversification ratio and concentration of absolute Euler risk contributions.

    DR = sum(w_i * sigma_i) / sigma_portfolio. Contributions are proportional to
    w_i * (covariance @ w)_i. Concentration uses their absolute magnitudes,
    normalized to sum to one; negative hedge contributions remain identifiable.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    missing = dict.fromkeys(RISK_FIELDS, np.nan)
    if cov.shape != (len(w), len(w)):
        raise ValueError("Covariance dimensions must match weights")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Risk diagnostics require finite long-only weights")
    if not w.any() or not np.isfinite(cov).all():
        return missing
    marginal = cov @ w
    variance = float(w @ marginal)
    if variance <= 1e-18 or (np.diag(cov) < 0).any():
        return missing
    contributions = w * marginal
    shares = np.abs(contributions) / np.abs(contributions).sum()
    return {"risk_diversification_ratio": float(w @ np.sqrt(np.diag(cov)) / np.sqrt(variance)),
            "risk_top_share": float(shares.max()),
            "risk_effective_contributors": float(1 / (shares @ shares)),
            "risk_estimated_vol": float(np.sqrt(variance*252)),
            "risk_hedge_share": float(shares[contributions < 0].sum())}


class RiskDiagnostics:
    """Refresh covariance monthly and on target-weight changes; cache across grids.

    At a snapshot date, use only preceding returns. Sample covariances require
    complete selected-stock observations; insufficient histories remain missing.
    No risk-free quote, market data download or change of strategy weights occurs.
    """
    def __init__(self, prices, window=252, min_observations=126):
        if int(window) != window or int(min_observations) != min_observations or not 2 <= min_observations <= window:
            raise ValueError("Risk window/minimum must be integers with 2 <= minimum <= window")
        self.window = int(window)
        self.min_observations = int(min_observations)
        self.returns = prices.ffill().pct_change(fill_method=None)
        self.cache = {}

    def covariance(self, date):
        if date not in self.cache:
            end = self.returns.index.searchsorted(date, side="left")
            history = self.returns.iloc[max(0, end-self.window):end].dropna(how="all")
            if len(history) < self.min_observations:
                self.cache[date] = pd.DataFrame()
            else:
                complete = history.columns[np.isfinite(history.to_numpy()).all(axis=0)]
                if len(complete):
                    cov = np.atleast_2d(np.cov(history[complete].to_numpy(), rowvar=False, ddof=1))
                    self.cache[date] = pd.DataFrame(cov, index=complete, columns=complete)
                else:
                    self.cache[date] = pd.DataFrame()
        return self.cache[date]

    def evaluate(self, weights):
        if weights.empty:
            return pd.DataFrame(index=weights.index, columns=RISK_FIELDS, dtype=float)
        months = weights.index.to_period("M")
        monthly = np.r_[True, months[1:] != months[:-1]]
        changes = weights.diff().abs().sum(axis=1).gt(1e-12).to_numpy()
        values = weights.to_numpy(dtype=float)
        snapshots = {}
        for row in np.flatnonzero(monthly | changes):
            date = weights.index[row]
            held = np.flatnonzero(values[row] > 0)
            metrics = dict.fromkeys(RISK_FIELDS, np.nan)
            if len(held):
                covariance = self.covariance(date)
                position = covariance.columns.get_indexer(weights.columns[held])
                if (position >= 0).all():
                    metrics = risk_metrics(values[row, held], covariance.to_numpy()[np.ix_(position, position)])
            snapshots[date] = metrics
        # Method-based reindex carries snapshots, preserving explicit NaN cash rows.
        return pd.DataFrame.from_dict(snapshots, orient="index").reindex(weights.index, method="ffill")

    def latest_contributions(self, weights):
        result = pd.DataFrame(columns=["weight", "signed_variance_share", "absolute_risk_share"])
        if weights.empty:
            return result
        months = weights.index.to_period("M")
        refresh = np.r_[True, months[1:] != months[:-1]] | weights.diff().abs().sum(axis=1).gt(1e-12).to_numpy()
        snapshot = weights.index[refresh][-1]
        selected = weights.iloc[-1][weights.iloc[-1] > 0]
        covariance = self.covariance(snapshot)
        result.attrs["snapshot_date"] = str(snapshot.date())
        if selected.empty or not selected.index.isin(covariance.columns).all():
            return result
        values = selected.to_numpy() * (covariance.loc[selected.index, selected.index].to_numpy() @ selected.to_numpy())
        if values.sum() <= 1e-18:
            return result
        result = pd.DataFrame({"weight": selected, "signed_variance_share": values/values.sum(),
                               "absolute_risk_share": np.abs(values)/np.abs(values).sum()})
        result = result.sort_values("absolute_risk_share", ascending=False)
        result.attrs["snapshot_date"] = str(snapshot.date())
        return result

    def summarize(self, weights):
        daily = self.evaluate(weights)
        invested = weights.sum(axis=1).gt(0)
        valid = invested & daily.risk_diversification_ratio.notna()
        metrics = daily.loc[valid].mean().to_dict()
        metrics["risk_coverage"] = float(valid.sum()/invested.sum()) if invested.any() else 0.0
        return metrics
