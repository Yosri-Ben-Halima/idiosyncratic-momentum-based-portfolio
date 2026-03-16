import pandas as pd


def compute_weights(
    selected: list[str],
    tickers: list[str],
    index: pd.Index,
    period_mask: pd.Series,
) -> pd.DataFrame:
    """
    Given a list of selected tickers and a boolean mask for the holding period,
    return a weights DataFrame slice with equal weights for selected tickers.
    Returns zeros (cash) if selected is empty.
    """
    weights_slice = pd.DataFrame(0.0, index=index[period_mask], columns=tickers)
    if selected:
        weights_slice[selected] = 1.0 / len(selected)
    return weights_slice
