"""Batched price download returns exactly what the former single download returned, for the same Yahoo data."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data_fetcher import fetch_prices


def yahoo(seed=0, n=123):
    """Fake yfinance: OHLCV for n tickers + ^GSPC, with late listings, gaps and empty tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=300)
    tickers = [f'T{i:03d}' for i in range(n)] + ['^GSPC']
    close = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0, .02, (len(dates), len(tickers))), axis=0)),
                         index=dates, columns=tickers)
    close.iloc[:150, min(7, n-1)] = np.nan                      # listed late
    close.iloc[rng.integers(0, 300, 200), min(11, n-1)] = np.nan  # gaps
    close.iloc[:, min(42, n-2)] = np.nan                        # delisted: no data at all
    close.iloc[10, :min(60, n)] = np.nan                        # a date with no prices for the first batch
    fields = {'Close': close, 'Open': close*1.01, 'High': close*1.02, 'Low': close*.99, 'Volume': close*1e6}

    def download(batch, start, end, **kwargs):
        frame = pd.concat({f: v[sorted(batch)] for f, v in fields.items()}, axis=1, names=['Price', 'Ticker'])
        frame.index.name = 'Date'
        return frame.dropna(how='all')               # Yahoo omits dates with no data for any requested ticker
    return download, tickers[:-1]


def reference_fetch(download, tickers, start, end):
    """The former fetch_prices: one call for every ticker."""
    raw = download(tickers+['^GSPC'], start=start, end=end, auto_adjust=True, progress=True, threads=True)['Close']
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)
    return raw


class FetchParityTests(unittest.TestCase):
    def check(self, seed, n, order=None):
        download, tickers = yahoo(seed, n)
        if order is not None:
            tickers = [tickers[i] for i in order(len(tickers))]
        expected = reference_fetch(download, tickers, '2020-01-01', '2021-03-01')
        with patch('src.data_fetcher.fetcher_service.yf.download', side_effect=download) as calls:
            actual = fetch_prices(tickers, '2020-01-01', '2021-03-01')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertEqual(calls.call_count, -(-(n+1)//50))
        self.assertTrue(all(len(c.args[0]) <= 50 for c in calls.call_args_list))

    def test_identical_to_single_download(self):
        self.check(0, 123)

    def test_identical_with_unsorted_universe(self):
        self.check(1, 123, order=lambda k: np.random.default_rng(5).permutation(k))

    def test_identical_when_last_batch_has_one_ticker(self):
        self.check(2, 99)   # 99 tickers + ^GSPC = 100: two full batches
        self.check(3, 100)  # 101 symbols: last batch is ^GSPC alone

    def test_identical_for_a_small_universe(self):
        self.check(4, 5)
