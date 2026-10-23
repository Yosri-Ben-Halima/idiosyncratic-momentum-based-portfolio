# Audit tab for the completed backtest

The user wants the recent strategy audit inside the existing dashboard, presented intuitively. They selected recomputation for the current completed run instead of a saved-report-only tab. Retain the approved dark navy/teal design and existing FastAPI/vanilla JavaScript/Plotly stack.

## Approaches

- **Selected: audit the completed run.** An explicit Run audit button computes against that run's immutable prices, configuration and results. More computation, but every comparison applies to what the user is viewing.
- Saved audit only: inexpensive, but findings from the original 500-stock run cannot safely describe other settings or refreshed data. Not selected.

## Presentation

1. **Verdict and scope.** Separate signal timing from historical realism. Show checked/issue/not tested labels, run dates, snapshot identity, settings and the future-price cutoff actually used. No composite confidence score or claim that passing tests validates expected returns.
2. **Information timeline.** Display the rolling-estimation history, residual momentum window, skipped sessions and selection point using this run's actual window/skip settings. Explain that the skip is a signal choice and chronology prevents future-price use.
3. **Evidence cards.** Future-price perturbation and deletion checks show whether earlier signals, weights and returns changed. A paired-observation CAPM check explains the known missing-data issue and measures its impact without changing production calculations.
4. **Return comparison.** Baseline, weights drifting between rebalances, paired-observation estimation, and zero constant risk-free rate. Show CAGR and percentage-point changes alongside total return and drawdown; counterfactuals are isolated diagnostics, not a corrected/validated strategy.
5. **Concentration.** Average/minimum holdings, maximum stock weight and an additive wealth-contribution chart identify what drove the result. Include other stocks and costs so attribution reconciles with total return.
6. **Remaining gaps and details.** Historical constituents/delistings, full-history sample filtering, vendor data, execution and independent parameter validation remain explicitly untested. Expandable method/evidence and downloadable JSON support reproducibility.

## Computation and state

Add an engine audit module and audit jobs to the bounded existing worker. Keep the parent run's snapshot for recomputation, including after data refresh. Use a deterministic midpoint cutoff and fixed random seed; report the exact date. Very short runs show insufficient evidence instead of a pass. Reuse completed audit jobs for identical runs. New completed runs clear the displayed audit; unsubmitted sidebar edits do not alter it. Errors preserve the previous completed backtest. Audit is available as a fifth tab with a useful empty state. PNG exports continue to describe the original backtest.

## Verification

Start with tests for real current-run audit computations and API lifecycle. Check future-only changes cannot change prior decisions, paired-observation estimates match least squares with missing data, attribution reconciles, and cash/short runs remain honest. Browser-check audit navigation, computation progress, pending edits, new-run reset, evidence download and narrow layouts. Run the full existing test suite. Do not silently change the strategy while adding its audit view.
