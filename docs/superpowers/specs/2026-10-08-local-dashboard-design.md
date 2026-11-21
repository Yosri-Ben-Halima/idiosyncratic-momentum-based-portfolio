# Local strategy dashboard and PNG output

Status: proposed written specification for user review. The conversational design was approved on 2026-10-08. Implementation has not started.

## Outcome

Add a polished localhost application where the user changes strategy parameters, explicitly runs the existing computation engine, and explores the same results as the five PNG tearsheets through interactive dashboards. Preserve existing command-line PNG output and provide PNG downloads from dashboard runs.

The user specifically wants appropriate selectors for each parameter, backend computation, dynamic results, and stylish presentation. Previously requested outputs include benchmark comparisons, drawdown relative to CAGR, holdings, and three-dimensional sensitivity surfaces over momentum window and z-score cutoff, including total return and risk diversification/concentration.

## Selected architecture

Use a Python FastAPI service serving a small HTML/CSS/JavaScript frontend and local Plotly JavaScript. This is a single local process with no Node build step or external frontend CDN requirement. The backend serves both the UI and its JSON API from the same loopback origin. Default address: `http://127.0.0.1:8000`.

A custom frontend is preferred over Streamlit for layout, accessible controls, chart interactions, and precise control of pending edits versus completed results. A separate React/Vite application would add build and startup complexity without being necessary for this local tool.

Extract a reusable backtest orchestration service from `main.py`. Both the PNG command and web worker call that service; they continue using the existing feature engineering, portfolio construction, backtesting, and sensitivity modules. Presentation layers consume a result object rather than independently implementing the strategy.

Suggested module boundaries:

- `src/engine/`: immutable run configuration validation, snapshot loading, backtest orchestration, shared diagnostics, and result/export preparation.
- `src/dashboard/`: app factory, local server entry point, background job management, API serialization, and static frontend assets.
- Existing `src/visualizations/` and `src/sensitivity/`: PNG exports and sensitivity calculation/rendering, adapted only where necessary to accept shared results.

These names are implementation guidance, not a requirement to create empty abstractions.

## Data and reproducibility

On startup, select the newest usable saved snapshot under `results/` that contains prices and run metadata. Support an explicit `--source` argument for another snapshot. Show the selected snapshot, price coverage, evaluation dates, universe size, and risk-free rate in the UI. The existing 2026-10-06 snapshot should work without downloading data.

Offer an explicit Refresh data action. Refresh runs in the background using the existing S&P 500 universe and adjusted-price downloader, calculates sufficient warmup for the requested strategy and sensitivity windows, and saves a new versioned snapshot. It must not overwrite the snapshot used by a completed run. Failed refreshes leave the prior snapshot and results usable. When no saved snapshot exists, show a useful empty state and the refresh action.

Dates outside cached data coverage require refresh; never silently pretend an extended backtest used unavailable prices. Show warmup limitations when a requested window has insufficient preceding data, and reject a request that cannot produce a usable evaluation period. Report excluded stocks and reasons. Retain the saved risk-free rate until the user explicitly refreshes or changes it.

Record normalized configuration, snapshot identifier/hash, observed evaluation dates, universe, and Monte Carlo seed with every completed run. Cache keys include all computation inputs and the snapshot version. Reusing completed results must never display another configuration's output as current.

## Parameter controls

Use labeled native controls with keyboard support, visible units, inline validation, and concise help text. Derive defaults from the selected snapshot/configuration. Controls use the following forms:

| Parameter | Selector | Behavior |
| --- | --- | --- |
| Backtest start/end | Two date pickers | End is explicitly labeled exclusive; constrained by available data unless refresh is requested. |
| Momentum window | Integer slider plus synchronized numeric input | Trading sessions; controls both CAPM estimation and residual summation, as today. |
| Momentum skip | Integer input with stepper | Trading sessions; nonnegative. |
| Z-score cutoff | Decimal slider plus synchronized numeric input | Continuous threshold with useful increments, including exact typed values. |
| Maximum holdings (`top_k`) | Integer slider plus numeric input | Clearly a cap; actual holdings can be fewer. Bound by available universe. |
| Rebalance frequency | Dropdown | Human labels for supported weekly, month-end, and quarter-end schedules; map to validated pandas frequencies. |
| Minimum history | Integer input with stepper | Required observed sessions; positive. |
| Trading costs | Decimal numeric input | Basis points; convert once to the engine's existing cost fraction. |
| Annual risk-free rate | Percentage numeric input | Recompute derived daily rate consistently; show source/default. |
| Include sensitivity | Switch/checkbox | Expensive computation is opt-in for dashboard runs; independent of existing CLI default. |
| Sensitivity window axis | Integer min/max/step controls | Preview actual tested values and total grid size; include current window. |
| Sensitivity cutoff axis | Decimal min/max/step controls | Preview actual tested values; include current cutoff. |
| Diagnostic split date | Date picker | Must lie within the evaluation interval when sensitivity is enabled. |
| Risk covariance window/minimum observations | Integer inputs | Minimum observations cannot exceed risk window. |
| Monte Carlo paths | Preset dropdown plus custom integer input | Advanced panel; cap work and memory with a clear validation message. |
| Monte Carlo horizon | Preset dropdown plus custom integer input | Trading sessions; advanced panel. |
| Monte Carlo seed | Integer input | Advanced panel; reproducible runs. |

Window/cutoff slider bounds, grid size, and Monte Carlo workload receive documented practical limits. Numeric inputs must use the same validated limits as the server; no browser-only validation. Weighting remains the existing equal-weight rule because other weighting strategies do not exist in this project.

## Run flow and state

The sidebar has grouped primary strategy settings and collapsible advanced settings, Reset to snapshot defaults, and a prominent Run backtest button. Editing controls updates a pending configuration and a Changes not run indicator. It does not submit work on every slider movement.

A background worker executes a run so the HTTP server stays responsive. Use one computation worker initially to avoid simultaneous expensive grids and Matplotlib thread-safety problems. Return a run identifier immediately, then expose queued/running/succeeded/failed state, computation stage, elapsed time, and actionable error messages. The frontend polls run status with a bounded interval and stops when the job finishes or fails. Prevent duplicate submissions for an already running identical configuration. Bound queued jobs and cached results.

Keep the last completed dashboard visible during computation. Clearly distinguish its completed configuration from pending settings. Replace its results atomically only after the new run succeeds. A failed run shows its error without erasing previous results. Switching dashboard tabs or rotating a surface never recomputes the backtest.

PNG export uses the completed result/configuration, never current unsaved sidebar edits. Generate expensive exports in the worker and provide individual PNG downloads plus a bundle containing tearsheets and numerical CSV/metadata outputs. Sensitivity pages are present only for runs that computed sensitivity.

## Dashboard parity

Provide four dashboard tabs with Plotly charts, zoom/hover controls, responsive sizing, useful empty states, and a clear strategy/benchmark legend. The dashboard renders numeric results as interactive charts rather than displaying PNG screenshots.

Performance:

- Cumulative returns and drawdowns for strategy and benchmark.
- Full metrics comparison, including total return, CAGR, volatility, Sharpe, Sortino, Calmar, alpha, beta, information ratio, and win rate.
- Rolling Sharpe, beta, and annualized alpha using the same definitions/windows as the PNG presentation.
- Drawdown/CAGR expressed in years/months, with undefined values represented honestly. Label it a recovery heuristic, not observed recovery time.

Risk:

- Daily return distributions and strategy normal QQ plot.
- Historical VaR/CVaR at the existing 90/95/99% levels.
- Monthly return heatmaps for strategy and benchmark.
- Underwater charts for both series.

Portfolio:

- Annual strategy/benchmark return comparison, monthly turnover, active holdings over time, and the existing signal-exposure diagnostic.
- Latest holdings table with capital weights and available risk contributions.
- Monte Carlo fan chart and terminal return distribution using the same method as PNG output. Share simulation diagnostics or deterministic seeds so equivalent runs have consistent outputs. Monte Carlo is presented as a simulation diagnostic.

Sensitivity:

- Rotatable filled surfaces with x = cutoff, y = momentum window, and selectable height: Sharpe, Calmar, total return, average holdings, diversification ratio, top-stock risk share, effective risk contributors, realized volatility, and maximum drawdown loss.
- Full/early/recent period selection for metrics already computed for those periods; do not imply period-specific risk diagnostics exist where they do not.
- Mark the submitted run's setting on each surface and use clear percentage/ratio units.
- Retain measured grid geometry and avoid smoothing away spikes. Include heatmaps/local stability diagnostics and the relevant risk concentration detail.
- If sensitivity was disabled, show an explanation and a way to enable it for the next run.

## Visual and interaction design

Use dark navy page/panel backgrounds, restrained teal for strategy/positive emphasis, blue for benchmark, amber for current settings and pending changes, and red for losses/errors. Favor readable typography, generous spacing, consistent chart styling, and aligned metric cards. Avoid decorative complexity that competes with the results.

Desktop layout: fixed-width parameter sidebar, slim header with snapshot/run status, KPI strip, then chart tabs and cards. On narrow screens, collapse controls into an accessible drawer and stack charts. Honor reduced-motion preferences, visible keyboard focus, adequate contrast, and readable chart labels. Include skeleton/loading states, offline/retry feedback, and no inaccessible color-only status cues. All chart dependencies/assets are local.

## API, local scope, and errors

Expose endpoints for defaults/snapshot metadata, run submission/status/results, data refresh, and downloads. Return finite JSON values or `null` for undefined diagnostics. Validate all configuration at the server boundary and reject unsupported fields/invalid ranges with field-specific errors.

Bind to loopback by default. Reject untrusted cross-origin mutation requests and arbitrary filesystem paths. Downloads resolve server-owned run identifiers and allowlisted artifact filenames. Do not add authentication, public deployment, user accounts, or a database to this local application.

Handle all-cash portfolios, short histories, missing prices, invalid dates, undefined risk metrics, failed downloads, and grid/Monte Carlo resource limits. Never show NaN as a misleading zero or a stale result as newly computed.

## Compatibility and task commands

Existing `task run`, `task backtest`, and `task sensitivity` behavior remains available. Add `task dashboard` with Python interpreter, source snapshot, port, and optional browser-open settings. Document one clear startup command and retain a direct `python -m ...` entry point. PNG-only users do not need to open a browser.

Pin any new Python dependencies compatibly with the project's Python 3.12 environment. Serve Plotly JavaScript from the installed Plotly package or a versioned local asset. Do not require an internet connection to render a previously computed dashboard.

## Verification and acceptance

1. Existing regression suite passes; identical saved inputs produce the existing strategy daily returns and main metrics within numerical tolerance after extraction of the shared engine.
2. Tests cover configuration normalization, derived costs/rates, invalid dates and ranges, snapshot boundaries, cash/short-history diagnostics, job lifecycle/failures, cache isolation, and finite JSON serialization.
3. API integration tests submit real small-fixture computations and retrieve actual results and export artifacts; they do not replace the computation engine with a mock.
4. Browser verification changes parameters through their real controls, submits a run, observes progress, confirms chart updates and completed configuration, changes tabs, rotates/selects surface metrics, resets defaults, and downloads PNG output.
5. Inspect desktop and narrow layouts, keyboard navigation, labels, loading/empty/error states, and browser console/network failures. Capture reviewable screenshots.
6. Run at least one saved full-universe baseline through the web backend and compare with the existing snapshot. Exercise sensitivity with a small grid before a larger run.
7. Document any unavailable browser tooling or checks that could not be completed; do not claim unperformed validation.

## Scope boundaries and preserved assumptions

This feature changes execution access and presentation. Preserve current signal definitions, universe methodology, portfolio selection, cost treatment, and benchmark series. Surface existing methodological limitations in concise run details: retrospective current constituents, fixed equal target weights, incomplete trading-cost treatment, and a price-only benchmark. Do not introduce parameter optimization or recommendations.

Public hosting, broker integration, multiple users, alternative weighting engines, historical constituent reconstruction, and execution-model changes are outside this feature.
