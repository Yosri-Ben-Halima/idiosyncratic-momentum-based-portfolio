# Local Strategy Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Native execution in this session is recommended.

**Goal:** Deliver a polished localhost dashboard backed by the same computation engine as PNG tearsheets, with validated parameter controls, background runs, interactive charts, sensitivity surfaces, and downloadable exports.

**Architecture:** Extract backtest orchestration into a Python service consuming versioned price snapshots and normalized configuration. FastAPI serves a local static frontend and run API; a single background worker owns computation/export jobs. Plotly renders shared numerical diagnostics rather than PNG screenshots.

**Tech Stack:** Python 3.12, existing NumPy/pandas/SciPy/Matplotlib/Plotly, FastAPI/Uvicorn, vanilla JavaScript/CSS/HTML, unittest, and browser automation where available.

**Spec:** [Approved dashboard specification](../specs/2026-10-08-local-dashboard-design.md)

## Global Constraints

- Preserve current signal definitions, universe methodology, portfolio selection, cost treatment, and benchmark series.
- Both the PNG command and web worker call the reusable backtest service.
- Default address: `http://127.0.0.1:8000`; all chart dependencies/assets are local.
- Dates outside cached data coverage require refresh; do not silently extend or truncate the requested evaluation period.
- Editing controls updates pending configuration without submitting computation. Previous completed results remain visible until a new run succeeds.
- Undefined diagnostics serialize to JSON null, never misleading zeros.
- Existing `task run`, `task backtest`, and `task sensitivity` behavior remains available.
- Dashboard sensitivity is opt-in; the existing CLI sensitivity default stays unchanged.
- Downloads use the completed configuration and server-owned run IDs/allowlisted filenames.
- Public deployment, authentication, alternative weighting, and execution-model changes are out of scope.
- Use the existing worktree and preserve all pre-existing uncommitted work. Stage only task-owned files; do not accidentally commit prior sensitivity/backtest work.

## Review Focus

- Editing controls while an earlier request finishes must preserve pending edits and identify which parameters produced the displayed results: browser test in Task 6.
- Refreshing a snapshot must not alter inputs, downloads, or cache identity for existing completed runs: immutable-snapshot test in Task 3.
- All-cash and short-history runs must produce usable charts/exports without singular KDE or fixed twelve-month assumptions: diagnostics/export tests in Task 2.
- An apparently local API must reject foreign-origin mutation and artifact traversal without blocking normal loopback browser requests: HTTP tests in Task 4.
- Cost-only or RF-only changes must not reuse incompatible cached results or signals: identity/derived-rate tests in Tasks 1 and 4.

## File map

| File | Responsibility |
| --- | --- |
| `src/engine/config.py` | Controls schema, normalization, validation, derived rates/costs, grid construction. |
| `src/engine/snapshots.py` | Saved snapshot discovery, loading, immutable refresh/version identity. |
| `src/engine/backtest.py` | `Snapshot`/`BacktestResult` contracts and shared numerical backtest orchestration. |
| `src/engine/diagnostics.py` | Presentation-independent rolling, distribution, calendar, portfolio, recovery, and MC diagnostics. |
| `src/engine/exports.py` | CSV/metadata writing, existing PNG renderer integration, export ZIP. |
| `src/engine/__init__.py` | Small public engine interface. |
| `src/dashboard/jobs.py` | Bounded background job lifecycle and completed-result cache. |
| `src/dashboard/charts.py` | Plotly figures and finite API payload assembly. |
| `src/dashboard/app.py` | FastAPI app factory, local API, static assets, download protection. |
| `src/dashboard/__main__.py` | Loopback CLI startup and optional browser open. |
| `src/dashboard/static/index.html` | Accessible page structure and controls. |
| `src/dashboard/static/style.css` | Responsive dark design and interaction states. |
| `src/dashboard/static/app.js` | Pending/completed state, controls, API calls, polling, charts, tabs, downloads. |
| `main.py` | Existing CLI using shared engine; retain acquisition and human-readable reporting. |
| `src/visualizations/viz_ops.py` | Accept shared MC diagnostics; fix short/cash plotting failures needed for export. |
| `taskfile.yaml`, `requirements.txt`, `README.md`, `.gitignore` | Startup, dependencies, usage, generated-run ignores. |
| `tests/test_engine.py`, `tests/test_dashboard.py`, `tests/test_dashboard_charts.py` | Numerical, API/job, chart/export regressions. |

## Task 1: Validated configuration and shared backtest engine

**Interfaces:** `Snapshot` is a dataclass with `id: str`, `prices: pd.DataFrame`, `tickers: list[str]`, `config: dict`, and `metadata: dict`. `normalize_config(values: dict, defaults: dict, snapshot: Snapshot) -> dict` returns a new engine-compatible dictionary. `run_backtest(snapshot: Snapshot, config: dict, progress: Callable[[str], None] | None = None) -> BacktestResult` returns config, snapshot identity, aligned strategy/benchmark returns, both metrics dictionaries, sliced signal/weights, risk diagnostics, optional sensitivity result, warnings, and eventual shared diagnostics. Neither function mutates caller inputs.

- [ ] Add `tests/test_engine.py` using a deterministic price fixture with three stocks plus `^GSPC`, enough warmup, and an explicit exclusive end date. Assert engine returns equal a direct invocation of the existing signal/portfolio/returns functions; also compare dates, weights, and metrics.
- [ ] Add normalization assertions: `transaction_cost_bps=20` becomes `cost_bps=.002`, `rf_annual=.04` becomes `rf_daily=1.04**(1/252)-1`; a second normalization does not convert twice. Reject unknown fields, booleans as integers, nonfinite values, reversed/out-of-coverage dates, unsupported frequency, and risk minimum exceeding its window. Assert changing the RF rate actually changes the CAPM input and a changed cost changes charged returns.
- [ ] Run `.venv312/Scripts/python.exe -m unittest discover -s tests -p test_engine.py -v`; expect feature-missing assertions before production implementation.
- [ ] Implement contracts/normalization in the engine files. Primary limits: window 2–756, skip 0–252, cutoff -3–6, holdings 1–universe size, history 2–2520, costs 0–500 bps, RF annual -0.05–0.30, risk window 2–756 with minimum 2–window. MC paths 100–20000, horizon 21–1260, product at most 12,600,000, seed 0–2**32-1. Sensitivity grid at most 225 settings; validate positive steps and include current axes values.
- [ ] Extract existing orchestration without changing order: build warmup signal/weights, slice at start and exclusive end, compute strategy returns on sliced prices, align with benchmark returns computed using its prior history, calculate metrics, evaluate baseline risk, and optionally call sensitivity on end-bounded full warmup prices. Verify sensitivity against engine baseline returns. Report inadequate warmup and reject unusable evaluation intervals without rejecting valid cash portfolios.
- [ ] Run new engine tests plus the existing suite. Compare the saved 2026-10-06 full-universe baseline with `daily_returns.csv` and `metrics.csv` at `rtol=1e-10, atol=1e-12`; save a local validation receipt.
- [ ] Review this task's diff; create a focused commit only if its files are independent of prior uncommitted changes. Otherwise retain the change for the final integration commit and explicitly record why.

## Task 2: Shared diagnostics and robust PNG exports

**Consumes:** `BacktestResult` from Task 1. **Produces:** `build_diagnostics(result: BacktestResult) -> dict`; `export_result(result: BacktestResult, directory: Path, include_png: bool = True) -> dict[str, Path]`. Diagnostics include named numeric series, chart arrays, and seed-controlled Monte Carlo bands/terminal returns. Exports contain existing CSV/metadata contracts, three or five tearsheets, and an allowlisted ZIP bundle.

- [ ] Add failing numerical tests for calendar compounding, half-L1 turnover, holdings, entry z-score, rolling metrics matching existing PNG formulas, historical VaR/CVaR, QQ quantiles, and DD/CAGR with undefined zero-CAGR handling.
- [ ] Add failing tests for deterministic MC seed, sensitivity-disabled exports, all-cash returns, and fewer than 21/12-month observations. Assert PNG files open and CSV returns match the result; constant distributions do not attempt singular KDE and incomplete years preserve missing months.
- [ ] Run the engine tests and confirm the new assertions fail before implementation.
- [ ] Implement diagnostics preserving existing formulas. Extract the existing 21-session block bootstrap to local seeded `np.random.RandomState(seed)` so seed 42 preserves legacy sequences; handle short histories with an available positive block length and constant paths honestly. Preserve the legacy fan presentation and label its simulation interpretation. Reuse exact computed MC arrays for both views.
- [ ] Extend `plot_sheet3`/`plot_tearsheets` with optional shared diagnostics, preserving callers without it. Make cash/short-history plotting robust; apply only necessary fixes to the existing renderer. Add optional parameters at the end of signatures.
- [ ] Update `main.py` to call the shared service and export layer while retaining CLI options, downloads, summary reporting, provenance, and PNG default behavior. Avoid invoking a second backtest for export.
- [ ] Run `.venv312/Scripts/python.exe -m unittest discover -s tests -v` and an offline saved-snapshot PNG export; compare numerical output with Task 1's receipt. Inspect exported images.
- [ ] Review and record the task's change set; avoid staging unrelated prior changes.

## Task 3: Snapshot discovery and explicit data refresh

**Consumes:** Task 1 `Snapshot` and normalized config. **Produces:** `discover_snapshot(root: Path, source: Path | None = None) -> Snapshot | None`; `refresh_snapshot(config: dict, root: Path, progress: Callable[[str], None] | None = None) -> Snapshot`.

- [ ] Add failing tests using temporary saved price/metadata files: latest usable snapshot wins, incomplete directories are skipped, explicit source errors are useful, IDs change when data changes, and loading defaults performs no RF or network request.
- [ ] Test refresh at the acquisition boundary with a deterministic downloader: save to a fresh unique directory, calculate adequate warmup for the maximum enabled grid window, record exclusions, preserve old snapshot bytes, and leave the selected snapshot untouched on download failure. Also assert completed result inputs retain their original identity after refresh.
- [ ] Run snapshot tests to observe missing behavior, then implement discovery/load/refresh. Reuse `fetch_prices`, ticker helper, and explicit RF refresh; network calls belong only to Refresh or the existing CLI acquisition path. Use a staging directory and promote only a validated usable snapshot.
- [ ] Persist immutable snapshots under ignored `results/dashboard/snapshots/`; hash source price bytes plus relevant metadata for identity. Treat observed coverage and requested exclusive download cutoff distinctly.
- [ ] Run engine/snapshot tests; document offline startup and explicit refresh semantics.
- [ ] Review and record the task's change set.

## Task 4: Responsive backend jobs and localhost API

**Consumes:** Tasks 1–3. **Produces:** `JobManager` with `submit(kind: str, config: dict, snapshot: Snapshot | None) -> dict`, `status(job_id: str) -> dict`, and `result(job_id: str) -> BacktestResult`; `create_app(source: Path | None = None, results_root: Path = Path('results')) -> FastAPI`. Jobs use a single worker, at most two queued jobs, and at most eight retained completed results. Evict matching artifacts safely; never evict active jobs.

- [ ] Add failing tests for queued→running→succeeded/failed lifecycle, stage messages, duplicate active submission reuse, queue saturation, and cache isolation when costs/RF/date/seed/snapshot change. Use real small-fixture engine computations for successful jobs; inject controlled functions only for lifecycle failure/queue timing tests.
- [ ] Add HTTP tests: `GET /api/bootstrap`, `POST /api/runs`, `GET /api/jobs/{id}`, `GET /api/runs/{id}/results`, `POST /api/refresh`, `POST /api/runs/{id}/exports`, `GET /api/runs/{id}/files/{filename}`. Assert field-specific 422 validation, 404 unknown job, useful no-snapshot response, JSON null instead of NaN, and previous completed data surviving failure.
- [ ] Add HTTP tests rejecting foreign Origin headers and traversal/unknown filenames; test same-origin/loopback browser behavior and requests without Origin. Validate the Host header against localhost/loopback to prevent DNS rebinding.
- [ ] Run failing API/job tests; install compatible pinned FastAPI/Uvicorn and test client dependencies into the Python 3.12 environment after verifying official package documentation. Keep all API data same-origin and local.
- [ ] Implement `jobs.py`/`app.py`, bounded run cache keyed by canonical normalized configuration and snapshot identity, finite serialization, safe artifact routing, and local Plotly JS asset delivery from the installed package. Set Matplotlib Agg for the backend and serialize PNG work with computation jobs.
- [ ] Test real small-grid sensitivity and generated PNG/ZIP downloads through the HTTP app. Run all regression tests and dependency consistency checks.
- [ ] Review and record the task's change set.

## Task 5: Interactive figures with complete tearsheet coverage

**Consumes:** Tasks 1–2. **Produces:** `build_dashboard_payload(result: BacktestResult) -> dict` in `charts.py`, containing completed configuration/provenance, KPI values, metrics, named Plotly figures grouped into performance/risk/portfolio/sensitivity, holdings, and surface mode metadata.

- [ ] Add failing chart tests asserting displayed cumulative returns, percentage units, benchmark values, DD/CAGR, holdings and risk shares against literal known arrays. Test cash/short history and missing covariance without fabricated zeros.
- [ ] Add surface tests asserting x=cutoffs, y=windows, unaltered z grid for total return/holdings/diversification/top-risk-share, and current-setting marker. Period switches apply only to available performance fields. Assert raw data retains a fixture spike.
- [ ] Run chart tests to observe failures, then implement every chart in the spec's Dashboard parity section. Share diagnostic values, do not recalculate strategy or MC. Use consistent dark Plotly styling, clear units, and null-safe empty annotations.
- [ ] Return only figures needed by the completed result; return a sensitivity-disabled state explicitly. Bundle Plotly locally; use `Plotly.react` and lazy rendering on tab activation to avoid unnecessary WebGL surfaces.
- [ ] Run `.venv312/Scripts/python.exe -m unittest discover -s tests -v`; compare full-universe payload metrics with saved baseline results.
- [ ] Review and record the task's change set.

## Task 6: Polished frontend, parameter controls, and startup

**Consumes:** API and payload contracts from Tasks 4–5. **Produces:** local responsive dashboard, `python -m src.dashboard --source PATH --port 8000 [--open]`, and `task dashboard` with `PYTHON`, `SOURCE`, `PORT`, and CLI argument overrides.

- [ ] Define browser acceptance checks before implementation: default controls reflect bootstrap configuration; synchronized range/number inputs submit exact values; invalid forms do not submit; pending edits during computation remain pending after completion; failed runs preserve old results; disabled sensitivity shows its empty state; exports identify the completed configuration.
- [ ] Implement `index.html`/`style.css` with desktop sidebar, mobile control drawer, snapshot status, KPI strip, four tabs, chart cards, holdings/metrics tables, help text, reset, loading/error states, and export controls. Use all selector types from the spec, native form semantics, visible focus, non-color status text, reduced-motion handling, and local fonts/assets.
- [ ] Implement `app.js` with distinct pending/completed/submitted configurations and run IDs. Poll only active jobs, handle reconnect/network errors, stop on completion, render charts atomically, and retain old results during work. Do not let a stale response override a newer submitted run or edits.
- [ ] Add exact tested-values/grid-count previews, separate metric/period dropdowns, filled 3D surfaces, orange current markers, camera persistence, and refresh requiring an explicit click. Reset updates controls without recomputation.
- [ ] Implement loopback startup entry point and Taskfile command; optional browser opening occurs only after server is available. Handle occupied port with a useful startup error.
- [ ] Run the backend and browser-check desktop and narrow widths, mouse and keyboard inputs, real runs, progress, stale edits, reset, refresh failure, surface selection/rotation, and PNG/ZIP downloads. Inspect screenshots and console/network errors; correct failures and repeat affected checks.
- [ ] Review and record the task's change set.

## Task 7: Full validation, usage documentation, and review

- [ ] Add README startup/install examples for Python 3.12, PNG mode versus dashboard, `--source`, default latest usable snapshot, ports, exports, caching, practical limits, and offline/refresh behavior. Ignore generated dashboard runs/snapshots and verification artifacts.
- [ ] Run `.venv312/Scripts/python.exe -m unittest discover -s tests -v`, `git diff --check`, Python compile checks, and dependency consistency checks. Do not broaden testing after these pass without a new issue.
- [ ] Run the saved full-universe baseline through the real web API and compare daily returns/metrics; run a small-grid sensitivity job, select all requested surface heights, and export its PNGs. Verify the CLI uses the same engine and numerical outputs. Save a validation receipt and reviewable screenshots outside committed source.
- [ ] Inspect all changes against the approved spec, record any unmet acceptance checks, and obtain the execution method's required final independent review. If native execution is chosen, use one fresh reviewer at the end; do not delegate implementation tasks.
- [ ] Fix actionable review findings, rerun affected checks, and report the local URL, exact startup command, verified capabilities, and any real remaining limitations. Do not publish externally or merge a PR.

## Self-review and execution handoff

The tasks cover the approved spec: shared engine/parity (1–2), saved/refresh data and provenance (3), background work/cache/local API (4), all chart families and surfaces (5), selector and visual behavior/startup (6), and documentation/end-to-end evidence (7). Review Focus conditions have explicit owning tests. Interfaces use one `Snapshot` and `BacktestResult` contract throughout; derived costs/rates are normalized once. Existing numerical assumptions remain unchanged.

Recommended execution: native in the current session, since the tasks share result/configuration interfaces and introduce one cohesive local application. Implementation waits for the user's plan review and choice of execution method, as required by the writing-plans skill.
