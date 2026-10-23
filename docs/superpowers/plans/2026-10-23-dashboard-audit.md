# Dashboard Audit Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in the existing isolated worktree.

**Goal:** Make current-run audit evidence understandable and explorable inside a fifth dashboard tab.

**Architecture:** A read-only engine module recomputes controlled variants against the parent's immutable Snapshot and completed BacktestResult. Existing JobManager handles audit work, caching and progress. A dedicated static audit renderer uses the existing Plotly/theme and keeps report state tied to the completed run.

**Tech Stack:** Existing Python/pandas/numpy, FastAPI, unittest, vanilla JavaScript/CSS, local Plotly; no new runtime dependencies.

**Spec:** docs/superpowers/specs/2026-10-23-dashboard-audit-design.md (approved by user).

## Global constraints

- Keep original signal, execution and PNG outputs unchanged.
- Audit the completed run, preserving its snapshot after refresh and pending parameter edits.
- Passing timing checks do not validate historical constituents, future returns or parameter choices.
- Short/cash samples must not produce fabricated passes or invalid JSON.

## Review focus

- Runs too short to support an internal cutoff: report insufficient evidence.
- Missing stock data: use paired observations in the isolated estimator comparison and verify against least squares.
- Audit failure, refresh and new-run completion: retain correct provenance and clear only obsolete audit state.
- Negative/zero returns: contribution charts reconcile in initial-capital units and avoid misleading gain-share ratios.
- Narrow screens and no completed run: usable empty states and no overflow.

### Task 1: Engine evidence

- [x] Add tests/test_audit.py for invariance probes, paired OLS, drift/cash cases, attribution and immutability.
- [x] Run tests red, then implement src/engine/audit.py with run_audit(snapshot,result,progress=None), deterministic cutoff/seed, evidence checks, counterfactual metrics, concentration and limits.
- [x] Verify tests green with finite/nullable JSON-compatible results.

### Task 2: Worker/API lifecycle

- [x] Add tests/test_dashboard_audit.py for POST/GET audit, caching, protected parent snapshot, missing/unfinished runs, same-origin protection and payload privacy.
- [x] Retain completed run snapshots privately in src/dashboard/jobs.py. Add audit job dispatch and report retrieval.
- [x] Add POST and GET /api/runs/{identifier}/audit in src/dashboard/app.py. Verify tests green.

### Task 3: Dashboard experience

- [x] Add Audit tab/empty state in static/index.html; add dedicated static/audit.js and static/audit.css.
- [x] Wire state reset, busy state, render routing and JSON download to the existing completed-run lifecycle in static/app.js.
- [x] Render verdict, exact provenance, responsive information timeline, evidence cards, CAGR comparison, wealth contributions and remaining gaps. Escape all dynamic content.
- [x] Browser-check live computation, unchanged draft settings, report reset for a new run, download, short/cash run and mobile overflow.

### Task 4: Finish

- [x] Update README with audit usage, costs and interpretation limits.
- [x] Run task test PYTHON=.venv312/Scripts/python.exe and inspect all results.
- [x] Inspect screenshots; record observed timings, receipts and any limitations. Commit source/docs/tests only.


## Execution record

- Implemented inline in the existing isolated worktree. No new runtime dependencies.
- Engine/API tests were observed failing before implementation, then passing. Final full suite: **30 tests passed** (23.1 seconds). Node syntax checks and git diff whitespace checks pass.
- Ruling: fixed deterministic midpoint cutoff and seed 42 rather than another parameter control — the tab explains the exact cutoff and holds the universe fixed. Timing verdict requires at least 20 earlier sessions, 10 later sessions and earlier active holdings. Cash/short cases remain insufficient.
- Ruling: initial sample filtering and historical membership are explicitly untested; they are not hidden behind a passing temporal-causality verdict.
- Browser verification with real local Chromium: 500-stock, 1,698-session run; original CAGR 49.9017%, drift 49.1974%, matched-observation CAPM 49.7579%, zero fixed RF 49.9017%. Uncached audit took 16.64 seconds on this machine. All three timing/replay checks unchanged; synthetic estimator issue correctly identified.
- Browser checks: empty state, audit computation, pending edits preserved, exact JSON download, current-run provenance, new-run reset, cash/short insufficient state, widths 375/768/1024, readable responsive comparison chart. Unexpected console errors: zero.
- Independent fresh review found two drift-comparison issues: unchanged selections skipped scheduled rebalances, and initial trading cost differed from production. Both reproduced in new regressions, corrected, then independently re-reviewed with no remaining important findings. Production strategy code was not altered.
- Queue regression: a just-completed child job could evict its old parent immediately. Parent is now protected in the completion sweep; API test verifies audit retrieval after refresh and near the retention limit.
- Generated screenshots, downloaded evidence, browser script and test logs remain ignored under results/causality_audit/. Source/docs/tests only are included in the feature commit.
