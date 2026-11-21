# Execution ledger — plan: 2026-10-08-local-dashboard.md

Visual direction A approved. Native execution in current existing worktree. Base d984186; prior work remains uncommitted and must be preserved.

Pre-flight: Tasks 1→2→4→5 share BacktestResult and configuration; compatible. Task 3 supplies Snapshot to 1/4. Task 6 consumes finite JSON from 4/5.

Ruling: proceed natively after user selected A — latest instruction authorizes implementing selected mock; no additional approval loop.
Ruling: use a normal checked-in ledger instead of Unix skill helper scripts on Windows; preserve change history and avoid creating another worktree over existing work.
Ruling: do not create integration commits containing the user's pre-existing uncommitted main/visualization/sensitivity work; leave reviewable changes in this worktree.

Task 1: complete. Initial RED then GREEN; shared pipeline, validation, seeded Monte Carlo implemented. Full-universe parity verified through both web API and CLI.
Tasks 3–5: API/snapshot/chart tests written and observed RED before implementation.

Task 2: complete. PNG exports consume shared simulations. Cash/short-history PNG test RED (missing simulation interface) then GREEN. Legacy monthly columns and singular constant KDE fixed.
Task 3: complete. Immutable saved snapshot loading and explicit versioned refresh implemented; snapshot identity test GREEN.
Task 4: complete. Bounded single worker, local API, origin/Host/download guards, all-input cache. Real fixture HTTP run passes. Independent review identified completed-refresh reuse; regression observed RED and fixed.
Task 5: complete. Interactive performance/risk/portfolio diagnostics and exact measured surfaces; chart numeric/percentage tests GREEN. Rolling Sharpe matched to existing PNG raw-return convention after dedicated regression.
Task 6: complete. Chromium browser verified full-universe baseline, pending edits through computation, surface metrics/periods/rotation, five-page ZIP download, mobile layout/drawer, and zero console errors. Mobile drawer close-control overlap reproduced and fixed; responsive Plotly sizing added.
Task 7: complete. Baseline HTTP and CLI run/export reproduced all metrics and 1,698 daily returns at rtol 1e-10, atol 1e-12. Final `task test PYTHON=.venv312/Scripts/python.exe`: 21 passing tests. JavaScript syntax, Python compilation, git diff whitespace, dependency consistency (55 packages), and Taskfile startup/help checks passed. Actual live Yahoo refresh was not triggered; saved snapshot loading and explicit-refresh lifecycle were verified with controlled acquisition boundaries.

Ruling: preserve the PNG rolling-Sharpe convention (raw return mean/std) while headline Sharpe remains excess-return based; chart parity takes precedence over silently changing an existing diagnostic.
Ruling: downloaded files persist for reproducibility, while in-memory jobs/results are bounded; document disk retention explicitly instead of deleting historical export artifacts automatically.
Ruling: server coverage is limited by observed last price + next business day; recorded requested download end cannot expand it. Conservative exchange-holiday behavior documented.
Final independent review: two P2 findings (refresh cache and metadata coverage); both reproduced by RED tests and fixed. No further important findings.

Verification artifacts: ignored `results/dashboard_verification/` contains API/browser receipts, screenshots, and downloaded five-sheet bundle; `results/dashboard-verification-cli.log` records CLI parity. Local server remains at http://127.0.0.1:8000, using saved prices through 2026-10-05. Existing CLI PNG workflow remains available.
Ruling: retain all changes in the user's externally managed worktree; pushing, merging, and deleting the worktree are outside the request.
