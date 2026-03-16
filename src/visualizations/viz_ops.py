import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats


PALETTE = {
    "strategy": "#2ecc71",
    "benchmark": "#3498db",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "accent": "#f39c12",
    "dark_bg": "#0d1117",
    "panel_bg": "#161b22",
    "grid": "#21262d",
    "text": "#c9d1d9",
}


def set_dark_style():
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["dark_bg"],
            "axes.facecolor": PALETTE["panel_bg"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "text.color": PALETTE["text"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "legend.facecolor": PALETTE["panel_bg"],
            "legend.edgecolor": PALETTE["grid"],
            "font.family": "monospace",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def fmt_x(x):
    return f"{x:.2f}"


def plot_tearsheet(
    port_ret: pd.Series,
    bench_ret: pd.Series,
    metrics_port: dict,
    metrics_bench: dict,
    signal_z: pd.DataFrame,
    weights: pd.DataFrame,
    top_k: int,
    rf_daily: float,
    window: int = 252,
    n_sims: int = 1000,
    horizon_days: int = 504,
    save_path: str = "tearsheet.png",
):
    """
    Single-figure tearsheet composing all analysis panels.
    Layout (5 rows × 4 cols):
      Row 0 [full width]   — Cumulative returns + drawdown sub-panel
      Row 1 [full width]   — Metrics table
      Row 2 [4 cols]       — Sharpe bar | Sortino bar | Calmar bar | Rolling beta
      Row 3 [full width]   — Rolling Sharpe / Beta / Alpha (3 sub-axes stacked)
      Row 4 [2+2 cols]     — Monthly heatmap × 2
      Row 5 [3 cols]       — Return dist | QQ | VaR
      Row 6 [2 cols]       — Turnover | Holdings
      Row 7 [2 cols]       — MC fan (left+centre) | Terminal dist (right)
    """
    set_dark_style()

    fig = plt.figure(figsize=(28, 72), facecolor=PALETTE["dark_bg"])
    # Use nested GridSpecs for cleaner control
    outer = gridspec.GridSpec(
        9,
        1,
        figure=fig,
        hspace=0.55,
        top=0.97,
        bottom=0.01,
        left=0.05,
        right=0.97,
    )

    # ── Helper: shared spine / grid cleanup ───────────────────────────────────
    def _polish(ax):
        ax.set_facecolor(PALETTE["panel_bg"])
        ax.grid(True, alpha=0.35)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 0: Cumulative returns (3 height) + drawdown (1 height)
    # ══════════════════════════════════════════════════════════════════════════
    gs0 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=[3, 1], hspace=0.08
    )
    ax_cum = fig.add_subplot(gs0[0])
    ax_dd_main = fig.add_subplot(gs0[1], sharex=ax_cum)

    cum_port = (1 + port_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()

    ax_cum.plot(
        cum_port.index,
        cum_port.values,
        color=PALETTE["strategy"],
        lw=2,
        label="Strategy",
    )
    ax_cum.plot(
        cum_bench.index,
        cum_bench.values,
        color=PALETTE["benchmark"],
        lw=1.5,
        label="SP500",
        alpha=0.8,
    )
    ax_cum.fill_between(
        cum_port.index,
        cum_port,
        cum_bench,
        where=cum_port >= cum_bench,
        alpha=0.15,
        color=PALETTE["strategy"],
    )
    ax_cum.fill_between(
        cum_port.index,
        cum_port,
        cum_bench,
        where=cum_port < cum_bench,
        alpha=0.15,
        color=PALETTE["negative"],
    )
    ax_cum.axhline(1, color=PALETTE["neutral"], lw=0.8, linestyle="--")
    ax_cum.text(
        cum_port.index[-1],
        cum_port.iloc[-1],
        f" {cum_port.iloc[-1]:.2f}x",
        color=PALETTE["strategy"],
        fontsize=10,
        va="bottom",
    )
    ax_cum.text(
        cum_bench.index[-1],
        cum_bench.iloc[-1],
        f" {cum_bench.iloc[-1]:.2f}x",
        color=PALETTE["benchmark"],
        fontsize=10,
        va="bottom",
    )
    ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax_cum.set_title(
        "Cumulative Performance — Strategy vs SP500",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_cum.legend(fontsize=10)
    _polish(ax_cum)

    dd_port = cum_port / cum_port.cummax() - 1
    dd_bench = cum_bench / cum_bench.cummax() - 1
    ax_dd_main.fill_between(
        dd_port.index,
        dd_port,
        0,
        color=PALETTE["strategy"],
        alpha=0.6,
        label="Strategy DD",
    )
    ax_dd_main.fill_between(
        dd_bench.index,
        dd_bench,
        0,
        color=PALETTE["benchmark"],
        alpha=0.4,
        label="Benchmark DD",
    )
    ax_dd_main.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
    )
    ax_dd_main.legend(fontsize=9)
    _polish(ax_dd_main)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 1: Metrics table + 3 bar charts
    # ══════════════════════════════════════════════════════════════════════════
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.35)
    ax_tbl = fig.add_subplot(gs1[0, :3])
    ax_tbl.axis("off")
    ax_tbl.set_facecolor(PALETTE["panel_bg"])

    keys = [
        "CAGR",
        "Vol",
        "Sharpe",
        "Sortino",
        "Max DD",
        "Calmar",
        "Alpha",
        "Beta",
        "IR",
        "Win Rate",
    ]
    fmts = {
        "CAGR": fmt_pct,
        "Vol": fmt_pct,
        "Sharpe": fmt_x,
        "Sortino": fmt_x,
        "Max DD": fmt_pct,
        "Calmar": fmt_x,
        "Alpha": fmt_pct,
        "Beta": lambda x: f"{x:.3f}",
        "IR": fmt_x,
        "Win Rate": fmt_pct,
    }
    table_data = []
    for k in keys:
        s_val = (
            fmts[k](metrics_port[k]) if pd.notna(metrics_port.get(k, np.nan)) else "N/A"
        )
        b_val = (
            fmts[k](metrics_bench[k])
            if pd.notna(metrics_bench.get(k, np.nan))
            else "N/A"
        )
        table_data.append([k, s_val, b_val])

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=["Metric", "Strategy", "SP500"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(PALETTE["panel_bg"] if r > 0 else "#1f2937")
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_text_props(
            color=PALETTE["strategy"] if (r > 0 and c == 1) else PALETTE["text"],
            fontweight="bold" if (r > 0 and c == 1) else "normal",
        )
    ax_tbl.set_title("Performance Metrics", fontsize=11, fontweight="bold", pad=6)

    for col_idx, metric in enumerate(["Sharpe", "Sortino", "Calmar"]):
        fig.add_subplot(gs1[0, col_idx]) if col_idx < 3 else None

    # Redraw bars properly alongside the table (table uses cols 0–2, bars go in col 3 split manually)
    gs1b = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs1[0, 3], hspace=0.6)
    for i, metric in enumerate(["Sharpe", "Sortino", "Calmar"]):
        ax_b = fig.add_subplot(gs1b[i])
        vals = [metrics_port.get(metric, np.nan), metrics_bench.get(metric, np.nan)]
        bars = ax_b.bar(
            ["Strat", "SP500"],
            vals,
            color=[PALETTE["strategy"], PALETTE["benchmark"]],
            width=0.5,
            edgecolor=PALETTE["grid"],
        )
        for bar, val in zip(bars, vals):
            if pd.notna(val):
                ax_b.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=PALETTE["text"],
                )
        ax_b.set_title(metric, fontsize=9, fontweight="bold")
        ax_b.axhline(0, color=PALETTE["neutral"], lw=0.7)
        _polish(ax_b)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 2: Rolling metrics (Sharpe / Beta / Alpha stacked)
    # ══════════════════════════════════════════════════════════════════════════
    gs2 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[2], hspace=0.15)

    ax_rs = fig.add_subplot(gs2[0])
    roll_sharpe_port = (
        port_ret.rolling(window).mean() / port_ret.rolling(window).std()
    ) * np.sqrt(252)
    roll_sharpe_bench = (
        bench_ret.rolling(window).mean() / bench_ret.rolling(window).std()
    ) * np.sqrt(252)
    ax_rs.plot(
        roll_sharpe_port.index,
        roll_sharpe_port,
        color=PALETTE["strategy"],
        lw=1.5,
        label="Strategy",
    )
    ax_rs.plot(
        roll_sharpe_bench.index,
        roll_sharpe_bench,
        color=PALETTE["benchmark"],
        lw=1.2,
        alpha=0.8,
        label="SP500",
    )
    ax_rs.axhline(0, color=PALETTE["neutral"], lw=0.8, linestyle="--")
    ax_rs.set_title(f"Rolling {window}d Sharpe", fontsize=10, fontweight="bold")
    ax_rs.legend(fontsize=8)
    _polish(ax_rs)

    ax_rb = fig.add_subplot(gs2[1], sharex=ax_rs)
    roll_cov = port_ret.rolling(window).cov(bench_ret)
    roll_var = bench_ret.rolling(window).var()
    roll_beta = roll_cov / roll_var
    ax_rb.plot(roll_beta.index, roll_beta, color=PALETTE["accent"], lw=1.5)
    ax_rb.axhline(1, color=PALETTE["neutral"], lw=0.8, linestyle="--", label="β=1")
    ax_rb.set_title(f"Rolling {window}d Beta", fontsize=10, fontweight="bold")
    ax_rb.legend(fontsize=8)
    _polish(ax_rb)

    ax_ra = fig.add_subplot(gs2[2], sharex=ax_rs)
    roll_alpha = (
        port_ret.rolling(window).mean()
        - (rf_daily + roll_beta * (bench_ret.rolling(window).mean() - rf_daily))
    ) * 252
    ax_ra.plot(roll_alpha.index, roll_alpha * 100, color=PALETTE["strategy"], lw=1.5)
    ax_ra.fill_between(
        roll_alpha.index,
        roll_alpha * 100,
        0,
        where=roll_alpha >= 0,
        alpha=0.2,
        color=PALETTE["strategy"],
    )
    ax_ra.fill_between(
        roll_alpha.index,
        roll_alpha * 100,
        0,
        where=roll_alpha < 0,
        alpha=0.2,
        color=PALETTE["negative"],
    )
    ax_ra.axhline(0, color=PALETTE["neutral"], lw=0.8, linestyle="--")
    ax_ra.set_title(f"Rolling {window}d Alpha (ann. %)", fontsize=10, fontweight="bold")
    ax_ra.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    _polish(ax_ra)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 3: Monthly heatmaps (side by side)
    # ══════════════════════════════════════════════════════════════════════════
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], wspace=0.3)

    def _monthly_matrix(rets):
        monthly = rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        df = monthly.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        return pivot

    cmap_rg = LinearSegmentedColormap.from_list(
        "rg", [PALETTE["negative"], "#1a1a2e", PALETTE["strategy"]]
    )
    for col_i, (rets, title) in enumerate(
        [
            (port_ret, "Strategy Monthly Returns"),
            (bench_ret, "SP500 Monthly Returns"),
        ]
    ):
        ax_hm = fig.add_subplot(gs3[col_i])
        sns.heatmap(
            _monthly_matrix(rets) * 100,
            ax=ax_hm,
            cmap=cmap_rg,
            center=0,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 7},
            linewidths=0.4,
            linecolor=PALETTE["grid"],
            cbar_kws={"label": "Return (%)"},
        )
        ax_hm.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax_hm.set_xlabel("")
        ax_hm.set_facecolor(PALETTE["panel_bg"])
        plt.setp(ax_hm.get_xticklabels(), rotation=0, fontsize=8)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 4: Return distribution | QQ | VaR
    # ══════════════════════════════════════════════════════════════════════════
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[4], wspace=0.35)

    ax_hist = fig.add_subplot(gs4[0])
    for rets, color, label in [
        (port_ret, PALETTE["strategy"], "Strategy"),
        (bench_ret, PALETTE["benchmark"], "SP500"),
    ]:
        mu, sigma = rets.mean(), rets.std()
        ax_hist.hist(
            rets * 100, bins=80, alpha=0.4, color=color, density=True, label=label
        )
        x = np.linspace(rets.min() * 100, rets.max() * 100, 300)
        ax_hist.plot(x, stats.norm.pdf(x, mu * 100, sigma * 100), color=color, lw=1.5)
    ax_hist.axvline(0, color=PALETTE["neutral"], lw=0.8, linestyle="--")
    ax_hist.set_title("Return Distribution", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Daily Return (%)")
    ax_hist.legend(fontsize=8)
    _polish(ax_hist)

    ax_qq = fig.add_subplot(gs4[1])
    sorted_rets = np.sort(port_ret.dropna().values)
    n = len(sorted_rets)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    ax_qq.scatter(
        theoretical, sorted_rets * 100, s=2, color=PALETTE["strategy"], alpha=0.5
    )
    mn, mx = theoretical.min(), theoretical.max()
    ax_qq.plot(
        [mn, mx],
        [
            mn * port_ret.std() * 100 + port_ret.mean() * 100,
            mx * port_ret.std() * 100 + port_ret.mean() * 100,
        ],
        color=PALETTE["accent"],
        lw=1.5,
        label="Normal line",
    )
    ax_qq.set_title("QQ-Plot (Strategy)", fontsize=11, fontweight="bold")
    ax_qq.set_xlabel("Theoretical Quantiles")
    ax_qq.set_ylabel("Sample Quantiles (%)")
    ax_qq.legend(fontsize=8)
    _polish(ax_qq)

    ax_var = fig.add_subplot(gs4[2])
    cl_colors = [PALETTE["accent"], PALETTE["negative"], "#8e44ad"]
    for rets, ls, lbl in [(port_ret, "-", "Strategy"), (bench_ret, "--", "SP500")]:
        sorted_r = np.sort(rets.dropna().values)
        for cl, color in zip([0.90, 0.95, 0.99], cl_colors):
            ax_var.axvline(
                np.percentile(sorted_r, (1 - cl) * 100) * 100,
                color=color,
                lw=1.2,
                linestyle=ls,
                alpha=0.8,
            )
        ax_var.hist(
            rets * 100,
            bins=80,
            alpha=0.3,
            color=PALETTE["strategy"] if lbl == "Strategy" else PALETTE["benchmark"],
            density=True,
            label=lbl,
        )
    for cl, color in zip([0.90, 0.95, 0.99], cl_colors):
        ax_var.plot([], [], color=color, lw=1.5, label=f"VaR {int(cl * 100)}%")
    ax_var.set_title("VaR / CVaR (90/95/99%)", fontsize=11, fontweight="bold")
    ax_var.set_xlabel("Daily Return (%)")
    ax_var.legend(fontsize=7)
    _polish(ax_var)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 5: Annual returns bar chart (full width)
    # ══════════════════════════════════════════════════════════════════════════
    ax_ann = fig.add_subplot(outer[5])
    annual_port = port_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    annual_bench = bench_ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100
    years = annual_port.index.year
    x_pos = np.arange(len(years))
    w = 0.35
    bars1 = ax_ann.bar(
        x_pos - w / 2,
        annual_port.values,
        w,
        color=PALETTE["strategy"],
        alpha=0.85,
        label="Strategy",
    )
    bars2 = ax_ann.bar(
        x_pos + w / 2,
        annual_bench.values,
        w,
        color=PALETTE["benchmark"],
        alpha=0.85,
        label="SP500",
    )
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax_ann.text(
            bar.get_x() + bar.get_width() / 2,
            h + (0.5 if h >= 0 else -1.5),
            f"{h:.1f}%",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
            color=PALETTE["text"],
        )
    ax_ann.set_xticks(x_pos)
    ax_ann.set_xticklabels(years, fontsize=9)
    ax_ann.axhline(0, color=PALETTE["neutral"], lw=0.8)
    ax_ann.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_ann.set_title(
        "Annual Returns — Strategy vs SP500", fontsize=11, fontweight="bold"
    )
    ax_ann.legend(fontsize=9)
    _polish(ax_ann)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 6: Turnover | Holdings | Entry z-score threshold
    # ══════════════════════════════════════════════════════════════════════════
    gs6 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[6], wspace=0.35)

    ax_tv = fig.add_subplot(gs6[0])
    turnover = weights.diff().abs().sum(axis=1).resample("ME").sum() / 2
    ax_tv.bar(
        turnover.index, turnover.values, color=PALETTE["accent"], alpha=0.8, width=20
    )
    ax_tv.axhline(
        turnover.mean(),
        color=PALETTE["strategy"],
        lw=1.5,
        linestyle="--",
        label=f"Avg: {turnover.mean():.2%}",
    )
    ax_tv.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))
    ax_tv.set_title("Monthly Turnover", fontsize=10, fontweight="bold")
    ax_tv.legend(fontsize=8)
    _polish(ax_tv)

    ax_nh = fig.add_subplot(gs6[1])
    n_holdings = (weights > 0).sum(axis=1)
    ax_nh.plot(n_holdings.index, n_holdings.values, color=PALETTE["strategy"], lw=1.2)
    ax_nh.fill_between(
        n_holdings.index, n_holdings.values, alpha=0.2, color=PALETTE["strategy"]
    )
    ax_nh.set_title("Active Holdings Over Time", fontsize=10, fontweight="bold")
    ax_nh.set_ylabel("# Holdings")
    _polish(ax_nh)

    ax_ez = fig.add_subplot(gs6[2])

    def _row_cutoff(row):
        valid = row.dropna()
        if len(valid) < top_k:
            return np.nan
        return valid.nlargest(top_k).min()

    cutoff_z = signal_z.apply(_row_cutoff, axis=1)
    ax_ez.plot(cutoff_z.index, cutoff_z, color=PALETTE["accent"], lw=1.5)
    ax_ez.axhline(0, color=PALETTE["neutral"], lw=0.7, linestyle="--")
    ax_ez.set_title(
        f"Top-{top_k} Entry Z-score Threshold", fontsize=10, fontweight="bold"
    )
    ax_ez.set_ylabel("Min z-score to enter")
    _polish(ax_ez)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 7: Underwater plots
    # ══════════════════════════════════════════════════════════════════════════
    gs7 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[7], wspace=0.3)
    for col_i, (rets, color, label) in enumerate(
        [
            (port_ret, PALETTE["strategy"], "Strategy"),
            (bench_ret, PALETTE["benchmark"], "SP500"),
        ]
    ):
        ax_uw = fig.add_subplot(gs7[col_i])
        cum = (1 + rets).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax_uw.fill_between(dd.index, dd, 0, color=color, alpha=0.6)
        ax_uw.plot(dd.index, dd, color=color, lw=0.8)
        ax_uw.axhline(0, color=PALETTE["neutral"], lw=0.7)
        worst_idx = dd.idxmin()
        ax_uw.annotate(
            f"Max DD: {dd.min():.1f}%",
            xy=(worst_idx, dd.min()),
            xytext=(worst_idx, dd.min() * 0.5),
            arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=0.8),
            fontsize=9,
            color=PALETTE["text"],
        )
        ax_uw.set_title(f"{label} — Underwater Chart", fontsize=10, fontweight="bold")
        ax_uw.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        _polish(ax_uw)

    # ══════════════════════════════════════════════════════════════════════════
    # ROW 8: Monte Carlo fan + terminal distribution
    # ══════════════════════════════════════════════════════════════════════════
    gs8 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[8], wspace=0.3)
    ax_fan = fig.add_subplot(gs8[0])
    ax_ter = fig.add_subplot(gs8[1])

    BLOCK_SIZE = 21
    returns_arr = port_ret.dropna().values
    n_hist = len(returns_arr)
    np.random.seed(42)
    sim_cum = np.zeros((n_sims, horizon_days))
    for i in range(n_sims):
        path = []
        while len(path) < horizon_days:
            start = np.random.randint(0, n_hist - BLOCK_SIZE)
            path.extend(returns_arr[start : start + BLOCK_SIZE].tolist())
        sim_cum[i] = np.cumprod(1 + np.array(path[:horizon_days])) - 1

    last_date = port_ret.dropna().index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon_days
    )
    horizon_days_bd = len(future_dates)
    sim_cum = sim_cum[:, :horizon_days_bd]
    pcts = [1, 5, 25, 50, 75, 95, 99]
    bands = {p: np.percentile(sim_cum, p, axis=0) for p in pcts}
    actual_cum = np.cumprod(1 + returns_arr) - 1

    ax_fan.plot(
        port_ret.dropna().index,
        actual_cum * 100,
        color=PALETTE["strategy"],
        lw=2,
        label="Actual",
    )
    ax_fan.axvline(last_date, color=PALETTE["neutral"], lw=1, linestyle="--", alpha=0.7)
    for lo, hi, alpha, color, lbl in [
        (5, 95, 0.15, "#8e44ad", "90% CI"),
        (25, 75, 0.30, "#2980b9", "50% CI"),
        (1, 99, 0.10, "#c0392b", "98% CI"),
    ]:
        ax_fan.fill_between(
            future_dates,
            bands[lo] * 100 + actual_cum[-1] * 100,
            bands[hi] * 100 + actual_cum[-1] * 100,
            alpha=alpha,
            color=color,
            label=lbl,
        )
    ls_map = {
        1: ("--", PALETTE["negative"]),
        5: ("--", "#d6d921"),
        25: ("--", "#5dade2"),
        50: ("-", PALETTE["strategy"]),
        75: ("--", "#5dade2"),
        95: ("--", "#d6d921"),
        99: ("--", PALETTE["negative"]),
    }
    for p in pcts:
        ls, color = ls_map[p]
        ax_fan.plot(
            future_dates,
            bands[p] * 100 + actual_cum[-1] * 100,
            color=color,
            lw=1.5 if p == 50 else 0.9,
            linestyle=ls,
        )
        ax_fan.annotate(
            f"p{p}: {bands[p][-1] * 100:.1f}%",
            xy=(future_dates[-1], bands[p][-1] * 100 + actual_cum[-1] * 100),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            va="center",
        )
    prob_pos = (sim_cum[:, -1] > 0).mean()
    ax_fan.text(
        0.02,
        0.97,
        f"P(profit): {prob_pos:.1%}\nMedian: {bands[50][-1]:+.1%}\n90% CI: [{bands[5][-1]:+.1%}, {bands[95][-1]:+.1%}]",
        transform=ax_fan.transAxes,
        fontsize=8,
        va="top",
        fontfamily="monospace",
        color=PALETTE["text"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=PALETTE["panel_bg"],
            edgecolor=PALETTE["grid"],
            alpha=0.9,
        ),
    )
    ax_fan.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y / 100 + 1:.1f}x")
    )
    ax_fan.set_title(
        f"Monte Carlo Fan — {n_sims:,} paths ({horizon_days // 252}yr)",
        fontsize=10,
        fontweight="bold",
    )
    ax_fan.legend(fontsize=8)
    _polish(ax_fan)

    ter_pct = sim_cum[:, -1] * 100
    kde_ter = stats.gaussian_kde(ter_pct, bw_method="scott")
    x_ter = np.linspace(ter_pct.min(), ter_pct.max(), 500)
    y_ter = kde_ter(x_ter)
    ax_ter.hist(ter_pct, bins=60, color=PALETTE["neutral"], alpha=0.25, density=True)
    ax_ter.plot(x_ter, y_ter, color=PALETTE["benchmark"], lw=2)
    ax_ter.fill_between(
        x_ter,
        y_ter,
        where=(x_ter >= 0),
        alpha=0.20,
        color=PALETTE["strategy"],
        label="Gain",
    )
    ax_ter.fill_between(
        x_ter,
        y_ter,
        where=(x_ter < 0),
        alpha=0.20,
        color=PALETTE["negative"],
        label="Loss",
    )
    for p, ls, col in [
        (5, ":", PALETTE["negative"]),
        (25, "--", PALETTE["neutral"]),
        (50, "-", PALETTE["strategy"]),
        (75, "--", PALETTE["neutral"]),
        (95, ":", PALETTE["accent"]),
    ]:
        val = np.percentile(ter_pct, p)
        ax_ter.axvline(val, color=col, lw=1.3, linestyle=ls, label=f"p{p}: {val:+.1f}%")
    p_pos = (ter_pct > 0).mean()
    ax_ter.text(
        0.97,
        0.96,
        f"P(gain)={p_pos:.1%}",
        transform=ax_ter.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        fontfamily="monospace",
        color=PALETTE["text"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=PALETTE["panel_bg"],
            edgecolor=PALETTE["grid"],
            alpha=0.9,
        ),
    )
    ax_ter.set_title("MC Terminal Return Distribution", fontsize=10, fontweight="bold")
    ax_ter.set_xlabel("Terminal Return (%)")
    ax_ter.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax_ter.legend(fontsize=7, ncol=2)
    _polish(ax_ter)

    # ── Title banner ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Strategy Tearsheet — Idiosyncratic Momentum",
        fontsize=18,
        fontweight="bold",
        color=PALETTE["text"],
        y=0.985,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["dark_bg"])
    plt.show()
    print(f"✓ Saved tearsheet → {save_path}")
