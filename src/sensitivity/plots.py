"""Static and interactive sensitivity plots; reusable by the tearsheet bundle."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np


def _bounds(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return (float(finite.min()), float(finite.max())) if len(finite) else (0, 1)


def _surface_figure(result, dark=False, risk=False, concentration_only=False):
    rows = result.grid
    start, end = result.start_date, result.end_date
    WINDOWS, CUTOFFS = result.windows, result.cutoffs
    current_window = result.config["momentum_window"]
    current_cutoff = result.config["z_threshold"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#0d1117" if dark else "white",
                         "axes.facecolor": "#161b22" if dark else "white",
                         "text.color": "#c9d1d9" if dark else "black",
                         "axes.labelcolor": "#c9d1d9" if dark else "black",
                         "xtick.color": "#c9d1d9" if dark else "black",
                         "ytick.color": "#c9d1d9" if dark else "black"})
    x, y = np.meshgrid(CUTOFFS, WINDOWS)
    fig = plt.figure(figsize=(20, 11.25))
    modes = ([("risk_diversification_ratio", "Mean diversification ratio"),
              ("risk_top_share", "Mean top-stock risk share (%)"),
              ("full_Vol", "Realized volatility (%)"),
              ("full_Max DD", "Maximum drawdown loss (%)")]
             if risk else [("full_Sharpe", "Sharpe"),
                                       ("full_Calmar", "Calmar"),
                                       ("full_Total Return", "Total return (%)"),
                                       ("mean_holdings", "Average holdings")])
    if concentration_only:
        modes = modes[:2]
    for i, (value, label) in enumerate(modes):
        ax = fig.add_subplot(1 if concentration_only else 2, 2, i + 1,
                             projection="3d", computed_zorder=False)
        scale = (-100 if value == "full_Max DD" else
                 100 if value.endswith("Total Return") or value in ("risk_top_share", "full_Vol") else 1)
        z = matrix(rows, value) * scale
        surf = None
        if not np.isfinite(z).any():
            ax.text2D(0.25, 0.5, "N/A: insufficient usable data", transform=ax.transAxes)
        elif len(WINDOWS) < 2 or len(CUTOFFS) < 2:
            ax.plot(x.ravel(), y.ravel(), z.ravel(), color="#2ecc71")
        else:
            surf = ax.plot_surface(x, y, z, cmap="viridis", linewidth=0,
                                   edgecolor="none", antialiased=True, alpha=1,
                                   norm=Normalize(*_bounds(z)), zorder=1)
        base = rows[(rows.window == current_window) & (rows.cutoff == current_cutoff)].iloc[0]
        current_label = f"{base[value]*scale:,.1f}%" if abs(scale) == 100 else f"{base[value]:.2f}"
        known = np.isfinite(base[value])
        current_label = current_label if known else "N/A"
        ax.scatter([current_cutoff] if known else [], [current_window] if known else [],
                   [base[value]*scale] if known else [], color="#bc4b00", s=85,
                   edgecolor="black", depthshade=False, zorder=10,
                   label=f"Current: {current_label}")
        ax.set(xlabel="Z-score cutoff", ylabel="Window (trading days)",
               zlabel=label, title=f"Full-period {label}")
        ax.view_init(elev=30, azim=-125)
        ax.set_box_aspect((1.2, 1.2, .8))
        if abs(scale) == 100:
            ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.zaxis.labelpad = 16 if abs(scale) == 100 else 10
        ax.set_xticks(np.linspace(min(CUTOFFS), max(CUTOFFS), min(5, len(CUTOFFS))))
        ax.tick_params(colors="#c9d1d9" if dark else "black")
        if dark:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.086, 0.106, 0.133, 1))
                axis._axinfo["grid"]["color"] = "#30363d"
        ax.legend(loc="upper left", frameon=False)
        if surf is not None:
            fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1, format=FuncFormatter(lambda v, _: f"{v:,.0f}") if abs(scale) == 100 else None)
    heading = ("Sheet 5 / 5 - Risk concentration" if risk else "Sheet 4 / 5 - Sensitivity") if dark else ("Risk diversification and concentration" if risk else "Signal horizon and concentration")
    fig.suptitle(f"{heading} | {start} to {end}",
                 fontsize=17, fontweight="bold", y=0.97)
    options = result.config.get("sensitivity", {})
    caption = (f"Risk snapshots use preceding {options.get('risk_window',252)} sessions; monthly and on weight changes. Means exclude cash/unknown risk.\n"
               "Diversification ratio: higher means greater correlation/volatility diversification. Top-stock share uses normalized absolute Euler contributions.\n"
               "Volatility and drawdown are realized over the full backtest. Associations across settings do not establish causation.\n"
               if risk else "The window controls both CAPM estimation and residual summation. Other rules and data are fixed.\n"
               "Holdings are averaged over all backtest sessions, including cash sessions as zero.\n")
    fig.text(0.04, 0.025,
             f"{len(rows)} tested settings; unsmoothed mesh. Orange dot: current {current_window} days / z > {current_cutoff:g}.\n"
             + caption +
             "Exploratory simulation: current-member universe; fixed equal weights; incomplete trading costs; price-only benchmark.",
             fontsize=10, color="#95a5a6" if dark else "#444444")
    fig.subplots_adjust(left=0.04, right=0.95, top=0.91, bottom=0.15, wspace=0.08, hspace=0.22)
    return fig


def plot_sensitivity_sheet(result, save_path):
    """Add a fourth 16:9 page using the same measured surface as the exports."""
    with plt.rc_context():
        fig = _surface_figure(result, dark=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"Saved sensitivity tearsheet: {save_path}")


def plot_risk_sheet(result, save_path):
    """Fifth tearsheet: diversification, risk dominance and realized losses."""
    with plt.rc_context():
        fig = _surface_figure(result, dark=True, risk=True, concentration_only=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"Saved risk concentration tearsheet: {save_path}")


def render_sensitivity(result, output):
    """Export surfaces, period heatmaps, and a standalone rotatable HTML plot."""
    with plt.rc_context():
        _render_sensitivity(result, output)


def matrix(rows, value):
    return rows.pivot(index="window", columns="cutoff", values=value).reindex(
        index=sorted(rows.window.unique()), columns=sorted(rows.cutoff.unique())
    ).to_numpy()


def _render_sensitivity(result, output):
    rows = result.grid
    start, end = result.start_date, result.end_date
    WINDOWS, CUTOFFS = result.windows, result.cutoffs
    current_window = result.config["momentum_window"]
    current_cutoff = result.config["z_threshold"]
    recent_title = f"{result.split_date} onward"
    early_title = f"Before {result.split_date}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig = _surface_figure(result)
    fig.savefig(output / "surfaces.png", dpi=160)
    fig.savefig(output / "surfaces.pdf")
    plt.close(fig)

    if "risk_diversification_ratio" in rows:
        fig = _surface_figure(result, risk=True)
        fig.savefig(output / "risk_surfaces.png", dpi=160)
        fig.savefig(output / "risk_surfaces.pdf")
        plt.close(fig)
        fig = _surface_figure(result, risk=True, concentration_only=True)
        fig.savefig(output / "concentration_surfaces.png", dpi=160)
        fig.savefig(output / "concentration_surfaces.pdf")
        plt.close(fig)
        _risk_impact(result, output)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    sharpe_values = rows[["full_Sharpe", "early_Sharpe", "recent_Sharpe"]].to_numpy()
    common = Normalize(*_bounds(sharpe_values))
    for ax, value, title in zip(axes.ravel(), ["full_Sharpe", "early_Sharpe", "recent_Sharpe", "mean_holdings"],
                              ["Full-period Sharpe", f"{early_title} Sharpe", f"{recent_title} Sharpe", "Mean number of holdings"]):
        z = matrix(rows, value)
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis",
                       norm=common if value.endswith("Sharpe") else None)
        ax.set_xticks(range(len(CUTOFFS)), [f"{c:g}" for c in CUTOFFS], fontsize=10)
        ax.set_yticks(range(len(WINDOWS)), WINDOWS)
        ax.set(xlabel="Z-score cutoff", ylabel="Window (trading days)", title=title)
        for yi in range(len(WINDOWS)):
            for xi in range(len(CUTOFFS)):
                shade = im.norm(z[yi, xi]) if np.isfinite(z[yi, xi]) else 0
                ax.text(xi, yi, ("N/A" if not np.isfinite(z[yi, xi]) else f"{z[yi, xi]:.2f}" if value.endswith("Sharpe") else f"{z[yi, xi]:.1f}"),
                        ha="center", va="center", fontsize=8.5,
                        color="black" if shade > 0.65 else "white")
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((CUTOFFS.index(current_cutoff)-0.5, WINDOWS.index(current_window)-0.5), 1, 1,
                               fill=False, edgecolor="#f08a30", linewidth=3))
        if np.isfinite(z).any():
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Local stability, period sensitivity, and concentration", fontsize=18, y=0.97)
    fig.text(0.04, 0.025,
             f"Source: saved Yahoo Finance prices, {start} to {end}. Orange box: current settings.\n"
             "Sharpe panels share a color scale. Rows/columns are tested settings (uneven numeric spacing).\n"
             "Recent-period results reuse the same historical data; they are a diagnostic, not an untouched validation set.",
             fontsize=11, color="#444444")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.12, wspace=0.23, hspace=0.25)
    fig.savefig(output / "heatmaps.png", dpi=160)
    plt.close(fig)

    # Standalone plot with bundled JavaScript: no network is needed to view it.
    import plotly.graph_objects as go
    chart = go.Figure()
    modes = [("full_Sharpe", "Full-period Sharpe"), ("recent_Sharpe", f"{recent_title} Sharpe"),
             ("full_Calmar", "Full-period Calmar"), ("recent_Calmar", f"{recent_title} Calmar"),
             ("full_Total Return", "Full-period total return (%)"),
             ("recent_Total Return", f"{recent_title} total return (%)"),
             ("mean_holdings", "Full-period average holdings")]
    risk_labels = {"risk_diversification_ratio": "Mean diversification ratio",
                   "risk_top_share": "Mean top-stock risk share (%)",
                   "risk_effective_contributors": "Mean effective risk contributors",
                   "full_Vol": "Full-period realized volatility (%)",
                   "full_Max DD": "Full-period maximum drawdown loss (%)"}
    if "risk_diversification_ratio" in rows:
        modes.extend(risk_labels.items())
    base = rows[(rows.window == current_window) & (rows.cutoff == current_cutoff)].iloc[0]
    buttons = []
    for i, (value, label) in enumerate(modes):
        metric = (risk_labels[value] if value in risk_labels else
                  "Average holdings" if value == "mean_holdings" else value.split("_", 1)[1])
        scale = (-100 if value == "full_Max DD" else
                 100 if metric == "Total Return" or value in ("risk_top_share", "full_Vol") else 1)
        z = matrix(rows, value) * scale
        scale_columns = [value] if value == "mean_holdings" or value in risk_labels else [f"full_{metric}", f"recent_{metric}"]
        axis_title = "Total return (%)" if metric == "Total Return" else metric
        bounds = sorted(np.array(_bounds(rows[scale_columns].to_numpy())) * scale)
        hover_value = ("Value: %{z:,.1f}%" if abs(scale) == 100 else
                       "Average holdings: %{z:.2f}" if value == "mean_holdings" else "Ratio: %{z:.3f}")
        chart.add_trace(go.Surface(x=CUTOFFS, y=WINDOWS, z=z, colorscale="Viridis",
            cmin=bounds[0], cmax=bounds[1],
            visible=i == 0, name=label, colorbar={"title":axis_title},
            customdata=matrix(rows, "mean_holdings"),
            hovertemplate="Window: %{y} sessions<br>Z-score cutoff: %{x}<br>"+hover_value+"<br>Mean holdings: %{customdata:.1f}<extra></extra>"))
        chart.add_trace(go.Scatter3d(x=[current_cutoff],y=[current_window],z=[base[value]*scale],mode="markers",
            marker={"size":7,"color":"#bc4b00","line":{"color":"black","width":2}},
            visible=i == 0,name="Current settings",
            hovertemplate=f"Current: {current_window} days / cutoff {current_cutoff:g}<br>"+hover_value+"<extra></extra>"))
        visible = [False]*(2*len(modes))
        visible[2*i:2*i+2] = [True, True]
        buttons.append({"label":label,"method":"update","args":[{"visible":visible},
                        {"scene.zaxis.title":axis_title}]})
    chart.update_layout(title=f"Window / cutoff sensitivity | {start} to {end}",
        height=850, paper_bgcolor="white", font={"family":"Arial","size":14},
        scene={"xaxis_title":"Z-score cutoff","yaxis_title":"Window (trading days)",
               "zaxis_title":"Sharpe","camera":{"eye":{"x":1.6,"y":-1.6,"z":1.1}}},
        margin={"l":30,"r":30,"b":135,"t":130}, uirevision="keep-camera",
        updatemenus=[{"buttons":buttons,"direction":"down","x":0,"y":1.12}],
        legend={"x":0.7,"y":1.1},
        annotations=[{"text":f"Drag to rotate; scroll to zoom. The mesh joins {len(rows)} measured settings without smoothing.<br>"
            "Current settings are marked in orange. Same metric uses a shared color scale across periods.<br>"
            "Average holdings includes cash sessions as zero.<br>"
            "Risk means use trailing covariance snapshots and exclude cash/unknown risk; concentration uses absolute contributions.<br>"
            "Window controls CAPM fit and residual summation. Current constituents; fixed equal weights; incomplete costs.<br>"
            "Period splits are diagnostics, not untouched validation sets. Data: Yahoo Finance snapshot.",
            "xref":"paper","yref":"paper","x":0,"y":-0.12,"showarrow":False,"align":"left","font":{"size":13}}])
    chart.write_html(output / "interactive_surface.html", include_plotlyjs=True,
                     config={"displaylogo":False,"responsive":True})


def _risk_impact(result, output):
    rows = result.grid
    valid = rows.dropna(subset=["risk_diversification_ratio", "full_Vol", "full_Max DD"])
    has_contributions = result.latest_risk_contributions is not None and not result.latest_risk_contributions.empty
    fig, axes = plt.subplots(1, 3 if has_contributions else 2, figsize=(20 if has_contributions else 15, 7))
    for ax, metric, label, scale in zip(axes[:2], ["full_Vol", "full_Max DD"],
                                      ["Realized volatility (%)", "Maximum drawdown loss (%)"], [100, -100]):
        points = ax.scatter(valid.risk_diversification_ratio, valid[metric]*scale,
                            s=20+valid.mean_holdings*4, c=valid.cutoff, cmap="viridis",
                            edgecolor="#333333", linewidth=.4, alpha=.8)
        base = result.baseline
        if np.isfinite(base.risk_diversification_ratio):
            ax.scatter([base.risk_diversification_ratio], [base[metric]*scale],
                       marker="*", s=220, color="#bc4b00", edgecolor="black", label="Current setting")
            ax.legend(frameon=False)
        ax.set(xlabel="Mean trailing diversification ratio", ylabel=label)
        ax.grid(alpha=.2)
        fig.colorbar(points, ax=ax, label="Z-score cutoff")
    if has_contributions:
        contributions = result.latest_risk_contributions
        positions = np.arange(len(contributions))
        axes[2].barh(positions-.18, contributions.weight*100, height=.35, color="#606870", label="Capital weight")
        axes[2].barh(positions+.18, contributions.absolute_risk_share*100, height=.35, color="#2a7f9e", label="Absolute risk share")
        axes[2].set_yticks(positions, contributions.index)
        axes[2].invert_yaxis()
        axes[2].set(xlabel="Share (%)", title=f"Latest portfolio | risk snapshot {contributions.attrs.get('snapshot_date')}")
        axes[2].legend(frameon=False)
    fig.suptitle("Diversification versus realized risk across parameter settings", fontsize=16)
    fig.text(.05,.02,"Larger dots = more holdings. Higher diversification ratio = more risk reduction from combining stocks.\n"
             "These are descriptive associations across the same backtest, not controlled causal estimates.",fontsize=11)
    fig.tight_layout(rect=(0,.1,1,.94))
    fig.savefig(output / "risk_impact.png", dpi=160)
    plt.close(fig)

