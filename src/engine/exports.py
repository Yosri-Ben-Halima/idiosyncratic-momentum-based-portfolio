"""Export a completed computation without running the strategy again."""
from pathlib import Path
import zipfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from src.sensitivity.analysis import write_json


def export_result(result,directory,include_png=True):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    result.returns.to_csv(directory/'daily_returns.csv')
    pd.DataFrame(result.metrics).to_csv(directory/'metrics.csv')
    result.diagnostics['annual'].to_csv(directory/'annual_returns.csv')
    result.weights.to_csv(directory/'weights.csv')
    result.weights.iloc[-1][result.weights.iloc[-1]>0].sort_values(ascending=False).to_csv(directory/'latest_holdings.csv',header=['weight'])
    result.risk_daily.to_csv(directory/'risk_daily.csv')
    result.contributions.to_csv(directory/'risk_contributions.csv')
    write_json(directory/'run_metadata.json',{**result.provenance,'config':result.config,
        'first_return_date':str(result.returns.index[0].date()),'last_return_date':str(result.returns.index[-1].date()),
        'observations':len(result.returns),'warnings':result.warnings})
    if result.sensitivity is not None:
        result.sensitivity.save(directory/'sensitivity',snapshot_id=result.snapshot_id)
    if include_png:
        from src.visualizations import plot_tearsheets
        cfg = result.config
        plot_tearsheets(result.returns.Strategy,result.returns.SP500,
            result.metrics['Strategy'],result.metrics['SP500'],result.signal,result.weights,
            cfg['top_k'],cfg['rf_daily'],cfg['momentum_window'],cfg['monte_carlo_runs'],
            cfg['monte_carlo_days'],save_prefix=str(directory/'tearsheet'),
            sensitivity_result=result.sensitivity,simulation=result.diagnostics['mc'])
    files = {p.name:p for p in directory.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.png')}
    bundle = directory/'results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in ('.csv','.json','.png'):
                archive.write(p,p.relative_to(directory))
    files['results.zip'] = bundle
    return files
