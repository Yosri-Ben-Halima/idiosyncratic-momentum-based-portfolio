"""Interactive figures from the shared engine's measured results."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy import stats
from src.engine.diagnostics import recovery_years

COLORS = ['#36d6ba','#669cff','#f5ba62']


def clean_json(value):
    if isinstance(value,dict):
        return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    return value


def figure(title,traces,ytitle='',wide=False):
    chart = go.Figure(traces)
    chart.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',font={'family':'Segoe UI, sans-serif','color':'#aab9cf','size':12},
        margin={'l':55,'r':20,'t':20,'b':45},height=340,hovermode='x unified',
        legend={'orientation':'h','y':1.12,'x':0},yaxis_title=ytitle,
        colorway=COLORS,xaxis={'gridcolor':'#203047','zeroline':False},
        yaxis={'gridcolor':'#203047','zerolinecolor':'#34425a'})
    return {'title':title,'wide':wide,'plot':json.loads(chart.to_json())}


def series_traces(data,scale=1):
    if isinstance(data,pd.Series):
        data = data.to_frame(data.name or 'Strategy')
    return [go.Scatter(x=data.index,y=data[col]*scale,name=col,mode='lines',
        line={'width':2,'color':COLORS[i%3]}) for i,col in enumerate(data)]


def build_dashboard_payload(result):
    d,cfg = result.diagnostics,result.config
    returns = result.returns
    port = returns.Strategy
    figures = {key:[] for key in ['performance','risk','portfolio']}
    perf = figures['performance']
    perf.append(figure('Cumulative return',series_traces(d['cumulative'],100),'Return (%)',True))
    perf.append(figure('Drawdown',series_traces(d['drawdown'],100),'Drawdown (%)',True))
    perf.append(figure(f"Rolling {cfg['momentum_window']}-session Sharpe",series_traces(d['rolling_sharpe']),'Ratio'))
    perf.append(figure('Rolling beta',series_traces(d['rolling_beta']),'Beta'))
    perf.append(figure('Rolling annualized alpha',series_traces(d['rolling_alpha'],100),'Alpha (%)'))
    risk = figures['risk']
    risk.append(figure('Daily return distribution',[go.Histogram(x=returns[col]*100,name=col,
        opacity=.55,histnorm='probability density',nbinsx=65,marker_color=COLORS[i]) for i,col in enumerate(returns)],'Density'))
    theoretical = stats.norm.ppf(np.linspace(.01,.99,len(port)))
    risk.append(figure('Normal QQ plot',[go.Scatter(x=theoretical,y=np.sort(port)*100,mode='markers',name='Observed',marker={'size':3}),
        go.Scatter(x=theoretical,y=(theoretical*port.std()+port.mean())*100,name='Normal reference',mode='lines')],'Daily return (%)'))
    tails = []
    for col in returns:
        values = returns[col]
        var = [values.quantile(1-cl) for cl in [.9,.95,.99]]
        cvar = [values[values<=v].mean() for v in var]
        tails.extend([go.Bar(x=['90%','95%','99%'],y=np.array(var)*100,name=col+' VaR'),
                      go.Bar(x=['90%','95%','99%'],y=np.array(cvar)*100,name=col+' CVaR')])
    risk.append(figure('Historical VaR / CVaR',tails,'Daily return (%)'))
    for col in returns:
        monthly = d['monthly'][col]
        table = pd.DataFrame({'year':monthly.index.year,'month':monthly.index.month,'return':monthly.values}).pivot(index='year',columns='month',values='return').reindex(columns=range(1,13))
        risk.append(figure(col+' · monthly returns',[go.Heatmap(x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
            y=table.index.astype(str),z=table.to_numpy()*100,colorscale='RdBu',zmid=0,
            hovertemplate='%{y} %{x}<br>%{z:.2f}%<extra></extra>',colorbar={'title':'%','thickness':10})]))
        risk.append(figure(col+' · underwater',[go.Scatter(x=d['drawdown'].index,y=d['drawdown'][col]*100,
            fill='tozeroy',mode='lines',line={'color':COLORS[0] if col=='Strategy' else COLORS[1]},name=col)],'Drawdown (%)'))
    portfolio = figures['portfolio']
    portfolio.append(figure('Annual returns',[go.Bar(x=d['annual'].index.year.astype(str),y=d['annual'][col]*100,
        name=col,marker_color=COLORS[i]) for i,col in enumerate(returns)],'Return (%)',True))
    portfolio.append(figure('Monthly turnover',[go.Bar(x=d['turnover'].index,y=d['turnover']*100,marker_color=COLORS[2],name='One-way turnover')],'Turnover (%)'))
    portfolio.append(figure('Active holdings',series_traces(d['holdings']),'Stocks'))
    portfolio.append(figure(f"Top-{cfg['top_k']} entry z-score",series_traces(d['entry_z']),'Z-score'))
    portfolio.append(figure('Trailing risk diversification',series_traces(result.risk_daily['risk_diversification_ratio']),'Diversification ratio'))
    mc = d['mc']
    # Preserve the PNG's additive fan presentation and use its exact simulated arrays.
    last = d['cumulative'].Strategy.iloc[-1]*100
    fan = series_traces(d['cumulative'].Strategy,100)
    for lo,hi,opacity in [(1,99,.08),(5,95,.12),(25,75,.24)]:
        fan.append(go.Scatter(x=mc['dates'],y=mc['bands'][lo]*100+last,mode='lines',line={'width':0},showlegend=False))
        fan.append(go.Scatter(x=mc['dates'],y=mc['bands'][hi]*100+last,mode='lines',line={'width':0},
            fill='tonexty',fillcolor=f'rgba(54,214,186,{opacity})',name=f'{hi-lo}% interval'))
    fan.append(go.Scatter(x=mc['dates'],y=mc['bands'][50]*100+last,mode='lines',name='Simulation median',line={'color':COLORS[2]}))
    portfolio.append(figure('Block-bootstrap Monte Carlo · additive return fan',fan,'Return (%)',True))
    portfolio.append(figure('Monte Carlo terminal return',[go.Histogram(x=mc['terminal']*100,nbinsx=60,marker_color=COLORS[0],name='Simulated paths')],'Paths'))
    contributions = result.contributions
    holdings = result.weights.iloc[-1][result.weights.iloc[-1]>0].sort_values(ascending=False)
    if not contributions.empty:
        portfolio.append(figure('Capital weight vs absolute risk share',[
            go.Bar(x=contributions.index,y=contributions.weight*100,name='Capital weight',marker_color=COLORS[1]),
            go.Bar(x=contributions.index,y=contributions.absolute_risk_share*100,name='Absolute risk share',marker_color=COLORS[2])],'Share (%)',True))
    surfaces = None
    if result.sensitivity is not None:
        study = result.sensitivity
        modes = {}
        for period in ['full','early','recent']:
            for field in ['Sharpe','Calmar','Total Return','CAGR','Vol','Max DD']:
                key = period+'_'+field
                scale = -100 if field=='Max DD' else 100 if field in ('Total Return','CAGR','Vol') else 1
                modes[key] = {'label':field if field!='Max DD' else 'Drawdown loss','scale':scale,'unit':'%' if abs(scale)==100 else ''}
        for key,label,scale,unit in [('mean_holdings','Average holdings',1,'stocks'),
            ('risk_diversification_ratio','Diversification ratio',1,''),('risk_top_share','Top-stock risk share',100,'%'),
            ('risk_effective_contributors','Effective risk contributors',1,'stocks')]:
            modes[key] = dict(label=label,scale=scale,unit=unit)
        for key,mode in modes.items():
            mode['z'] = study.grid.pivot(index='window',columns='cutoff',values=key).reindex(index=study.windows,columns=study.cutoffs).to_numpy()*mode.pop('scale')
        surfaces = {'windows':study.windows,'cutoffs':study.cutoffs,'metrics':modes,
                    'baseline':{'window':cfg['momentum_window'],'cutoff':cfg['z_threshold']},
                    'summary':study.summary(),'split_date':study.split_date}
    rows = [{'ticker':ticker,'weight':float(weight),
             'risk_share':float(contributions.loc[ticker,'absolute_risk_share']) if ticker in contributions.index else None}
            for ticker,weight in holdings.items()]
    payload = {'config':cfg,'snapshot_id':result.snapshot_id,'dates':[str(returns.index[0].date()),str(returns.index[-1].date())],
        'metrics':result.metrics,'figures':figures,'surfaces':surfaces,'holdings':rows,'warnings':result.warnings,
        'risk_snapshot_date':contributions.attrs.get('snapshot_date'),
        'recovery_years':{name:recovery_years(m['Max DD'],m['CAGR']) for name,m in result.metrics.items()},
        'risk_means':result.risk_daily.mean().to_dict(),
        'mean_holdings':float(d['holdings'].mean()),'observations':len(returns)}
    # PlotlyJSONEncoder also turns datetime axes and arrays into JSON-compatible lists.
    return clean_json(json.loads(json.dumps(clean_json(payload),cls=PlotlyJSONEncoder)))
