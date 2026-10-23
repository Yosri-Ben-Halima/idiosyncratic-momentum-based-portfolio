/* Presentation only: every measured value comes from the current run's audit API. */
'use strict';
window.AuditView=(()=>{
  const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=(value,digits=2)=>value==null?'N/A':Number(value).toLocaleString(undefined,{maximumFractionDigits:digits});
  const percent=value=>value==null?'N/A':number(value*100,2)+'%';
  const delta=value=>value==null?'N/A':(value>0?'+':'')+number(value*100,2)+' pp';
  const labels={passed:'Checked · unchanged',failed:'Check failed',issue:'Issue found',insufficient:'Insufficient evidence',not_tested:'Not tested'};
  const pill=(status,text)=>`<span class="audit-pill ${esc(status)}">${esc(text||labels[status])}</span>`;
  const card=(label,value,detail)=>`<article class="audit-stat"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(detail)}</small></article>`;
  const options={responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d']};
  const layout={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Segoe UI, sans-serif',color:'#aab9cf',size:12},margin:{l:20,r:55,t:20,b:45},showlegend:false};
  function timeline(config){
    const w=config.momentum_window,s=config.momentum_skip;
    return `<article class="audit-panel audit-timeline"><div class="audit-heading"><span class="audit-number">01</span><div><h3>What was known at selection?</h3><p>Each block describes a different part of the information history.</p></div></div>
      <div class="audit-time-track" aria-label="Earlier estimation history, ${w} residual sessions, ${s} skipped sessions, then selection">
        <div class="time-block estimation"><strong>Earlier history</strong><span>Rolling CAPM fits</span><small>Preceding ${esc(w)} sessions per residual</small></div>
        <div class="time-block momentum"><strong>${esc(w)} sessions</strong><span>Residual momentum</span><small>t − ${esc(w+s-1)} through t − ${esc(s)}</small></div>
        <div class="time-block skip"><strong>${esc(s)} sessions</strong><span>${s?'Excluded from momentum':'No skip'}</span><small>${s?'Most recent observations':'Momentum includes t'}</small></div>
        <div class="time-block selection"><strong>Selection →</strong><span>${esc({'ME':'Month end','QE':'Quarter end','W-FRI':'Weekly · Friday'}[config.rebalance_freq]||config.rebalance_freq)}</span><small>z &gt; ${esc(config.z_threshold)} · cap ${esc(config.top_k)}</small></div>
      </div><p class="audit-caption">Schematic spacing. t is the last session strictly before selection. The skip is a momentum choice; past-only estimation and selection enforce chronology. Eligibility uses earlier price history, and portfolio returns use lagged weights.</p></article>`;
  }
  function checkCards(report){
    return report.checks.map(check=>{
      const diff=check.differences;
      const evidence=diff?`<dl class="audit-deviations">${[['Signal',diff.signal],['Weights',diff.weights],['Returns',diff.returns]].map(([name,value])=>`<div><dt>${name} Δ max</dt><dd>${esc(value==null?'N/A':value===0?'0':value.toExponential(2))}</dd></div>`).join('')}</dl>`:'';
      return `<article class="audit-check"><div>${pill(check.status)}</div><h4>${esc(check.title)}</h4><p>${esc(check.description)}</p>${evidence}${check.missing_changes?`<small>Changed missing-value states: ${esc(check.missing_changes)}</small>`:''}</article>`;
    }).join('');
  }
  function render(result,report,busy){
    const root=document.querySelector('#audit-content');
    for(const plot of root.querySelectorAll('.js-plotly-plot'))Plotly.purge(plot);
    document.querySelector('#audit-run').disabled=busy||!result;
    document.querySelector('#audit-download').disabled=busy||!report;
    document.querySelector('#audit-run').textContent=report?'Recheck this run →':'Run audit →';
    root.dataset.snapshot=report?.scope.snapshot_id||'';
    if(!result){root.innerHTML='<div class="audit-intro"><span class="eyebrow">EVIDENCE STARTS WITH A RUN</span><h3>Run a backtest, then challenge it.</h3><p>The audit will use its completed settings and frozen prices, even if you edit parameters afterward.</p><div class="audit-preview"><span>01 · Information timing</span><span>02 · Return comparisons</span><span>03 · Concentration & gaps</span></div></div>';return;}
    if(!report){root.innerHTML=`<div class="audit-intro"><span class="eyebrow">YOUR COMPLETED RUN · READY TO CHECK</span><h3>Strong results deserve strong checks.</h3><p>Run the audit to change future prices, remove future data, and compare modelling assumptions against this exact backtest.</p><div class="audit-preview"><span>${esc(result.dates.join(' → '))}</span><span>${esc(result.config.momentum_window)} sessions / z &gt; ${esc(result.config.z_threshold)}</span><span>Snapshot ${esc(result.snapshot_id.slice(0,12))}</span></div><small>Several replays run locally. Previous results stay available. A passing timing check does not validate expected returns.</small></div>${timeline(result.config)}`;return;}
    const r=report,timing=r.timing.status,c=r.concentration;
    const title=timing==='passed'?'Past decisions stayed unchanged.':timing==='failed'?'A timing or replay check failed.':'There is not enough evidence for a timing verdict.';
    const paired=r.comparisons.find(row=>row.id==='paired');
    const verdict=timing==='passed'?'Future-price shocks and deletion left earlier signals, holdings and returns unchanged at the tested cutoff.':r.timing.note;
    root.innerHTML=`<div class="audit-verdict ${esc(timing)}"><div class="audit-verdict-main"><span class="eyebrow">SIGNAL TIMING · ${esc(labels[timing])}</span><h3>${esc(title)}</h3><p>${esc(verdict)}</p></div><aside>${pill('not_tested','Historical realism · not validated')}<p>The universe is held fixed. Survivorship bias and historical data quality remain open questions.</p></aside></div>
      <div class="audit-provenance"><span>Completed run: <b>${esc(r.scope.dates.join(' → '))}</b></span><span>${esc(r.scope.stocks)} stocks · ${esc(r.scope.observations)} sessions</span><span>Future starts: <b>${esc(r.scope.cutoff)}</b></span><span>Snapshot <code>${esc(r.scope.snapshot_id.slice(0,16))}</code></span></div>
      ${timeline(r.scope.config)}
      <article class="audit-panel"><div class="audit-heading"><span class="audit-number">02</span><div><h3>What did we actually verify?</h3><p>${esc(r.timing.prior_sessions)} earlier sessions compared; ${esc(r.timing.future_sessions)} later sessions challenged. One cutoff, fixed universe, fixed settings.</p></div></div><div class="audit-check-grid">${checkCards(r)}</div><p class="audit-caption">Tolerance: ${esc(r.scope.tolerance)}. No active earlier holdings, or too few observations, produces insufficient evidence rather than a pass.</p></article>
      <article class="audit-panel"><div class="audit-heading"><span class="audit-number">03</span><div><h3>How much do the assumptions matter?</h3><p>Each comparison changes one calculation or holding assumption. None replaces the displayed backtest.</p></div></div>
        <div class="audit-estimator-note">${pill(r.estimator.status,r.estimator.status==='issue'?'Numerical issue detected':'Estimator check passed')}<p>${r.estimator.status==='issue'?'Missing stock observations are not consistently matched to market observations in the production CAPM fit.':'The synthetic missing-observation check matches the expected zero residual.'} Matched-observation comparison: <strong>${esc(delta(paired.delta_cagr))} CAGR</strong>.</p></div>
        <div id="audit-comparison-chart" class="audit-chart" aria-label="CAGR comparison chart"></div>
        <div class="audit-table-wrap"><table><thead><tr><th scope="col">Comparison</th><th scope="col">CAGR</th><th scope="col">Change</th><th scope="col">Total return</th><th scope="col">Max drawdown</th></tr></thead><tbody>${r.comparisons.map(row=>`<tr><td><strong>${esc(row.label)}</strong><small>${esc(row.description)}</small></td><td>${esc(percent(row.metrics.cagr))}</td><td>${esc(delta(row.delta_cagr))}</td><td>${esc(percent(row.metrics.total_return))}</td><td>${esc(percent(row.metrics.maxdd))}</td></tr>`).join('')}</tbody></table></div>
      </article>
      <article class="audit-panel"><div class="audit-heading"><span class="audit-number">04</span><div><h3>Where did the gains come from?</h3><p>Concentration can explain exceptional outcomes without demonstrating a timing error.</p></div></div><div class="audit-stats">
        ${card('Average holdings',number(c.mean_holdings),`${c.min_holdings}–${c.max_holdings} stocks across sessions`)}${card('Largest stock allocation',percent(c.max_stock_weight),'Maximum target weight at any time')}${card('Five holdings or fewer',percent(c.fraction_5_or_fewer),'Share of all evaluation sessions')}</div>
        <div id="audit-attribution-chart" class="audit-chart attribution" aria-label="Additive wealth attribution chart"></div><p class="audit-caption">Additive contribution per $1 of starting capital, including compounding. Other stocks and trading costs reconcile to the original total return. This is attribution, not risk share.</p></article>
      <article class="audit-panel audit-limits"><div class="audit-heading"><span class="audit-number">05</span><div><h3>What this audit cannot establish</h3><p>A causal signal is one part of a credible backtest.</p></div></div><div class="audit-limit-grid">${r.limitations.map(item=>`<div>${pill(item.status)}<h4>${esc(item.title)}</h4><p>${esc(item.description)}</p></div>`).join('')}</div></article>
      <details class="audit-evidence"><summary>Method, scope & numerical evidence</summary><p>${esc(r.timing.note)}</p><p>Computed ${esc(r.computed_at)} · deterministic seed ${esc(r.scope.seed)} · audit version ${esc(r.version)}.</p><p>Perfect-linear-stock test: expected residual ${esc(r.estimator.expected_residual)}; original maximum absolute residual ${esc(r.estimator.original_max_abs_residual.toExponential(3))}; matched-observation residual ${esc(r.estimator.paired_max_abs_residual.toExponential(3))}.</p><p>Wealth-attribution reconciliation error: ${esc(r.attribution_reconciliation_error.toExponential(3))}. The JSON download includes full completed-run settings and unrounded measurements.</p></details>`;
    const small=window.innerWidth<600;
    Plotly.newPlot(document.querySelector('#audit-comparison-chart'),[{type:'bar',orientation:'h',y:r.comparisons.map((_,i)=>String(i)),x:r.comparisons.map(row=>row.metrics.cagr==null?null:row.metrics.cagr*100),
      text:r.comparisons.map(row=>percent(row.metrics.cagr)),textposition:'outside',cliponaxis:false,marker:{color:['#36d6ba','#669cff','#f5ba62','#94a6bf']},
      customdata:r.comparisons.map(row=>[row.label,delta(row.delta_cagr)]),hovertemplate:'%{customdata[0]}<br>CAGR %{x:.2f}%<br>Change %{customdata[1]}<extra></extra>'}],
      {...layout,height:small?300:270,margin:{l:small?115:230,r:75,t:20,b:45},xaxis:{title:'CAGR (%)',gridcolor:'#203047',zerolinecolor:'#53647a',rangemode:'tozero'},yaxis:{autorange:'reversed',tickvals:['0','1','2','3'],ticktext:small?['Original','Hold shares','Matched CAPM','Zero risk-free']:r.comparisons.map(row=>row.label),automargin:true}},options);
    const rows=r.attribution;
    Plotly.newPlot(document.querySelector('#audit-attribution-chart'),[{type:'bar',orientation:'h',y:rows.map(row=>row.label),x:rows.map(row=>row.value),marker:{color:rows.map(row=>row.value>=0?'#669cff':'#f5ba62')},hovertemplate:'%{y}<br>$%{x:.4f} per initial $1<extra></extra>'}],
      {...layout,height:440,margin:{l:95,r:30,t:15,b:50},xaxis:{title:'Contribution per initial $1',gridcolor:'#203047',zerolinecolor:'#53647a',rangemode:'tozero',tickprefix:'$'},yaxis:{autorange:'reversed',automargin:true}},options);
  }
  function download(report){
    if(!report)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='audit-'+report.scope.snapshot_id.slice(0,12)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function resize(){
    const plot=document.querySelector('#audit-comparison-chart');if(!plot?.data)return;
    const small=window.innerWidth<600;
    Plotly.relayout(plot,{'margin.l':small?110:230,'margin.r':small?50:75,'yaxis.ticktext':small?['Original','Hold shares','Matched CAPM','Zero risk-free']:plot.data[0].customdata.map(row=>row[0])});
  }
  return {render,download,resize};
})();
