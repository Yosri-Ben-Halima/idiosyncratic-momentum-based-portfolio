'use strict';
const $ = selector => document.querySelector(selector);
const form = $('#parameters');
const state = {bootstrap:null, result:null, runId:null, tab:'performance', active:false, completedFingerprint:null, completedSnapshot:null};
const escapeText = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => value == null ? 'N/A' : `${(value*100).toLocaleString(undefined,{maximumFractionDigits:1})}%`;
const ratio = value => value == null ? 'N/A' : value.toLocaleString(undefined,{maximumFractionDigits:2,minimumFractionDigits:2});
const plotOptions = {responsive:true,displaylogo:false,modeBarButtonsToRemove:['sendDataToCloud','select2d','lasso2d']};
const primary = [
  ['momentum_window','Momentum window','range',1,'sessions','Controls CAPM estimation and residual summation.'],
  ['z_threshold','Z-score cutoff','range',.05,'z-score','Stocks must exceed this cross-sectional threshold.'],
  ['momentum_skip','Skip period','number',1,'sessions'],
  ['top_k','Maximum holdings','range',1,'stocks','A cap: actual holdings depend on the cutoff.'],
  ['rebalance_freq','Rebalance','select',null,null,null,[['W-FRI','Weekly · Friday'],['ME','Month end'],['QE','Quarter end']]]
];
const advanced = [
  ['min_history','Minimum price history','number',1,'sessions'],
  ['transaction_cost_bps','Trading cost','number',.5,'basis points'],
  ['rf_annual','Annual risk-free rate','number',.01,'%','Applied consistently across the historical backtest.'],
  ['risk_window','Risk covariance window','number',1,'sessions'],
  ['risk_min_observations','Minimum risk observations','number',1,'sessions'],
  ['monte_carlo_runs','Monte Carlo paths','preset',1,'paths',null,[['1000','1,000 · quick'],['10000','10,000 · standard'],['20000','20,000 · detailed'],['custom','Custom']]],
  ['monte_carlo_days','Simulation horizon','preset',1,'sessions',null,[['252','1 year · 252 sessions'],['504','2 years · 504 sessions'],['1260','5 years · 1,260 sessions'],['custom','Custom']]],
  ['monte_carlo_seed','Simulation seed','number',1,'seed','Same seed gives the same simulated paths.']
];
const grid = [
  ['grid_window_min','Window minimum','number',1,'sessions'],['grid_window_max','Window maximum','number',1,'sessions'],
  ['grid_window_step','Window step','number',1,'sessions'],['grid_cutoff_min','Cutoff minimum','number',.05,'z-score'],
  ['grid_cutoff_max','Cutoff maximum','number',.05,'z-score'],['grid_cutoff_step','Cutoff step','number',.05,'z-score'],
  ['split_date','Early / recent split','date',null,null,'Diagnostic split, not untouched validation.']
];

function buildControls(definitions,container){
  for(const [name,label,type,step,unit,help,options] of definitions){
    const card = document.createElement('div'); card.className='control';
    const lab = document.createElement('label'); lab.htmlFor=name; lab.textContent=label;card.append(lab);
    if(type==='select'){
      const select=document.createElement('select');select.id=select.name=name;
      for(const [value,text] of options){const option=new Option(text,value);select.add(option);}card.append(select);
    } else {
      const row=document.createElement('div');row.className='control-row';
      if(type==='range'){
        const slider=document.createElement('input');slider.type='range';slider.id=name+'-range';slider.setAttribute('aria-label',label+' slider');slider.step=step;
        slider.addEventListener('input',()=>{form.elements[name].value=slider.value;markPending();});row.append(slider);
      }
      const input=document.createElement('input');input.id=input.name=name;input.type=type==='date'?'date':'number';input.required=true;if(step)input.step=['z_threshold','transaction_cost_bps','rf_annual'].includes(name)?'any':step;
      if(type!=='range')input.style.width='100%';
      if(type==='preset'){
        const preset=document.createElement('select');preset.id=name+'-preset';preset.setAttribute('aria-label',label+' preset');
        for(const [value,text] of options)preset.add(new Option(text,value));
        preset.addEventListener('change',()=>{if(preset.value!=='custom')input.value=preset.value;input.hidden=preset.value!=='custom';markPending();});
        card.append(preset);input.hidden=true;input.style.marginTop='8px';
      }
      input.addEventListener('input',()=>{const slider=$(`#${name}-range`);if(slider)slider.value=input.value;const preset=$(`#${name}-preset`);if(preset){preset.value='custom';input.hidden=false;}});
      row.append(input);card.append(row);
    }
    if(unit){const hint=document.createElement('div');hint.className='range-hint';hint.innerHTML=`<span id="${name}-bounds"></span><span>${unit}</span>`;card.append(hint);}
    if(help){const detail=document.createElement('small');detail.className='help';detail.textContent=help;card.append(detail);}
    $(container).append(card);
  }
}
buildControls(primary,'#primary-controls');buildControls(advanced,'#advanced-controls');buildControls(grid,'#grid-controls');

function fillDefaults(){
  const cfg=state.bootstrap.defaults, s=cfg.sensitivity||{};
  for(const [key,value] of Object.entries(cfg))if(form.elements[key])form.elements[key].value=key==='rf_annual'?value*100:value;
  form.elements.sensitivity_enabled.checked=!!s.enabled;
  for(const key of ['risk_window','risk_min_observations','split_date'])form.elements[key].value=s[key]??({risk_window:252,risk_min_observations:126,split_date:'2023-01-01'}[key]);
  const windows=s.windows||[126,252,378],cutoffs=s.cutoffs||[1,2,3];
  const defaults={grid_window_min:Math.min(...windows),grid_window_max:Math.max(...windows),grid_window_step:42,grid_cutoff_min:Math.min(...cutoffs),grid_cutoff_max:Math.max(...cutoffs),grid_cutoff_step:.25};
  for(const [key,value] of Object.entries(defaults))form.elements[key].value=value;
  for(const [key,bounds] of Object.entries(state.bootstrap.limits)){
    const input=form.elements[key];if(!input)continue;
    let [low,high]=bounds;if(key==='top_k'&&state.bootstrap.snapshot)high=state.bootstrap.snapshot.stocks;if(key==='rf_annual'){low*=100;high*=100;}
    input.min=low;input.max=high;
    const slider=$(`#${key}-range`);if(slider){slider.min=low;slider.max=high;slider.value=input.value;}
    const hint=$(`#${key}-bounds`);if(hint)hint.textContent=`${low.toLocaleString()} – ${high.toLocaleString()}`;
    const preset=$(`#${key}-preset`);if(preset){preset.value=[...preset.options].some(o=>o.value===String(input.value))?String(input.value):'custom';input.hidden=preset.value!=='custom';}
  }
  for(const key of ['risk_window','risk_min_observations']){form.elements[key].min=2;form.elements[key].max=756;}
  for(const key of ['grid_window_min','grid_window_max']){form.elements[key].min=2;form.elements[key].max=756;}
  form.elements.grid_window_step.min=1;form.elements.grid_window_step.max=756;
  for(const key of ['grid_cutoff_min','grid_cutoff_max']){form.elements[key].min=-3;form.elements[key].max=6;}
  form.elements.grid_cutoff_step.min=.05;form.elements.grid_cutoff_step.max=9;
  for(const [name] of grid)form.elements[name].required=form.elements.sensitivity_enabled.checked;
  markPending();
}
function axisValues(min,max,step){
  if(!Number.isFinite(min+max+step)||step<=0||max<min)throw Error('Sensitivity grid: maximum must follow minimum and steps must be positive.');
  if((max-min)/step>224)throw Error('Sensitivity axis has too many values. Increase its step.');
  return Array.from({length:Math.floor((max-min)/step+1e-9)+1},(_,i)=>Number((min+i*step).toFixed(8)));
}
function readConfig(){
  const cfg={};
  for(const [name,,,] of [...primary,...advanced]){
    if(name.startsWith('risk_'))continue;
    cfg[name]=name==='rebalance_freq'?form.elements[name].value:Number(form.elements[name].value);
  }
  cfg.rf_annual/=100;cfg.start_date=form.elements.start_date.value;cfg.end_date=form.elements.end_date.value;
  const s=cfg.sensitivity={enabled:form.elements.sensitivity_enabled.checked,risk_window:Number(form.elements.risk_window.value),risk_min_observations:Number(form.elements.risk_min_observations.value),split_date:form.elements.split_date.value};
  if(s.enabled){
    s.windows=axisValues(...['min','max','step'].map(key=>Number(form.elements['grid_window_'+key].value)));
    s.cutoffs=axisValues(...['min','max','step'].map(key=>Number(form.elements['grid_cutoff_'+key].value)));
    const n=new Set([...s.windows,cfg.momentum_window]).size*new Set([...s.cutoffs,cfg.z_threshold]).size;
    if(n>225)throw Error(`Sensitivity grid has ${n} settings; the limit is 225.`);
  }
  return cfg;
}
function markPending(){
  const enabled=form.elements.sensitivity_enabled.checked;$('#grid-settings').hidden=!enabled;
  for(const [name] of grid){form.elements[name].required=enabled;form.elements[name].disabled=!enabled;}
  let fingerprint;try{fingerprint=JSON.stringify(readConfig());}catch(e){fingerprint='invalid';}
  const changed=state.result&&(fingerprint!==state.completedFingerprint||state.bootstrap.snapshot?.id!==state.completedSnapshot);
  $('#pending').textContent=changed?'● Changes not run':state.result?'✓ Displaying these settings':'Ready to run';$('#pending').classList.toggle('changed',!!changed);
  if(enabled){try{const cfg=readConfig();const n=new Set([...cfg.sensitivity.windows,cfg.momentum_window]).size*new Set([...cfg.sensitivity.cutoffs,cfg.z_threshold]).size;$('#grid-preview').textContent=`${n} tested combinations, including the current setting. Windows: ${cfg.sensitivity.windows.join(', ')}. Cutoffs: ${cfg.sensitivity.cutoffs.join(', ')}.`;}catch(e){$('#grid-preview').textContent=e.message;}}
}
form.addEventListener('input',markPending);form.addEventListener('change',markPending);
function notice(text,info=false){$('#message').textContent=text;$('#message').hidden=!text;$('#message').classList.toggle('info',info);}
async function api(path,body){
  const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;
}
function setBusy(value){state.active=value;$('#progress').hidden=!value;for(const id of ['run','refresh','empty-run'])$('#'+id).disabled=value;$('#export').disabled=value||!state.result;$('#run-status').classList.toggle('busy',value);$('#run-status').textContent=value?'Computing':state.result?'Run complete':'Ready';}
async function waitForJob(job){
  for(;;){
    $('#stage').textContent=job.stage;$('#elapsed').textContent=`${Math.round(job.elapsed||0)}s`;
    if(job.state==='succeeded')return job;
    if(job.state==='failed')throw Error(job.error||'Computation failed.');
    await new Promise(resolve=>setTimeout(resolve,900));
    job=await api('/api/jobs/'+job.id);
  }
}
async function run(){
  if(state.active||!form.reportValidity())return;
  let cfg;try{cfg=readConfig();}catch(e){notice(e.message);return;}
  const fingerprint=JSON.stringify(cfg),snapshot=state.bootstrap.snapshot?.id;
  notice('');setBusy(true);
  try{
    const job=await api('/api/runs',cfg);await waitForJob(job);
    const result=await api('/api/runs/'+job.id+'/results');
    state.result=result;state.runId=job.id;state.completedFingerprint=fingerprint;state.completedSnapshot=snapshot;
    $('#downloads').hidden=true;renderResult();
    if(result.warnings.length)notice(result.warnings.join(' '),true);
  }catch(e){notice(e.message);}finally{setBusy(false);markPending();}
}
form.addEventListener('submit',event=>{event.preventDefault();run();});$('#empty-run').addEventListener('click',run);
$('#reset').addEventListener('click',()=>{fillDefaults();notice('');});
function syncSidebar(){const hidden=window.matchMedia('(max-width:850px)').matches&&!$('#sidebar').classList.contains('visible');$('#sidebar').inert=hidden;$('#sidebar').setAttribute('aria-hidden',String(hidden));$('#controls-toggle').setAttribute('aria-expanded',String(!hidden));}
$('#enable-grid').addEventListener('click',()=>{form.elements.sensitivity_enabled.checked=true;$('#grid-settings').open=true;markPending();if(window.matchMedia('(max-width:850px)').matches)$('#sidebar').classList.add('visible');syncSidebar();});
$('#controls-toggle').addEventListener('click',()=>{$('#sidebar').classList.toggle('visible');syncSidebar();});
document.addEventListener('keydown',event=>{if(event.key==='Escape'){$('#sidebar').classList.remove('visible');syncSidebar();}});
$('#refresh').addEventListener('click',async()=>{
  if(state.active||!form.reportValidity())return;
  try{const cfg=readConfig();setBusy(true);notice('');await waitForJob(await api('/api/refresh',cfg));await bootstrap(false);notice('New price snapshot saved. Run again to use it; existing results keep their original data.',true);}catch(e){notice(e.message);}finally{setBusy(false);markPending();}
});
$('#export').addEventListener('click',async()=>{
  const id=state.runId;if(!id||state.active)return;
  try{setBusy(true);notice('');const job=await waitForJob(await api('/api/runs/'+id+'/exports',{}));const links=$('#download-links');links.replaceChildren();for(const name of job.files.filter(x=>x.endsWith('.png')||x.endsWith('.zip'))){const a=document.createElement('a');a.className='button secondary';a.href='/api/runs/'+id+'/files/'+encodeURIComponent(name);a.textContent=name==='results.zip'?'↓ All results · ZIP':'↓ '+name.replace('tearsheet_','').replace('.png','').replaceAll('_',' ');a.download=name;links.append(a);}$('#downloads').hidden=false;$('#downloads').scrollIntoView({behavior:'smooth',block:'nearest'});}catch(e){notice(e.message);}finally{setBusy(false);}
});
function renderKpis(){
  const metrics=state.result?.metrics;
  $('#kpis').innerHTML=[['Total return','Total Return'],['CAGR','CAGR'],['Sharpe','Sharpe'],['Max drawdown','Max DD']].map(([label,key])=>{
    const format=key==='Sharpe'?ratio:pct;
    return `<article class="kpi"><p class="kpi-label">${label}</p><p class="kpi-value">${metrics?format(metrics.Strategy[key]):'—'}</p><p class="kpi-sub">vs benchmark <b>${metrics?format(metrics.SP500[key]):'—'}</b></p></article>`;
  }).join('');
}
function renderResult(){
  renderKpis();const r=state.result;$('#empty').hidden=true;$('#run-details').hidden=false;
  $('#result-context').textContent=`Completed: ${r.dates[0]} → ${r.dates[1]} · ${r.config.momentum_window} sessions / z > ${r.config.z_threshold} · ${r.observations.toLocaleString()} observations`;
  const detail=$('#run-detail-content');detail.replaceChildren();
  for(const text of [`Snapshot ${r.snapshot_id.slice(0,16)} · requested end ${r.config.end_date} (exclusive)`,
    `Maximum ${r.config.top_k} holdings · ${r.config.rebalance_freq} · ${r.config.transaction_cost_bps} bps costs · ${pct(r.config.rf_annual)} annual risk-free rate`,
    `Risk covariance: preceding ${r.config.sensitivity.risk_window} sessions, minimum ${r.config.sensitivity.risk_min_observations} observations. Latest risk snapshot: ${r.risk_snapshot_date||'unavailable'}.`,
    `Monte Carlo: ${r.config.monte_carlo_runs.toLocaleString()} paths × ${r.config.monte_carlo_days} sessions · seed ${r.config.monte_carlo_seed}. Fan preserves the tearsheet's additive presentation.`]){const p=document.createElement('p');p.textContent=text;detail.append(p);}
  renderTab();
}
function table(title,headers,rows){return `<article class="table-card"><h2>${escapeText(title)}</h2><table><thead><tr>${headers.map(x=>`<th scope="col">${escapeText(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((x,i)=>`<td class="${i===1?'strategy-cell':i===2?'benchmark-cell':''}">${escapeText(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></article>`;}
function renderTab(){
  const r=state.result;$('#dashboard').hidden=!r||state.tab==='sensitivity';$('#sensitivity').hidden=state.tab!=='sensitivity';$('#empty').hidden=!!r||state.tab==='sensitivity';
  if(state.tab==='sensitivity'){renderSensitivity();return;}if(!r)return;
  const charts=$('#charts');for(const el of charts.querySelectorAll('.chart'))Plotly.purge(el);charts.replaceChildren();
  for(const data of r.figures[state.tab]){
    const card=document.createElement('article');card.className='chart-card'+(data.wide?' wide':'');
    const header=document.createElement('div');header.className='card-header';const title=document.createElement('h2');title.textContent=data.title;header.append(title);card.append(header);
    const plot=document.createElement('div');plot.className='chart';card.append(plot);charts.append(card);
    Plotly.newPlot(plot,data.plot.data,{...data.plot.layout,height:Math.min(340,plot.clientHeight||340)},plotOptions);
  }
  const tables=$('#tables');tables.innerHTML='';
  if(state.tab==='performance'){
    const ratios=['Sharpe','Sortino','Calmar','Beta','IR'];
    const rows=Object.keys(r.metrics.Strategy).filter(k=>k!=='label').map(key=>[key,(ratios.includes(key)?ratio:pct)(r.metrics.Strategy[key]),(ratios.includes(key)?ratio:pct)(r.metrics.SP500[key])]);
    rows.push(['DD / CAGR · years',ratio(r.recovery_years.Strategy),ratio(r.recovery_years.SP500)]);
    rows.push(['DD / CAGR · months',ratio(r.recovery_years.Strategy==null?null:r.recovery_years.Strategy*12),ratio(r.recovery_years.SP500==null?null:r.recovery_years.SP500*12)]);
    tables.innerHTML=table('Performance comparison',['Metric','Strategy','S&P 500'],rows);
  }else if(state.tab==='portfolio'){
    tables.innerHTML=table(`Latest holdings · ${r.holdings.length} stocks · ${ratio(r.mean_holdings)} average`,['Ticker','Capital weight','Absolute risk share'],r.holdings.length?r.holdings.map(row=>[row.ticker,pct(row.weight),pct(row.risk_share)]):[['Cash','100%','N/A']]);
  }
}
for(const button of document.querySelectorAll('[data-tab]'))button.addEventListener('click',()=>{state.tab=button.dataset.tab;for(const el of document.querySelectorAll('[data-tab]'))el.setAttribute('aria-selected',String(el===button));renderTab();});
function surfacePlot(id,key,small=false){
  const s=state.result.surfaces,m=s.metrics[key];const wi=s.windows.indexOf(s.baseline.window),ci=s.cutoffs.indexOf(s.baseline.cutoff);
  const unit=m.unit?` (${m.unit})`:'';const color='#96abc6';
  const traces=[{type:'surface',x:s.cutoffs,y:s.windows,z:m.z,colorscale:'Viridis',colorbar:{title:{text:m.unit||m.label,side:'right'},thickness:10,len:.65},
    hovertemplate:'Cutoff %{x}<br>Window %{y} sessions<br>%{z:.3f}'+(m.unit||'')+'<extra></extra>'},
    {type:'scatter3d',x:[s.baseline.cutoff],y:[s.baseline.window],z:[m.z[wi][ci]],mode:'markers',marker:{size:5,color:'#f5ba62',line:{color:'#102030',width:1}},name:'Current settings',hovertemplate:'Current setting<br>%{z:.3f}'+(m.unit||'')+'<extra></extra>'}];
  const axis=title=>({title:{text:title,font:{size:11}},gridcolor:'#293a51',zerolinecolor:'#42516a',color,backgroundcolor:'#101c2d',showbackground:true,tickfont:{size:10}});
  Plotly.react($('#'+id),traces,{paper_bgcolor:'rgba(0,0,0,0)',font:{family:'Segoe UI, sans-serif',color},margin:{l:10,r:10,b:5,t:5},
    scene:{xaxis:axis('Z-score cutoff'),yaxis:axis('Momentum window'),zaxis:axis(m.label+unit),camera:{eye:{x:1.55,y:-1.65,z:1.05}}},
    showlegend:false,uirevision:'keep-camera',height:$('#'+id).clientHeight},plotOptions);
}
function updateMainSurface(){
  if(!state.result?.surfaces)return;
  const metric=$('#surface-metric').value;const period=$('#surface-period');const fullOnly=['mean_holdings','risk_diversification_ratio','risk_top_share','risk_effective_contributors'].includes(metric);
  period.disabled=fullOnly;if(fullOnly)period.value='full';
  surfacePlot('main-surface',fullOnly?metric:period.value+'_'+metric);
}
function renderSensitivity(){
  const s=state.result?.surfaces;$('#sensitivity-empty').hidden=!!s;$('#surface-content').hidden=!s;if(!s)return;
  $('#surface-subtitle').textContent=`${s.windows.length*s.cutoffs.length} measured settings · ${state.result.dates.join(' → ')} · unsmoothed surface`;
  updateMainSurface();surfacePlot('risk-surface','risk_top_share',true);
  Plotly.react($('#holdings-heatmap'),[{type:'heatmap',x:s.cutoffs,y:s.windows,z:s.metrics.mean_holdings.z,colorscale:'Viridis',
    colorbar:{title:'Stocks',thickness:10},hovertemplate:'Cutoff %{x}<br>Window %{y}<br>%{z:.2f} stocks<extra></extra>'}],
    {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#94a6bf',size:11},margin:{l:55,r:20,t:20,b:45},height:340,
      xaxis:{title:'Z-score cutoff'},yaxis:{title:'Momentum window'}},plotOptions);
  const sum=s.summary;$('#stability').innerHTML=`<strong>Local stability</strong> · Neighboring Sharpe range ${ratio(sum.local_sharpe_range[0])} – ${ratio(sum.local_sharpe_range[1])}. Local standard deviation ${ratio(sum.local_sharpe_std)}. ${sum.interior_local_maxima.length} interior local maxima.<small>Early / recent split: ${escapeText(s.split_date)}. Risk and holdings surfaces are full-period averages. Period splits reuse the historical sample and are not untouched validation.</small>`;
}
$('#surface-metric').addEventListener('change',updateMainSurface);$('#surface-period').addEventListener('change',updateMainSurface);
let resizeTimer;
window.addEventListener('resize',()=>{syncSidebar();clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{for(const el of document.querySelectorAll('.js-plotly-plot')){if(el.clientWidth&&el.clientHeight)Plotly.relayout(el,{width:el.clientWidth,height:el.clientHeight});}},120);});
async function bootstrap(reset=true){
  const b=await api('/api/bootstrap');state.bootstrap=b;const snap=b.snapshot;
  $('#snapshot-badge').textContent=snap?'Data through '+snap.last_price:'No saved data';
  $('#source-info').textContent=snap?`${snap.stocks} stocks · ${snap.first_price} → ${snap.last_price}${snap.excluded.length?' · '+snap.excluded.length+' excluded':''}`:'No usable saved snapshot. Refresh to download adjusted prices.';
  if(reset)fillDefaults();markPending();
}
syncSidebar();renderKpis();bootstrap().catch(e=>notice('Could not connect to the local engine: '+e.message+'. Reload to retry.'));
