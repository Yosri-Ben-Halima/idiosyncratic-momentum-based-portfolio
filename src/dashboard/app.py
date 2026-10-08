"""Same-origin, loopback-only local application and computation API."""
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit
import yaml
import plotly
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.engine import normalize_config
from src.engine.config import LIMITS
from src.engine.snapshots import discover_snapshot
from .jobs import JobManager

STATIC = Path(__file__).parent/'static'


def create_app(source=None,results_root=Path('results')):
    snapshot = discover_snapshot(results_root,source)
    state = {'snapshot':snapshot}
    manager = JobManager(results_root,lambda fresh:state.update(snapshot=fresh))
    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()
    app = FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.jobs = manager
    @app.middleware('http')
    async def local_only(request,call_next):
        host = request.url.hostname
        if host not in ('localhost','127.0.0.1','::1'):
            return JSONResponse({'detail':'Only localhost requests are accepted'},403)
        origin = request.headers.get('origin')
        if request.method not in ('GET','HEAD','OPTIONS') and origin:
            parsed = urlsplit(origin)
            if parsed.netloc!=request.headers.get('host') or parsed.scheme!=request.url.scheme:
                return JSONResponse({'detail':'Cross-origin mutation is not allowed'},403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    def defaults():
        snap = state['snapshot']
        if snap:
            cfg = deepcopy(snap.config)
        else:
            cfg = yaml.safe_load((Path(__file__).parents[1]/'environment'/'config.yaml').read_text())
        cfg.pop('rf_daily',None)
        cfg.pop('cost_bps',None)
        cfg.setdefault('rf_annual',.04)
        cfg.setdefault('monte_carlo_seed',42)
        cfg.setdefault('sensitivity',{}).update(enabled=False)
        cfg['sensitivity'].setdefault('risk_window',252)
        cfg['sensitivity'].setdefault('risk_min_observations',126)
        cfg['sensitivity'].setdefault('split_date','2023-01-01')
        return cfg

    @app.get('/api/bootstrap')
    def bootstrap():
        snap = state['snapshot']
        return {'defaults':defaults(),'limits':LIMITS,'snapshot':None if snap is None else {
            'id':snap.id,'stocks':len(snap.tickers),'first_price':str(snap.prices.index[0].date()),
            'last_price':str(snap.prices.index[-1].date()),'source':Path(snap.metadata['source']).name,
            'excluded':snap.metadata.get('excluded_tickers',[])}}

    @app.post('/api/runs',status_code=202)
    def submit(values:dict=Body(...)):
        snap = state['snapshot']
        if snap is None:
            raise HTTPException(409,'No saved price snapshot. Refresh data first.')
        try:
            cfg = normalize_config(values,defaults(),snap)
            return manager.submit('run',cfg,snap)
        except (ValueError,TypeError) as error:
            raise HTTPException(422,str(error)) from error

    @app.post('/api/refresh',status_code=202)
    def refresh(values:dict=Body(...)):
        try:
            cfg = normalize_config(values,defaults())
            return manager.submit('refresh',cfg)
        except (ValueError,TypeError) as error:
            raise HTTPException(422,str(error)) from error

    @app.get('/api/jobs/{identifier}')
    def status(identifier:str):
        try:
            return manager.status(identifier)
        except KeyError:
            raise HTTPException(404,'Unknown or expired job') from None

    @app.get('/api/runs/{identifier}/results')
    def results(identifier:str):
        try:
            manager.result(identifier)
            return manager.jobs[identifier]['payload']
        except KeyError:
            raise HTTPException(404,'Unknown or expired run') from None
        except ValueError as error:
            raise HTTPException(409,str(error)) from error

    @app.post('/api/runs/{identifier}/exports',status_code=202)
    def exports(identifier:str):
        try:
            result = manager.result(identifier)
            return manager.submit('export',result.config,parent=identifier)
        except KeyError:
            raise HTTPException(404,'Unknown or expired run') from None
        except ValueError as error:
            raise HTTPException(409,str(error)) from error

    @app.get('/api/runs/{identifier}/files/{filename}')
    def download(identifier:str,filename:str):
        try:
            file = manager.jobs[identifier].get('files',{}).get(filename)
            if file is None or not file.is_file():
                raise KeyError()
            return FileResponse(file,filename=filename)
        except KeyError:
            raise HTTPException(404,'Unknown artifact; generate the PNG export first') from None

    @app.get('/plotly.js')
    def plotly_asset():
        return FileResponse(Path(plotly.__file__).parent/'package_data'/'plotly.min.js',media_type='application/javascript')

    @app.get('/')
    def index():
        return FileResponse(STATIC/'index.html')

    app.mount('/static',StaticFiles(directory=STATIC,check_dir=False),name='static')
    return app
