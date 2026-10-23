"""Bounded single-worker computation queue; results keyed by all inputs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import RLock
import time
import uuid
from src.engine import run_backtest
from src.engine.exports import export_result
from src.engine.snapshots import refresh_snapshot
from src.engine.audit import run_audit
from .charts import build_dashboard_payload


class JobManager:
    def __init__(self,root,on_refresh=None):
        self.root = Path(root)
        self.on_refresh = on_refresh
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='backtest')
        self.lock = RLock()
        self.jobs = {}
        self.cache = {}

    def submit(self,kind,config,snapshot=None,parent=None):
        key = hashlib.sha256(json.dumps([kind,config,snapshot.id if snapshot else None,parent],sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                cached = self.jobs.get(self.cache[key])
                if cached and cached['state']!='failed' and not (kind=='refresh' and cached['state']=='succeeded'):
                    return self.status(cached['id'])
            if sum(j['state'] in ('queued','running') for j in self.jobs.values())>=3:
                raise ValueError('The computation queue is full; wait for a running job to finish')
            identifier = uuid.uuid4().hex
            self.jobs[identifier] = {'id':identifier,'kind':kind,'state':'queued','stage':'Waiting for computation worker',
                'created':time.time(),'elapsed':0,'key':key,'config':deepcopy(config),'parent':parent}
            self.cache[key] = identifier
            self.pool.submit(self._execute,identifier,snapshot)
            return self.status(identifier)

    def _execute(self,identifier,snapshot):
        with self.lock:
            job = self.jobs[identifier]
            job.update(state='running',started=time.time())
        def progress(message):
            with self.lock:
                job['stage'] = message
        try:
            if job['kind']=='run':
                result = run_backtest(snapshot,job['config'],progress)
                progress('Building interactive dashboard')
                payload = build_dashboard_payload(result)
                with self.lock:
                    job.update(result=result,payload=payload,snapshot=snapshot)
            elif job['kind']=='refresh':
                fresh = refresh_snapshot(job['config'],self.root,progress)
                if self.on_refresh:
                    self.on_refresh(fresh)
            elif job['kind']=='export':
                progress('Rendering PNG tearsheets and packaging results')
                original = self.result(job['parent'])
                files = export_result(original,self.root/'dashboard'/'runs'/job['parent'])
                with self.lock:
                    self.jobs[job['parent']]['files'] = files
                    job['files'] = list(files)
            elif job['kind']=='audit':
                original = self.result(job['parent'])
                with self.lock:
                    frozen = self.jobs[job['parent']]['snapshot']
                audit = run_audit(frozen,original,progress)
                with self.lock:
                    self.jobs[job['parent']]['audit'] = audit
            else:
                raise ValueError('Unknown job type')
            with self.lock:
                job.update(state='succeeded',stage='Complete',elapsed=time.time()-job['started'])
        except Exception as error:
            with self.lock:
                job.update(state='failed',stage='Computation failed',error=str(error),elapsed=time.time()-job['started'])
        finally:
            with self.lock:
                completed = [j for j in self.jobs.values() if j['state'] in ('succeeded','failed')]
                protected = {j['parent'] for j in self.jobs.values() if j['state'] in ('queued','running')}
                # Keep the just-produced child artifact readable by the next request.
                protected.add(job.get('parent'))
                for old in completed[:-8]:
                    if old['id'] in protected:
                        continue
                    self.jobs.pop(old['id'],None)
                    if self.cache.get(old['key'])==old['id']:
                        self.cache.pop(old['key'],None)

    def status(self,identifier):
        with self.lock:
            job = self.jobs[identifier]
            safe = {k:deepcopy(v) for k,v in job.items() if k not in ('key','result','payload','files','snapshot','audit')}
            if job['state']=='running':
                safe['elapsed'] = time.time()-job['started']
            if isinstance(job.get('files'),list):
                safe['files'] = job['files']
            return safe

    def result(self,identifier):
        with self.lock:
            job = self.jobs[identifier]
            if job['state']!='succeeded' or 'result' not in job:
                raise ValueError('This run has no completed results')
            return job['result']

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
