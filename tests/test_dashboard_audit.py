"""Audit jobs retain the completed run's snapshot and never expose private objects."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.dashboard.app import create_app
from src.engine import Snapshot
from test_engine import fixture


class DashboardAuditTests(unittest.TestCase):
    def wait(self,client,job):
        for _ in range(300):
            state=client.get('/api/jobs/'+job['id']).json()
            if state['state'] in ('succeeded','failed'):break
            time.sleep(.02)
        self.assertEqual(state['state'],'succeeded',state)
        return state

    def test_audit_lifecycle_cached_protected_and_private(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'snapshot';source.mkdir()
            p,cfg=fixture();p.to_csv(source/'prices.csv')
            (source/'run_metadata.json').write_text(json.dumps({'config':cfg,'universe_used':['AAA','BBB']}))
            app=create_app(source,Path(folder))
            with TestClient(app,base_url='http://127.0.0.1:8000') as client:
                self.assertEqual(client.post('/api/runs/missing/audit',json={}).status_code,404)
                run=client.post('/api/runs',json={}).json();self.wait(client,run)
                original_id=client.get('/api/bootstrap').json()['snapshot']['id']
                refreshed=Snapshot('refreshed',p.copy(),['AAA','BBB'],{**cfg,'momentum_window':12},{'source':str(source)})
                with patch('src.dashboard.jobs.refresh_snapshot',return_value=refreshed):
                    self.wait(client,client.post('/api/refresh',json={}).json())
                self.assertEqual(client.get('/api/bootstrap').json()['snapshot']['id'],'refreshed')
                url='/api/runs/'+run['id']+'/audit'
                self.assertEqual(client.get(url).status_code,409)
                self.assertEqual(client.post(url,json={},headers={'Origin':'https://evil.example'}).status_code,403)
                # The audit's completion must not immediately evict its old parent run.
                manager=app.state.jobs
                for n in range(8-len(manager.jobs)):
                    manager.jobs['filler-'+str(n)]={'id':'filler-'+str(n),'state':'failed','key':'filler-'+str(n)}
                submitted=client.post(url,json={})
                self.assertEqual(submitted.status_code,202)
                job=submitted.json();state=self.wait(client,job)
                self.assertNotIn('snapshot',state);self.assertNotIn('audit',state)
                response=client.get(url)
                self.assertEqual(response.status_code,200)
                report=response.json()
                self.assertEqual(report['scope']['snapshot_id'],original_id)
                self.assertEqual(report['scope']['config']['momentum_window'],10)
                self.assertNotIn('NaN',response.text)
                self.assertEqual(client.post(url,json={}).json()['id'],job['id'])
                # Rereading modified disk prices would produce a different report identity.
                p.iloc[-1,0]*=2;p.to_csv(source/'prices.csv')
                self.assertEqual(client.get(url).json(),report)
                self.assertEqual(client.get('/api/runs/'+run['id']+'/results').status_code,200)
                self.assertIn('Audit',client.get('/').text)


if __name__=='__main__':unittest.main()
