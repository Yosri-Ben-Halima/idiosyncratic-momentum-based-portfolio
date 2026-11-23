"""Hosted mode: a DASHBOARD_PASSWORD secret replaces the localhost-only guard with HTTP Basic auth."""
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.dashboard.app import create_app

HOST = 'https://residual-lab.onrender.com'


def basic(user, password):
    return {'Authorization': 'Basic '+base64.b64encode(f'{user}:{password}'.encode()).decode()}


class RemoteAccessTests(unittest.TestCase):
    def client(self, folder, **env):
        with patch.dict('os.environ', env, clear=False):
            return TestClient(create_app(None, Path(folder)), base_url=HOST)

    def test_password_required_on_public_host(self):
        with tempfile.TemporaryDirectory() as folder, \
                self.client(folder, DASHBOARD_PASSWORD='s3cret', DASHBOARD_USER='quant') as client:
            missing = client.get('/api/bootstrap')
            self.assertEqual(missing.status_code, 401)
            self.assertIn('Basic', missing.headers['www-authenticate'])
            self.assertEqual(client.get('/api/bootstrap', headers=basic('quant', 'wrong')).status_code, 401)
            self.assertEqual(client.get('/api/bootstrap', headers=basic('quant', 's3cret')).status_code, 200)

    def test_cross_site_post_still_blocked_with_password(self):
        with tempfile.TemporaryDirectory() as folder, \
                self.client(folder, DASHBOARD_PASSWORD='s3cret') as client:
            post = client.post('/api/refresh', json={'top_k': 0},
                               headers={**basic('admin', 's3cret'), 'Origin': 'https://evil.example'})
            self.assertEqual(post.status_code, 403)
            same = client.post('/api/refresh', json={'top_k': 0}, headers={**basic('admin', 's3cret'), 'Origin': HOST})
            self.assertEqual(same.status_code, 422)  # past the guards, rejected only for the bad setting

    def test_without_password_public_host_is_refused(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict('os.environ', {}, clear=False) as env:
            env.pop('DASHBOARD_PASSWORD', None)
            with TestClient(create_app(None, Path(folder)), base_url=HOST) as client:
                self.assertEqual(client.get('/api/bootstrap').status_code, 403)
