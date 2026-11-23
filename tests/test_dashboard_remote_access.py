"""Hosted mode: public=True accepts any host name; cross-site writes stay blocked."""
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from src.dashboard.app import create_app

HOST = 'https://residual-lab.onrender.com'


class RemoteAccessTests(unittest.TestCase):
    def test_public_app_serves_its_host_without_sign_in(self):
        with tempfile.TemporaryDirectory() as folder, \
                TestClient(create_app(None, Path(folder), public=True), base_url=HOST) as client:
            self.assertEqual(client.get('/api/bootstrap').status_code, 200)
            same = client.post('/api/refresh', json={'top_k': 0}, headers={'Origin': HOST})
            self.assertEqual(same.status_code, 422)  # past the guards, rejected only for the bad setting

    def test_public_app_still_blocks_cross_site_posts(self):
        with tempfile.TemporaryDirectory() as folder, \
                TestClient(create_app(None, Path(folder), public=True), base_url=HOST) as client:
            self.assertEqual(client.post('/api/refresh', json={}, headers={'Origin': 'https://evil.example'}).status_code, 403)

    def test_local_app_refuses_other_hosts(self):
        with tempfile.TemporaryDirectory() as folder, \
                TestClient(create_app(None, Path(folder)), base_url=HOST) as client:
            self.assertEqual(client.get('/api/bootstrap').status_code, 403)
