"""Browser regression for inactive parameter validation (optional Playwright)."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
from threading import Thread
import time
import unittest
from test_engine import fixture


class InactiveGridTests(unittest.TestCase):
    def test_disabled_grid_cannot_block_run_and_reenabling_restores_validation(self):
        # Regression: ISSUE-001 — hidden invalid grid blocked ordinary backtests.
        # Found by /qa on 2026-10-08
        # Report: .gstack/qa-reports/qa-report-localhost-2026-10-08.md
        if importlib.util.find_spec('playwright') is None:
            self.skipTest('Optional browser regression requires Playwright')
        from playwright.sync_api import sync_playwright
        import uvicorn
        from src.dashboard.app import create_app
        with tempfile.TemporaryDirectory() as folder, sync_playwright() as browser_api:
            configured=os.environ.get('QA_CHROMIUM_PATH')
            binary=Path(configured) if configured else Path(browser_api.chromium.executable_path)
            if not binary.exists() and not configured and os.name=='nt':
                options=list((Path(os.environ['LOCALAPPDATA'])/'ms-playwright').glob('chromium-*/chrome-win64/chrome.exe'))
                if options:binary=max(options,key=lambda p:p.stat().st_mtime)
            if not binary.is_file():
                self.skipTest('Install Playwright Chromium or set QA_CHROMIUM_PATH')
            prices,cfg=fixture()
            source=Path(folder)/'snapshot';source.mkdir()
            prices.to_csv(source/'prices.csv')
            (source/'run_metadata.json').write_text(json.dumps({'config':cfg,'universe_used':['AAA','BBB']}))
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
            server=uvicorn.Server(uvicorn.Config(create_app(source,Path(folder)),host='127.0.0.1',port=port,log_level='error'))
            thread=Thread(target=server.run,daemon=True);thread.start()
            browser=None
            try:
                for _ in range(100):
                    if server.started:break
                    time.sleep(.02)
                browser=browser_api.chromium.launch(headless=True,executable_path=str(binary))
                page=browser.new_page(viewport={'width':1280,'height':900})
                errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{port}')
                page.wait_for_function("document.querySelector('#snapshot-badge').textContent.includes('Data through')")
                page.locator('[name=sensitivity_enabled]').check()
                page.locator('#grid-settings summary').click()
                page.locator('[name=grid_window_step]').fill('0')
                self.assertFalse(page.locator('#parameters').evaluate('form=>form.checkValidity()'))
                page.locator('[name=sensitivity_enabled]').uncheck()
                self.assertTrue(page.locator('#parameters').evaluate('form=>form.checkValidity()'))
                page.locator('[name=sensitivity_enabled]').check()
                self.assertFalse(page.locator('#parameters').evaluate('form=>form.checkValidity()'))
                page.locator('[name=sensitivity_enabled]').uncheck()
                with page.expect_response(lambda response:response.url.endswith('/api/runs') and response.request.method=='POST') as submitted:
                    page.locator('#run').click()
                self.assertEqual(submitted.value.status,202)
                page.wait_for_function("document.querySelector('#run-status').textContent==='Run complete'",timeout=20000)
                self.assertIn('10 sessions',page.locator('#result-context').inner_text())
                self.assertEqual(errors,[])
            finally:
                if browser:browser.close()
                server.should_exit=True
                thread.join(10)


if __name__=='__main__':
    unittest.main()
