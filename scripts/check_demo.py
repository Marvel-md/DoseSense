#!/usr/bin/env python3
"""Start a disposable local API and run the real JavaScript render harness.

Requires generated cohort/model and Node.js >=18. Does not modify feedback.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=180,
                        help='Maximum API startup time in seconds')
    args = parser.parse_args()
    if not shutil.which('node'):
        parser.error('Node.js 18 or newer is required for the render harness.')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base_url = f'http://127.0.0.1:{port}'
    # Bypass environment proxies for this loopback-only readiness request.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix='dosesense-demo-') as tmp:
        env = dict(os.environ, DOSESENSE_DB=str(Path(tmp) / 'feedback.sqlite3'),
                   DOSESENSE_BASE_URL=base_url)
        log_path = Path(tmp) / 'api.log'
        with log_path.open('w+') as log:
            proc = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'backend.app.main:app',
                 '--host', '127.0.0.1', '--port', str(port)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + args.timeout
                last_error = 'API has not responded'
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f'API exited with code {proc.returncode}')
                    try:
                        with opener.open(base_url + '/api/health', timeout=2) as response:
                            health = json.load(response)
                        if health.get('status') == 'ok':
                            print(f"Ready: {health.get('patients_loaded')} patients", flush=True)
                            break
                        last_error = str(health)
                    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                        last_error = str(exc)
                    time.sleep(0.25)
                else:
                    raise RuntimeError(f'API readiness timeout: {last_error}')
                result = subprocess.run(['node', 'tests/render_harness.js'],
                                        cwd=ROOT, env=env, timeout=180)
                return result.returncode
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(str(exc), file=sys.stderr)
                log.flush()
                print(log_path.read_text()[-6000:], file=sys.stderr)
                return 1
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    raise SystemExit(main())
