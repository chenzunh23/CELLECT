"""One loopback entry point for independent hsctiles browser processes.

Only explicitly registered loopback ports can be proxied. This module uses the
standard library and never imports the model, Torch, or any dataset loader.
"""
from __future__ import annotations

import argparse
import fcntl
import html
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid

DEFAULT_PORT = 8800
DEFAULT_REGISTRY = Path.home() / '.cache/cellect/browser_portal'
IDENTITY = 'cellect-browser-portal'
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate',
               'proxy-authorization', 'te', 'trailer', 'transfer-encoding', 'upgrade'}
# These are URL namespaces, not arbitrary filesystem paths in API metadata.
ROOT_URL = re.compile(r'''(["'`(])(/(?:api|pages|assets|image|tile_map)/)''')


def rewrite_urls(body: bytes, prefix: str) -> bytes:
    text = body.decode('utf-8')
    return ROOT_URL.sub(lambda m: m[1] + prefix + m[2], text).encode('utf-8')


def register_backend(port: int, label: str, registry: Path = DEFAULT_REGISTRY,
                     *, pid: int | None = None, checkpoint: str = '', root: str = '') -> str:
    if not 1 <= port <= 65535:
        raise ValueError('backend port must be between 1 and 65535')
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = uuid.uuid4().hex
    row = dict(port=port, label=label, checkpoint=checkpoint, root=root, pid=pid,
               token=token, registered_at=time.time())
    temporary = registry / f'.{port}.{token}.tmp'
    temporary.write_text(json.dumps(row))
    temporary.chmod(0o600)
    temporary.replace(registry / f'{port}.json')
    return token


def unregister_backend(port: int, token: str, registry: Path = DEFAULT_REGISTRY) -> None:
    path = registry / f'{port}.json'
    try:
        if json.loads(path.read_text()).get('token') == token:
            path.unlink()
    except (OSError, ValueError):
        pass


def backends(registry: Path) -> dict[int, dict]:
    result = {}
    for path in registry.glob('*.json'):
        try:
            row = json.loads(path.read_text())
            port = int(row['port'])
            if path.stem != str(port) or not 1 <= port <= 65535:
                continue
            if row.get('pid'):
                try:
                    os.kill(int(row['pid']), 0)
                except ProcessLookupError:
                    continue
            result[port] = row
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def is_portal(port: int) -> bool:
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=0.4)
    try:
        conn.request('GET', '/healthz')
        response = conn.getresponse()
        return response.status == 200 and json.loads(response.read(4096)).get('service') == IDENTITY
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def ensure_portal(port: int = DEFAULT_PORT, registry: Path = DEFAULT_REGISTRY) -> None:
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (registry / 'startup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if is_portal(port):
            return
        with (registry / 'portal.log').open('ab') as log:
            process = subprocess.Popen(
                [sys.executable, '-u', str(Path(__file__).resolve()), '--port', str(port),
                 '--registry', str(registry)], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(30):
            if is_portal(port):
                return
            if process.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f'portal could not listen on {port}; see {registry / "portal.log"}')


class PortalHandler(BaseHTTPRequestHandler):
    server_version = 'CellectBrowserPortal/1.0'

    def log_message(self, format, *args):
        print(f'{self.address_string()} - {format % args}', flush=True)

    def send_body(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_body(json.dumps(value).encode(), 'application/json; charset=utf-8', status)

    def index(self):
        rows = backends(self.server.registry)
        cards = []
        for port, row in sorted(rows.items()):
            label = html.escape(str(row.get('label') or f'Browser {port}'))
            checkpoint = html.escape(str(row.get('root') or row.get('checkpoint', '')))
            cards.append(f'<li><a href="/browser/{port}/">{label}</a>'
                         f'<div>port {port}</div><small>{checkpoint}</small></li>')
        page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CELLECT browsers</title>
<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f4f6fa;color:#182537}
ul{list-style:none;padding:0}li{background:white;border:1px solid #d8deea;border-radius:10px;padding:20px;margin:16px 0}
a{color:#175eae;font-size:20px}small{overflow-wrap:anywhere}li div{margin:8px 0;color:#586578}</style>
</head><body><h1>CELLECT browsers</h1><p>选择实验；不同实验可在不同标签页同时打开。</p><ul>'''
        page += ''.join(cards) or '<li>暂无已登记的 browser。请先启动 hsctiles browser。</li>'
        page += '</ul><p><a href="/">刷新列表</a></p></body></html>'
        self.send_body(page.encode(), 'text/html; charset=utf-8')

    def proxy(self):
        parsed = urlsplit(self.path)
        match = re.fullmatch(r'/browser/(\d+)(/.*)?', parsed.path)
        if not match:
            self.send_json({'error': 'not found'}, 404)
            return
        port = int(match[1])
        if port == self.server.server_port or port not in backends(self.server.registry):
            self.send_json({'error': 'browser is not registered; return to the portal index'}, 404)
            return
        prefix = f'/browser/{port}'
        if match[2] is None:
            self.send_response(307)
            self.send_header('Location', prefix + '/' + ('?' + parsed.query if parsed.query else ''))
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        path = match[2] + ('?' + parsed.query if parsed.query else '')
        if self.headers.get('Transfer-Encoding'):
            self.send_json({'error': 'chunked requests are not supported'}, 400)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= 16 * 1024 * 1024:
                raise ValueError
        except ValueError:
            self.send_json({'error': 'invalid or excessive request size'}, 413)
            return
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in HOP_HEADERS | {'host', 'accept-encoding'}}
        headers['Accept-Encoding'] = 'identity'
        body = self.rfile.read(length) if length else None
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=self.server.request_timeout)
        sent = False
        try:
            conn.request(self.command, path, body=body, headers=headers)
            response = conn.getresponse()
            content_type = response.getheader('Content-Type', 'application/octet-stream')
            textual = any(x in content_type for x in ('text/', 'json', 'javascript'))
            translated = rewrite_urls(response.read(), prefix) if textual and self.command != 'HEAD' else None
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in HOP_HEADERS | {'content-length', 'server', 'date', 'cache-control'}:
                    continue
                if key.lower() == 'location' and value.startswith('/') and not value.startswith('//'):
                    value = prefix + value
                self.send_header(key, value)
            self.send_header('Cache-Control', 'no-store')
            if translated is not None:
                self.send_header('Content-Length', str(len(translated)))
            elif response.getheader('Content-Length'):
                self.send_header('Content-Length', response.getheader('Content-Length'))
            self.end_headers()
            sent = True
            if self.command != 'HEAD':
                if translated is not None:
                    self.wfile.write(translated)
                else:
                    while chunk := response.read(65536):
                        self.wfile.write(chunk)
        except (OSError, http.client.HTTPException, UnicodeError) as exc:
            if not sent:
                self.send_json({'error': f'browser {port} is unavailable: {exc}; check its server log'}, 502)
        finally:
            conn.close()

    def handle_request(self):
        path = urlsplit(self.path).path
        if path == '/healthz' and self.command in ('GET', 'HEAD'):
            self.send_json({'service': IDENTITY})
        elif path == '/' and self.command in ('GET', 'HEAD'):
            self.index()
        else:
            self.proxy()

    do_GET = do_POST = do_HEAD = handle_request


class PortalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, registry=DEFAULT_REGISTRY, request_timeout=900):
        self.registry = Path(registry)
        self.request_timeout = request_timeout
        super().__init__(address, PortalHandler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--request-timeout', type=float, default=900)
    parser.add_argument('--backend', action='append', default=[], metavar='PORT[=LABEL]',
                        help='Register an already running browser; may be repeated.')
    args = parser.parse_args()
    for item in args.backend:
        port, _, label = item.partition('=')
        if int(port) == args.port:
            parser.error('a backend cannot use the portal port')
        register_backend(int(port), label or f'Browser {port}', args.registry)
    server = PortalServer((args.host, args.port), args.registry, args.request_timeout)
    print(f'CELLECT browser portal: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
