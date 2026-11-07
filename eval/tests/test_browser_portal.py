import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from eval.hsctiles.browser_portal import (
    PortalServer, backends, register_backend, rewrite_urls, unregister_backend,
)


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.last_path = self.path
        if self.path == '/':
            body = b'<script src="/pages/app.js"></script><img src="/assets/blank.png"><link href="style.css"><script src="app.js"></script>'
            content_type = 'text/html'
        elif self.path.startswith('/image/'):
            body, content_type = b'\x89PNG\r\n\x00\xff', 'image/png'
        elif self.path in ('/pages/app.js', '/app.js'):
            body, content_type = b"fetch('/api/state');img.src=`/tile_map/${patch}.png`;", 'application/javascript'
        else:
            body = json.dumps(dict(path=self.path, backend=self.server.server_port,
                                   image_url='/image/t.png',
                                   checkpoint='/data/model.pt')).encode()
            content_type = 'application/json'
        self.send_response(400 if self.path == '/api/error' else 200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BrowserPortalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmp.name)
        self.servers = []
        self.threads = []
        self.a = self.start(ThreadingHTTPServer(('127.0.0.1', 0), Backend))
        self.b = self.start(ThreadingHTTPServer(('127.0.0.1', 0), Backend))
        register_backend(self.a.server_port, 'first <model>', self.registry)
        register_backend(self.b.server_port, 'second', self.registry)
        self.portal = self.start(PortalServer(('127.0.0.1', 0), self.registry))

    def start(self, server):
        self.servers.append(server)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        self.threads.append(thread)
        return server

    def tearDown(self):
        for server in reversed(self.servers):
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(1)
        self.tmp.cleanup()

    def request(self, path, method='GET', body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.portal.server_port, timeout=2)
        try:
            conn.request(method, path, body=body, headers={'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_independent_backends_and_query(self):
        for backend in [self.a, self.b]:
            prefix = f'/browser/{backend.server_port}'
            status, _, body = self.request(prefix + '/api/state?tile=x004_y005&v=1')
            row = json.loads(body)
            self.assertEqual(status, 200)
            self.assertEqual(row['backend'], backend.server_port)
            self.assertEqual(backend.last_path, '/api/state?tile=x004_y005&v=1')
            self.assertEqual(row['path'], prefix + '/api/state?tile=x004_y005&v=1')
            self.assertEqual(row['image_url'], prefix + '/image/t.png')
            self.assertEqual(row['checkpoint'], '/data/model.pt')

    def test_html_javascript_and_binary(self):
        prefix = f'/browser/{self.a.server_port}'
        self.assertIn((prefix + '/pages/app.js').encode(), self.request(prefix + '/')[2])
        script = self.request(prefix + '/pages/app.js')[2]
        self.assertIn((prefix + '/api/state').encode(), script)
        self.assertIn((prefix + '/tile_map/${patch}.png').encode(), script)
        self.assertEqual(self.request(prefix + '/image/t.png')[2], b'\x89PNG\r\n\x00\xff')

    def test_post_and_upstream_error(self):
        prefix = f'/browser/{self.a.server_port}'
        payload = b'{"selected":true,"token":"hello"}'
        self.assertEqual(self.request(prefix + '/api/select', 'POST', payload)[2], payload)
        self.assertEqual(self.request(prefix + '/api/error')[0], 400)

    def test_training_browser_relative_assets_and_query_api(self):
        prefix = f'/browser/{self.a.server_port}'
        page = self.request(prefix + '/')[2]
        self.assertIn(b'href="style.css"', page)
        self.assertIn(b'src="app.js"', page)
        status, _, script = self.request(prefix + '/app.js')
        self.assertEqual(status, 200)
        self.assertIn((prefix + '/api/state').encode(), script)
        self.request(prefix + '/api/image?token=test&view=gray')
        self.assertEqual(self.a.last_path, '/api/image?token=test&view=gray')

    def test_index_escape_redirect_and_allowlist(self):
        status, _, body = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'first &lt;model&gt;', body)
        status, headers, _ = self.request(f'/browser/{self.a.server_port}?v=2')
        self.assertEqual(status, 307)
        self.assertEqual(headers['Location'], f'/browser/{self.a.server_port}/?v=2')
        self.assertEqual(self.request('/browser/1/api/state')[0], 404)
        self.assertEqual(self.request(f'/browser/{self.portal.server_port}/')[0], 404)

    def test_dead_backend_returns_error(self):
        self.a.shutdown()
        self.a.server_close()
        status, _, body = self.request(f'/browser/{self.a.server_port}/api/state')
        self.assertEqual(status, 502)
        self.assertIn(b'unavailable', body)

    def test_unregister_cannot_remove_new_registration(self):
        old = register_backend(12345, 'old', self.registry)
        new = register_backend(12345, 'new', self.registry)
        unregister_backend(12345, old, self.registry)
        self.assertEqual(backends(self.registry)[12345]['label'], 'new')
        unregister_backend(12345, new, self.registry)
        self.assertNotIn(12345, backends(self.registry))

    def test_training_zarr_entry_displays_its_root(self):
        register_backend(self.a.server_port, 'Training Zarr · A', self.registry,
                         root='/data/version_A/zarr')
        status, _, body = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'/data/version_A/zarr', body)


class RewriteTest(unittest.TestCase):
    def test_keeps_science_paths_external_urls_and_queries(self):
        original = b'''{"path":"/data/shared/x.fits","url":"https://other/image/x.png","image_url":"/image/x.png?detect=1&mask_filter=1"}'''
        result = json.loads(rewrite_urls(original, '/browser/8813'))
        self.assertEqual(result['path'], '/data/shared/x.fits')
        self.assertEqual(result['url'], 'https://other/image/x.png')
        self.assertEqual(result['image_url'], '/browser/8813/image/x.png?detect=1&mask_filter=1')

    def test_current_frontend_root_urls_all_supported(self):
        pages = Path(__file__).resolve().parents[1] / 'hsctiles/pages'
        import re
        for path in pages.iterdir():
            if path.suffix not in ('.js', '.html', '.css'):
                continue
            text = path.read_bytes()
            rewritten = rewrite_urls(text, '/browser/8813').decode()
            unresolved = re.findall(r'''["'`]/(?:api|pages|assets|image|tile_map)/''', rewritten)
            self.assertEqual(unresolved, [], path.name)


if __name__ == '__main__':
    unittest.main()
