/**
 * Titanic DS App — local proxy server
 * Routes browser → this server → api.anthropic.com (bypasses CORS)
 *
 * Requirements: Node.js 18+ (built-in fetch, no npm needed)
 * Usage:  node server.js
 * Then open: http://localhost:3131
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = 3131;
const HTML_FILE = path.join(__dirname, 'titanic_ds_app.html');

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);

  // ── CORS preflight ──────────────────────────────────────────────
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-api-key, anthropic-version');
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // ── Serve the HTML app ──────────────────────────────────────────
  if (req.method === 'GET' && parsed.pathname === '/') {
    try {
      const html = fs.readFileSync(HTML_FILE, 'utf8');
      // Patch the app so it calls /proxy instead of api.anthropic.com directly
      const patched = html.replace(
        /https:\/\/api\.anthropic\.com\/v1\/messages/g,
        '/proxy'
      );
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(patched);
    } catch (e) {
      res.writeHead(500); res.end('Could not read titanic_ds_app.html — make sure it is in the same folder as server.js');
    }
    return;
  }

  // ── Proxy POST /proxy → api.anthropic.com ──────────────────────
  if (req.method === 'POST' && parsed.pathname === '/proxy') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const apiKey = req.headers['x-api-key'] || '';
      if (!apiKey) {
        res.writeHead(401, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: 'No API key provided. Paste your Anthropic key into the API Key field.' } }));
        return;
      }

      const options = {
        hostname: 'api.anthropic.com',
        path: '/v1/messages',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
          'x-api-key': apiKey,
          'Content-Length': Buffer.byteLength(body)
        }
      };

      const proxyReq = https.request(options, proxyRes => {
        res.writeHead(proxyRes.statusCode, { 'Content-Type': 'application/json' });
        proxyRes.pipe(res);
      });

      proxyReq.on('error', err => {
        res.writeHead(502, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: 'Proxy error: ' + err.message } }));
      });

      proxyReq.write(body);
      proxyReq.end();
    });
    return;
  }

  res.writeHead(404); res.end('Not found');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log('');
  console.log('  ✅  Titanic DS App running!');
  console.log(`  👉  Open in your browser: http://localhost:${PORT}`);
  console.log('');
  console.log('  Paste your Anthropic API key into the "API Key" field in the app.');
  console.log('  Press Ctrl+C to stop the server.');
  console.log('');
});
