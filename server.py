"""
server.py — avvialo con:  python server.py
Poi apri il browser su:   http://localhost:8765
"""

import http.server
import json
import os
import urllib.parse
from datetime import datetime

PORT     = 8765
DB_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "targhe_db.json")
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexV-1.html")


def load_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {"sessioni": {}, "targhe": []}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Log pulito
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

    def _send(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path

        # ── API: elenco targhe ──────────────────────────────────
        if path == "/api/targhe":
            db   = load_db()
            body = json.dumps(db, ensure_ascii=False, indent=2).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return

        # ── API: statistiche ────────────────────────────────────
        if path == "/api/stats":
            db      = load_db()
            targhe  = db.get("targhe", [])
            unique  = len({r["targa"] for r in targhe})
            sessioni = len(db.get("sessioni", {}))
            body = json.dumps({
                "totale":   len(targhe),
                "uniche":   unique,
                "sessioni": sessioni
            }).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return

        # ── Serve indexV-1.html ─────────────────────────────────────
        if path in ("/", "/indexV-1.html"):
            if os.path.exists(HTML_PATH):
                with open(HTML_PATH, "rb") as f:
                    body = f.read()
                self._send(200, "text/html; charset=utf-8", body)
            else:
                self._send(404, "text/plain", b"indexV-1.html non trovato")
            return

        self._send(404, "text/plain", b"Not found")


if __name__ == "__main__":
    print(f"\n  Archivio targhe — server avviato")
    print(f"  Database : {DB_PATH}")
    print(f"  Interfaccia : http://localhost:{PORT}")
    print(f"  Premi Ctrl+C per fermare\n")
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server fermato.")