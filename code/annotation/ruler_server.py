#!/usr/bin/env python3
"""DDI Ruler Annotation Server.

Serves a browser-based tool for marking two points exactly 1cm apart on ruler
markings in DDI clinical images. Annotations are used to compute metric scale
accuracy stratified by Fitzpatrick skin tone.

Usage:
    python code/annotation/ruler_server.py --port 8766

Then open http://localhost:8766 in your browser (port-forward if remote).

Keyboard shortcuts:
    Click        — add point (max 2)
    Z            — undo last point
    X            — clear points
    S            — save annotation (1cm = 2 points)
    N            — mark as "no ruler visible"
    Left/Right   — navigate samples
"""

import csv
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, unquote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DDI_DIR = PROJECT_ROOT / "data" / "DDI"
DDI_IMAGES = DDI_DIR / "images"
DDI_MAP = DDI_DIR / "map.csv"
ANNOTATIONS_FILE = PROJECT_ROOT / "output" / "annotations" / "ddi_ruler_annotations.json"
HTML_FILE = Path(__file__).parent / "ruler_annotator.html"


def load_ddi_metadata():
    """Load DDI map.csv into a dict keyed by filename."""
    meta = {}
    with open(DDI_MAP) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row['DDI_file']
            meta[fn] = {
                'ddi_id': int(row['DDI_ID']),
                'filename': fn,
                'skin_tone': str(row['skin_tone']),
                'malignant': row['malignant'] == 'True',
                'disease': row['disease'],
            }
    return meta


def get_samples():
    """Collect all DDI samples with metadata."""
    meta = load_ddi_metadata()
    samples = []
    for fn in sorted(meta.keys()):
        m = meta[fn]
        img_path = DDI_IMAGES / fn
        if img_path.exists():
            samples.append({
                'key': fn,
                'filename': fn,
                'skin_tone': m['skin_tone'],
                'disease': m['disease'],
                'malignant': m['malignant'],
            })
    return samples


def load_annotations():
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return {}


def save_annotations(annotations):
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)


class RulerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_FILE.read_bytes())

        elif path == '/api/samples':
            self.send_json(get_samples())

        elif path == '/api/annotations':
            self.send_json(load_annotations())

        elif path.startswith('/api/image/'):
            filename = path.replace('/api/image/', '')
            img_path = DDI_IMAGES / filename
            if img_path.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.end_headers()
                self.wfile.write(img_path.read_bytes())
            else:
                self.send_error(404)

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/save':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            annotations = load_annotations()
            annotations[data['key']] = data
            save_annotations(annotations)
            n_ruler = sum(1 for v in annotations.values() if v.get('type') == 'ruler')
            n_no = sum(1 for v in annotations.values() if v.get('type') == 'no_ruler')
            self.send_json({'status': 'ok', 'total': len(annotations),
                            'ruler': n_ruler, 'no_ruler': n_no})
        else:
            self.send_error(404)

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if 'POST' in str(args):
            print(f"  [SAVE] {args}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DDI Ruler Annotation Server')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()

    samples = get_samples()
    annotations = load_annotations()
    tone_counts = {}
    for s in samples:
        t = s['skin_tone']
        tone_counts[t] = tone_counts.get(t, 0) + 1

    print(f"DDI Ruler Annotator")
    print(f"  Images: {len(samples)} total")
    for t in sorted(tone_counts):
        print(f"    Fitzpatrick {t}: {tone_counts[t]}")
    print(f"  Annotations: {ANNOTATIONS_FILE}")
    n_ruler = sum(1 for v in annotations.values() if v.get('type') == 'ruler')
    n_no = sum(1 for v in annotations.values() if v.get('type') == 'no_ruler')
    print(f"  Existing: {n_ruler} ruler + {n_no} no-ruler = {n_ruler + n_no} total")
    print(f"\n  Open: http://localhost:{args.port}")
    print(f"  (Port-forward if on remote server)\n")

    server = HTTPServer(('0.0.0.0', args.port), RulerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
