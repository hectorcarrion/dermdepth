#!/usr/bin/env python3
"""Wound/Lesion Annotation Server.

Serves a browser-based polygon annotation tool for marking wound/lesion regions.
Annotations are saved as JSON and can be used to compute 3D surface area.

Usage:
    python code/annotation/wound_annotator.py --port 8765

Then open http://localhost:8765 in your browser (port-forward if remote).

Keyboard shortcuts:
    Click     — add polygon vertex
    Z         — undo last point
    C         — close polygon
    X         — clear polygon
    S         — save annotation
    Left/Right — navigate samples
"""

import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, unquote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DATA = PROJECT_ROOT / "output" / "eval_data"
ANNOTATIONS_FILE = PROJECT_ROOT / "output" / "annotations" / "wound_annotations.json"
HTML_FILE = Path(__file__).parent / "annotator.html"


def get_samples():
    """Collect all samples from eval datasets."""
    samples = []
    for dataset in ['woundsdb', 'skinl2']:
        data_dir = EVAL_DATA / dataset
        if not data_dir.exists():
            continue
        for d in sorted(data_dir.iterdir()):
            if d.is_dir() and (d / "image.png").exists():
                samples.append({
                    'dataset': dataset,
                    'name': d.name,
                    'key': f"{dataset}/{d.name}",
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


class AnnotatorHandler(SimpleHTTPRequestHandler):
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
            # /api/image/{dataset}/{sample_name}
            parts = path.replace('/api/image/', '').split('/', 1)
            if len(parts) == 2:
                img_path = EVAL_DATA / parts[0] / parts[1] / "image.png"
                if img_path.exists():
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.end_headers()
                    self.wfile.write(img_path.read_bytes())
                    return
            self.send_error(404)

        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/save':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            annotations = load_annotations()
            annotations[data['key']] = {
                'dataset': data['dataset'],
                'name': data['name'],
                'points': data['points'],
                'width': data['width'],
                'height': data['height'],
            }
            save_annotations(annotations)
            self.send_json({'status': 'ok', 'total': len(annotations)})
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
        # Quiet logging — only print saves
        if 'POST' in str(args):
            print(f"  [SAVE] {args}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wound/Lesion Annotation Server')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    print(f"DermDepth Wound/Lesion Annotator")
    print(f"  Samples: {len(get_samples())} ({sum(1 for s in get_samples() if s['dataset']=='woundsdb')} WoundsDB, "
          f"{sum(1 for s in get_samples() if s['dataset']=='skinl2')} SKINL2)")
    print(f"  Annotations: {ANNOTATIONS_FILE}")
    print(f"  Existing: {len(load_annotations())} annotated")
    print(f"\n  Open: http://localhost:{args.port}")
    print(f"  (Port-forward if on remote server)\n")

    server = HTTPServer(('0.0.0.0', args.port), AnnotatorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
