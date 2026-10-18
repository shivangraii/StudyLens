"""Pre-index demo notebooks so recruiters see content instantly.

    python -m scripts.ingest_samples samples/manifest.json

manifest.json example: see samples/manifest.example.json
"""
from __future__ import annotations

import json
import sys

from studylens.config import Config
from studylens.engine import StudyLens


def main(manifest_path: str) -> None:
    engine = StudyLens(Config())
    manifest = json.load(open(manifest_path))
    existing = {m["name"]: nid for nid, m in engine.list_notebooks().items()}
    for nb in manifest["notebooks"]:
        nb_id = existing.get(nb["name"]) or engine.create_notebook(nb["name"])
        print(f"== {nb['name']} ({nb_id})")
        for src in nb["sources"]:
            try:
                if src["type"] == "youtube":
                    title, n, how = engine.add_youtube(nb_id, src["value"])
                    print(f"  youtube: {title} -> {n} chunks ({how})")
                elif src["type"] == "web":
                    title, n = engine.add_web(nb_id, src["value"])
                    print(f"  web: {title} -> {n} chunks")
                else:
                    name, n = engine.add_file(nb_id, src["value"])
                    print(f"  file: {name} -> {n} chunks")
            except Exception as e:
                print(f"  !! failed {src}: {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples/manifest.json")
