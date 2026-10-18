import json
import os
import subprocess
import sys

import pytest

from studylens.config import Config
from studylens.engine import StudyLens


def _engine(tmp_path):
    return StudyLens(Config(data_dir=tmp_path / "data", embed_model="hashing", enable_reranker=False))


def test_notebooks_live_in_qdrant_not_a_json_file(tmp_path):
    e = _engine(tmp_path)
    nb = e.create_notebook("ML")
    assert e.list_notebooks()[nb]["name"] == "ML"
    assert not (tmp_path / "data" / "notebooks.json").exists()
    e.delete_notebook(nb)
    assert nb not in e.list_notebooks()


def test_legacy_notebooks_json_is_migrated(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "notebooks.json").write_text(json.dumps({"abc12345": {"name": "Old notebook", "created": 1.0}}))
    e = _engine(tmp_path)
    assert e.list_notebooks()["abc12345"]["name"] == "Old notebook"


def test_cache_rebuilds_when_chunks_change_underneath(tmp_path):
    e = _engine(tmp_path)
    nb = e.create_notebook("n")
    e.add_text(nb, "a", "alpha beta gamma " * 30)
    assert len(e.all_chunks(nb)) == 1
    # simulate ANOTHER process adding a source: write straight to the store, bypassing the engine
    other = StudyLens.__new__(StudyLens)
    other.__dict__.update(e.__dict__)
    other._cache = {}
    other.add_text(nb, "b", "delta epsilon zeta " * 30)
    assert len(e.all_chunks(nb)) == 2  # e's cache noticed the count changed


def test_second_process_on_same_local_folder_gets_a_clear_error(tmp_path):
    held = _engine(tmp_path)  # keep a reference so this process really holds the folder
    code = (
        "from studylens.config import Config\n"
        "from studylens.engine import StudyLens\n"
        f"StudyLens(Config(data_dir=__import__('pathlib').Path(r'{tmp_path / 'data'}'), embed_model='hashing', enable_reranker=False))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert r.returncode != 0 and "QDRANT_URL" in r.stderr
    assert held is not None
