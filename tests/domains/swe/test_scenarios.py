from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from domains.swe.scenarios import SWEBenchLoader


@pytest.fixture
def stub_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    # Pretend django repo, single commit
    django = cache / "django__django"
    django.mkdir()
    (django / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=django, check=True)
    subprocess.run(["git", "add", "."], cwd=django, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=django, check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=django, text=True).strip()
    # Instance manifest
    manifest = tmp_path / "lite.jsonl"
    manifest.write_text(json.dumps({
        "instance_id": "django__django-123",
        "repo": "django__django",
        "base_commit": head,
        "problem_statement": "broken thing",
        "FAIL_TO_PASS": ["tests/test_x.py::test_y"],
        "PASS_TO_PASS": [],
    }) + "\n")
    return tmp_path


def test_loader_reads_one_instance(stub_cache: Path) -> None:
    loader = SWEBenchLoader(manifest=stub_cache / "lite.jsonl", repo_cache=stub_cache / "cache")
    ids = loader.list_instance_ids()
    assert ids == ["django__django-123"]
    inst = loader.load("django__django-123")
    assert inst.instance_id == "django__django-123"
    assert inst.problem_statement == "broken thing"
    assert inst.fail_to_pass == ["tests/test_x.py::test_y"]
    assert inst.pass_to_pass == []
    assert (Path(inst.repo) / "file.py").exists()


def test_loader_handles_json_encoded_test_lists(tmp_path: Path) -> None:
    """Regression: SWE-bench Lite manifest delivers FAIL_TO_PASS / PASS_TO_PASS
    as JSON-encoded strings (verified on princeton-nlp/SWE-bench_Lite in
    April 2026), not native lists. A naive list() on such a string would
    yield single characters and silently poison the reward signal.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    repo = cache / "astropy__astropy"
    repo.mkdir()
    (repo / "file.py").write_text("x = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo, check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = tmp_path / "lite.jsonl"
    manifest.write_text(json.dumps({
        "instance_id": "astropy__astropy-1",
        "repo": "astropy__astropy",
        "base_commit": head,
        "problem_statement": "foo",
        # Value is a JSON-encoded STRING, matching the real Lite format.
        "FAIL_TO_PASS": '["tests/a.py::t1", "tests/b.py::t2"]',
        "PASS_TO_PASS": '["tests/c.py::t3"]',
    }) + "\n")
    loader = SWEBenchLoader(manifest=manifest, repo_cache=cache)
    inst = loader.load("astropy__astropy-1")
    assert inst.fail_to_pass == ["tests/a.py::t1", "tests/b.py::t2"]
    assert inst.pass_to_pass == ["tests/c.py::t3"]
