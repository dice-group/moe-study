import pytest

from pipeline_tools import cross_arch_model_cache as cache
from pipeline_tools.cross_arch_study import write_json, sha256


def test_verified_copy_and_fail_closed_receipt(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "model").mkdir(parents=True)
    weight = source / "model" / "model.safetensors"
    weight.write_bytes(b"small frozen model fixture")
    write_json(source / "assets.json", {"pass": True, "revision": "frozen", "files": {
        weight.name: {"bytes": weight.stat().st_size, "sha256": sha256(weight)}}})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(cache, "model_root", lambda _: source)
    monkeypatch.setattr(cache, "cache_root", lambda _: tmp_path / "local")
    monkeypatch.setattr(cache, "contract", lambda: {"models": {"test": {"revision": "frozen"}}})
    assert cache.local_model_path("test") == source / "model"
    cache.stage("test")
    local = cache.local_model_path("test")
    assert (local / weight.name).read_bytes() == weight.read_bytes()
    cache.stage("test")  # An already verified copy is reused.
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    with pytest.raises(ValueError, match="Stale or foreign"):
        cache.local_model_path("test")


def test_corrupt_copy_is_never_marked_ready(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "model").mkdir(parents=True)
    (source / "model" / "weight").write_bytes(b"changed")
    write_json(source / "assets.json", {"pass": True, "revision": "frozen", "files": {
        "weight": {"bytes": 7, "sha256": "0" * 64}}})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(cache, "model_root", lambda _: source)
    monkeypatch.setattr(cache, "cache_root", lambda _: tmp_path / "local")
    monkeypatch.setattr(cache, "contract", lambda: {"models": {"test": {"revision": "frozen"}}})
    with pytest.raises(ValueError, match="byte verification failed"):
        cache.stage("test")
    assert not (tmp_path / "local/ready.json").exists()
