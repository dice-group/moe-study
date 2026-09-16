"""CPU regression checks; no GPU allocation or scheduler submission."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.executor import _tokenized_copy_shell_lines
from pipeline.gpu_preflight import _run
from pipeline.training import _maybe_stage_tokenized_path


@pytest.mark.parametrize("destination", ["same", "parent", "child", "alias"])
def test_copy_rejects_overlapping_paths_without_removing_source(tmp_path, destination):
    source = tmp_path / "source"
    source.mkdir()
    sentinel = source / "data.arrow"
    sentinel.write_bytes(b"keep")
    target = {"same": source, "parent": tmp_path, "child": source / "copy", "alias": tmp_path / "alias"}[destination]
    if destination == "alias":
        target.symlink_to(source, target_is_directory=True)
    env = dict(os.environ, PROOF_PIPELINE_TOKENIZED_LOCAL_COPY="1",
               PROOF_PIPELINE_TOKENIZED_SRC=str(source), PROOF_PIPELINE_TOKENIZED_DST=str(target),
               PROOF_PIPELINE_TOKENIZED_CLEANUP="1", PROOF_PIPELINE_TOKENIZED_JOB_SCOPED="1")
    result = subprocess.run(["bash", "-c", "\n".join(_tokenized_copy_shell_lines(cleanup_trap=True))],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 96
    assert sentinel.read_bytes() == b"keep"


@pytest.mark.parametrize("name", ["..", "/", "../other", "nested/copy"])
def test_copy_name_cannot_escape_configured_root(tmp_path, name):
    with pytest.raises(SystemExit, match="single directory name"):
        _maybe_stage_tokenized_path(
            env={}, tokenized_path=tmp_path / "source", run_cfg={},
            common={"tokenized_local_copy": {"enabled": True, "root": str(tmp_path), "name": name}},
        )


def test_preflight_timeout_kills_descendants(tmp_path):
    marker = tmp_path / "child-survived"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).touch()"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)"
    with pytest.raises(SystemExit, match="timed out"):
        _run([sys.executable, "-c", parent], timeout_seconds=1)
    # A separate bounded process waits beyond the child's intended write time.
    subprocess.run([sys.executable, "-c", "import time; time.sleep(1.5)"], check=True, timeout=3)
    assert not marker.exists()


def test_preflight_reports_command_failure():
    with pytest.raises(SystemExit, match="failed"):
        _run([sys.executable, "-c", "raise SystemExit(2)"], timeout_seconds=2)
