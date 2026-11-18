from scripts.analysis.snapshot_slurm_job_graph import _normalize_dependency


def test_normalize_dependency_accepts_slurm_expanded_multi_job_form() -> None:
    manifest = "afterok:101:102:103"
    live = "afterok:101(unfulfilled),afterok:102(fulfilled),afterok:103(unfulfilled)"

    assert _normalize_dependency(manifest) == _normalize_dependency(live)


def test_normalize_dependency_keeps_dependency_types_distinct() -> None:
    assert _normalize_dependency("afterany:101,afterok:102") != _normalize_dependency(
        "afterok:101:102"
    )
