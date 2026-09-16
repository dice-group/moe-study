from pipeline_tools.cross_arch_node import CrossArchSlurmExecutor
from pipeline.executor import SlurmExecutor, _normalize_preflight_config


def test_gpu_cache_precedes_preflight_and_preserves_cleanup_and_signals():
    executor = CrossArchSlurmExecutor(gres="gpu:a100:4", preflight=_normalize_preflight_config({"enabled": True}))
    script = executor._build_job_script(["python", "worker.py"], {}, "test", "test.log")
    assert script.index("source '") < script.index("[GPU_PREFLIGHT]")
    assert "trap 'cleanup_tokenized_copy; cleanup_cross_arch_cache' EXIT" in script
    assert "trap 'signal_forward_usr1' USR1" in script
    assert "exec python worker.py" in script


def test_cpu_jobs_keep_the_original_script():
    arguments = (["python", "report.py"], {}, "report", "report.log")
    assert CrossArchSlurmExecutor(gres="")._build_job_script(*arguments) == SlurmExecutor(gres="")._build_job_script(*arguments)
