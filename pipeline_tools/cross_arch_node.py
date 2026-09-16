"""Campaign-only, verified startup cache; retain the original Slurm signal path."""
from pipeline.executor import SlurmExecutor
from pipeline_tools.cross_arch_study import ROOT, runtime_profile


def excluded_nodes():
    # Both nodes had explicit NHC/NVLink drain evidence on 2026-09-13.
    # This operational exclusion does not change the frozen scientific contract.
    return sorted(set(runtime_profile()["exclude_gpu_nodes"]))


class CrossArchSlurmExecutor(SlurmExecutor):
    def _build_job_script(self, *args, **kwargs):
        script = super()._build_job_script(*args, **kwargs)
        if not str(self.gres).startswith("gpu:"):
            return script
        marker = "export PYTHONUNBUFFERED=1\n"
        if script.count(marker) != 1:
            raise ValueError("Unexpected Slurm prologue; cannot safely install startup cache")
        setup = ROOT / "configs/cross_arch_study/activate_node_cache.sh"
        script = script.replace(marker, marker + f"source '{setup}' || exit $?\n")
        # Preserve both cleanup routines without changing signal forwarding.
        return script.replace("trap cleanup_tokenized_copy EXIT",
                              "trap 'cleanup_tokenized_copy; cleanup_cross_arch_cache' EXIT")
