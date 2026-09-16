"""Optional byte-identical, job-local shared libraries; never install packages."""
import os
import sys

_packages = os.environ.get("MOE_STUDY_LOCAL_PYTHON")
if _packages:
    sys.path.insert(0, _packages)

if os.environ.get("MOE_STUDY_DEBUG_STACKS") == "1":
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)

_runtime = os.environ.get("MOE_STUDY_LOCAL_TORCH_LIBS")
if _runtime:
    import ctypes
    try:
        for _library in ("libtorch_global_deps.so", "libtorch_python.so"):
            ctypes.CDLL(os.path.join(_runtime, _library), mode=ctypes.RTLD_GLOBAL)
    except Exception as _error:
        print(f"Node-local Torch runtime failed: {_error}", file=sys.stderr, flush=True)
        os._exit(90)
