from __future__ import annotations

import argparse
import runpy
import sys
from typing import Any


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        split_at = argv.index("--")
        own_args = argv[:split_at]
        module_args = argv[split_at + 1 :]
    else:
        own_args = argv
        module_args = []
    parser = argparse.ArgumentParser(description="Launch a module with PyCharm debugger attached.")
    parser.add_argument("--module", required=True, help="Module path, e.g. llamafactory.cli")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--suspend", default="false", help="true/false")
    args = parser.parse_args(own_args)
    return args, module_args


def main(argv: list[str] | None = None) -> int:
    args, module_args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.port <= 0:
        raise SystemExit(f"--port must be > 0 (got {args.port})")
    suspend = _parse_bool(args.suspend, default=False)
    try:
        import pydevd_pycharm
    except Exception as exc:
        raise SystemExit(
            "PyCharm debug requested but pydevd_pycharm is not installed. "
            "Install it in this environment (e.g. `pip install pydevd-pycharm`)."
        ) from exc
    pydevd_pycharm.settrace(
        str(args.host),
        port=int(args.port),
        stdout_to_server=True,
        stderr_to_server=True,
        suspend=suspend,
    )
    sys.argv = [str(args.module), *module_args]
    runpy.run_module(str(args.module), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
