#!/usr/bin/env python3
"""Training helpers for proof pipeline."""
from __future__ import annotations
import hashlib
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from .runtime import _abs, _build_run_label, _count_experts, _deep_merge, _load_json, _run_with_tee
from pipeline_tools.adapter_config_verify import verify_adapter_lora_hparams

if TYPE_CHECKING:
    from .executor import Executor
    from .context import PipelineContext


_CHECKPOINT_ADAPTER_RE = re.compile(r"^checkpoint-(\d+)_adapter$")
_DIRTY_ADAPTER_KEY_MARKERS = (".base_layer.", "embed_tokens.", "lm_head.")


def _adapter_weight_file(adapter_dir: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter_dir / name
        if path.exists():
            return path
    return None


def _adapter_checkpoint_step(adapter_dir: Path) -> int | None:
    match = _CHECKPOINT_ADAPTER_RE.match(adapter_dir.name)
    return int(match.group(1)) if match else None


def adapter_checkpoint_provenance(
    adapter_dir: Path,
    *,
    expected_step: int | None = None,
    require_direct_save_marker: bool = False,
) -> dict[str, Any]:
    """Return lightweight provenance/validity metadata for an adapter checkpoint."""
    adapter_dir = Path(adapter_dir)
    step = _adapter_checkpoint_step(adapter_dir)
    weight_file = _adapter_weight_file(adapter_dir)
    errors: list[str] = []
    warnings: list[str] = []
    dirty_keys: list[str] = []
    num_keys: int | None = None

    if expected_step is not None and step != int(expected_step):
        errors.append(f"checkpoint step {step!r} does not match expected step {int(expected_step)}")
    if not (adapter_dir / "adapter_config.json").exists():
        errors.append("adapter_config.json missing")
    if weight_file is None:
        errors.append("adapter weight file missing")
    if require_direct_save_marker and not (adapter_dir / ".adapter_direct_save_complete").exists():
        errors.append(".adapter_direct_save_complete missing")

    if weight_file is not None:
        if weight_file.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
                    keys = list(handle.keys())
                num_keys = len(keys)
                dirty_keys = [key for key in keys if any(marker in key for marker in _DIRTY_ADAPTER_KEY_MARKERS)]
                if dirty_keys:
                    errors.append(f"adapter contains base-model tensors: {dirty_keys[:5]}")
            except Exception as exc:
                errors.append(f"could not inspect adapter safetensors keys: {exc}")
        else:
            message = "adapter weight file is not safetensors; dirty-key inspection skipped"
            if require_direct_save_marker:
                errors.append(message)
            else:
                warnings.append(message)

    return {
        "adapter_path": str(adapter_dir),
        "checkpoint_step": step,
        "expected_step": int(expected_step) if expected_step is not None else None,
        "adapter_config_exists": (adapter_dir / "adapter_config.json").exists(),
        "adapter_weight_file": str(weight_file) if weight_file is not None else "",
        "adapter_weight_size": weight_file.stat().st_size if weight_file is not None else None,
        "direct_save_marker_exists": (adapter_dir / ".adapter_direct_save_complete").exists(),
        "num_adapter_keys": num_keys,
        "dirty_key_count": len(dirty_keys),
        "dirty_key_examples": dirty_keys[:10],
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }

def _cfg_value(cfg: dict[str, Any], key: str, default: Any, *aliases: str) -> Any:
    if key in cfg:
        return cfg[key]
    for alias in aliases:
        if alias in cfg:
            return cfg[alias]
    return default


def _format_steps_arg(value: Any, path: str) -> str:
    """Format step interval values for Trainer args.

    Supports positive integers and positive fractional ratios (e.g. 0.25).
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{path} must be numeric (got {value!r})") from exc
    if numeric <= 0:
        raise SystemExit(f"{path} must be > 0 (got {value!r})")
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


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


def _maybe_stage_tokenized_path(
    *,
    env: dict[str, str],
    tokenized_path: Path,
    common: dict[str, Any],
    run_cfg: dict[str, Any],
) -> Path:
    local_copy_cfg = _deep_merge(common.get("tokenized_local_copy", {}), run_cfg.get("tokenized_local_copy", {}))
    if not isinstance(local_copy_cfg, dict) or not _parse_bool(local_copy_cfg.get("enabled", False), default=False):
        return tokenized_path

    root = str(local_copy_cfg.get("root", "/dev/shm/moe-study-tokenized")).strip()
    if not root:
        root = "/dev/shm/moe-study-tokenized"
    expanded_root = Path(os.path.expandvars(root)).expanduser()
    source = tokenized_path.resolve()
    source_hash = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    local_name = str(local_copy_cfg.get("name", f"{tokenized_path.name}-{source_hash}")).strip()
    if not local_name:
        local_name = f"{tokenized_path.name}-{source_hash}"
    staged_path = expanded_root / local_name

    env["PROOF_PIPELINE_TOKENIZED_LOCAL_COPY"] = "1"
    env["PROOF_PIPELINE_TOKENIZED_SRC"] = str(source)
    env["PROOF_PIPELINE_TOKENIZED_DST"] = str(staged_path)
    env["PROOF_PIPELINE_TOKENIZED_COPY_DEREFERENCE"] = (
        "1" if _parse_bool(local_copy_cfg.get("dereference_symlinks", True), default=True) else "0"
    )
    return staged_path


def _wrap_with_pycharm_debugger(cmd: list[str], debugger_cfg: dict[str, Any]) -> list[str]:
    if len(cmd) < 5 or cmd[1:4] != ['-m', 'llamafactory.cli', 'train']:
        raise SystemExit("training.debugger supports only non-distributed llamafactory.cli train commands.")
    host = str(debugger_cfg.get('host', '127.0.0.1')).strip() or '127.0.0.1'
    try:
        port = int(debugger_cfg.get('port', 12345))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"training.debugger.port must be an integer (got {debugger_cfg.get('port')!r})") from exc
    if port <= 0:
        raise SystemExit(f"training.debugger.port must be > 0 (got {port})")
    suspend = _parse_bool(debugger_cfg.get('suspend', False), default=False)
    return [
        'python3',
        '-m',
        'pipeline.pycharm_debug_launcher',
        '--module',
        'llamafactory.cli',
        '--host',
        host,
        '--port',
        str(port),
        '--suspend',
        'true' if suspend else 'false',
        '--',
        *cmd[3:],
    ]


def _maybe_wrap_debugger_cmd(
    cmd: list[str],
    *,
    common: dict[str, Any],
    run_cfg: dict[str, Any],
    distributed_enabled: bool,
) -> list[str]:
    debugger_cfg = _deep_merge(common.get('debugger', {}), run_cfg.get('debugger', {}))
    if not isinstance(debugger_cfg, dict) or not _parse_bool(debugger_cfg.get('enabled', False), default=False):
        return cmd
    if distributed_enabled:
        raise SystemExit("training.debugger requires training.distributed.enabled=false.")
    kind = str(debugger_cfg.get('kind', 'pycharm')).strip().lower()
    if kind != 'pycharm':
        raise SystemExit(f"Unsupported training.debugger.kind: {kind!r}. Supported values: pycharm")
    return _wrap_with_pycharm_debugger(cmd, debugger_cfg)


def _lpr_args(
    cfg: dict[str, Any],
    common: dict[str, Any],
    language_map_path: Path,
    experts: int,
    prefix: str | None = None,
) -> list[str]:
    """Build the shared Language Prior Routing CLI args used by both CoLA and HydraLoRA."""
    router_mode = _cfg_value(cfg, f'{prefix}_router_mode', 'learned', 'router_mode') if prefix else cfg.get('router_mode', 'learned')
    head_router_mode = _cfg_value(cfg, f'{prefix}_head_router_mode', 'learned', 'head_router_mode') if prefix else cfg.get('head_router_mode', 'learned')
    prior_weight = _cfg_value(cfg, f'{prefix}_prior_weight', 0.1, 'prior_weight') if prefix else cfg.get('prior_weight', 0.1)
    bias_value = _cfg_value(cfg, f'{prefix}_bias_value', 0.0, 'bias_value') if prefix else cfg.get('bias_value', 0.0)
    head_bias_value = _cfg_value(cfg, f'{prefix}_head_bias_value', 0.0, 'head_bias_value') if prefix else cfg.get('head_bias_value', 0.0)
    guidance_scope = _cfg_value(cfg, f'{prefix}_guidance_scope', 'all', 'guidance_scope') if prefix else cfg.get('guidance_scope', 'all')
    return [
        '--language_column', str(common.get('language_column', 'language')),
        '--language_map', str(language_map_path),
        '--language_router_mode', str(router_mode),
        '--language_head_router_mode', str(head_router_mode),
        '--language_prior_weight', str(float(prior_weight)),
        '--language_bias_value', str(float(bias_value)),
        '--language_head_bias_value', str(float(head_bias_value)),
        '--language_guidance_scope', str(guidance_scope),
    ]


def _approach_specific_train_args(
    *,
    kind: str,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    language_map_path: Path,
    experts: int,
) -> list[str]:
    cfg = run_cfg.get('params', {})
    approach_args: list[str] = []
    if kind == 'lora':
        return approach_args
    if kind == 'cola':
        use_experts = bool(_cfg_value(cfg, 'cola_use_experts', True, 'use_experts'))
        approach_args.extend([
            '--num_A', str(int(_cfg_value(cfg, 'cola_num_A', 1, 'num_A'))),
            '--num_B', str(int(_cfg_value(cfg, 'cola_num_B', 3, 'num_B'))),
            '--cola_strategy', str(_cfg_value(cfg, 'cola_strategy', 'fully', 'strategy')),
            '--use_cola_experts', str(use_experts),
            '--cola_num_experts', str(int(_cfg_value(cfg, 'cola_num_experts', experts if use_experts else 1, 'num_experts'))),
            '--cola_top_k', str(int(_cfg_value(cfg, 'cola_top_k', 1, 'top_k'))),
            *_lpr_args(cfg, common, language_map_path, experts, prefix='cola'),
        ])
        use_cola_pissa_init = _cfg_value(cfg, 'use_cola_pissa_init', None, 'use_cola_pissa_init')
        if use_cola_pissa_init is not None:
            approach_args.extend(['--use_cola_pissa_init', str(bool(use_cola_pissa_init))])
        cola_init_lora_weights = _cfg_value(cfg, 'cola_init_lora_weights', None, 'init_lora_weights')
        if cola_init_lora_weights is not None:
            approach_args.extend(['--cola_init_lora_weights', str(cola_init_lora_weights)])
        return approach_args
    if kind == 'hydralora':
        use_experts = bool(_cfg_value(cfg, 'hydralora_use_experts', True, 'use_experts'))
        expert_lora_nums = _cfg_value(cfg, 'hydralora_expert_lora_nums', None, 'expert_lora_nums')
        if isinstance(expert_lora_nums, list):
            expert_lora_nums_arg = ",".join(str(int(x)) for x in expert_lora_nums)
        elif expert_lora_nums is None:
            expert_lora_nums_arg = ""
        else:
            expert_lora_nums_arg = str(expert_lora_nums)
        approach_args.extend([
            '--lora_num', str(int(_cfg_value(cfg, 'hydralora_lora_num', 3, 'lora_num'))),
            '--use_hydralora_experts', str(use_experts),
            '--hydralora_num_experts', str(int(_cfg_value(cfg, 'hydralora_num_experts', experts if use_experts else 1, 'num_experts'))),
            '--hydralora_top_k', str(int(_cfg_value(cfg, 'hydralora_top_k', 1, 'top_k'))),
        ])
        # Only pass language-prior routing args when HydraLoRA experts are enabled.
        if use_experts:
            approach_args.extend(_lpr_args(cfg, common, language_map_path, experts, prefix='hydralora'))
        if expert_lora_nums_arg.strip():
            approach_args.extend(['--hydralora_expert_lora_nums', expert_lora_nums_arg])
        return approach_args
    if kind == 'adamole':
        approach_args.extend([
            '--adamole_num_experts', str(int(_cfg_value(cfg, 'adamole_num_experts', 4, 'num_experts'))),
            '--adamole_aux_loss_coef', str(float(_cfg_value(cfg, 'adamole_aux_loss_coef', 0.0, 'aux_loss_coef'))),
        ])
        max_threshold = _cfg_value(cfg, 'adamole_max_threshold', None, 'max_threshold')
        if max_threshold is not None:
            approach_args.extend(['--adamole_max_threshold', str(float(max_threshold))])
        if bool(_cfg_value(cfg, 'adamole_debug_mode', False, 'debug_mode')):
            approach_args.extend(['--adamole_debug_mode', 'True'])
        return approach_args
    if kind == 'mixlora':
        approach_args.extend([
            '--mixlora_num_experts', str(int(_cfg_value(cfg, 'mixlora_num_experts', experts, 'num_experts'))),
            '--mixlora_top_k', str(int(_cfg_value(cfg, 'mixlora_top_k', 2, 'top_k'))),
            '--mixlora_router_init_range', str(float(_cfg_value(cfg, 'mixlora_router_init_range', 0.02, 'router_init_range'))),
            '--mixlora_jitter_noise', str(float(_cfg_value(cfg, 'mixlora_jitter_noise', 0.0, 'jitter_noise'))),
            '--mixlora_router_loss', str(bool(_cfg_value(cfg, 'mixlora_router_loss', True, 'router_loss'))),
            '--mixlora_router_aux_loss_coef', str(float(_cfg_value(cfg, 'mixlora_router_aux_loss_coef', 0.001, 'router_aux_loss_coef'))),
        ])
        act_fn = _cfg_value(cfg, 'mixlora_act_fn', None, 'act_fn')
        if act_fn is not None:
            approach_args.extend(['--mixlora_act_fn', str(act_fn)])
        moe_targets = _cfg_value(
            cfg,
            'mixlora_moe_target_modules',
            'gate_proj,down_proj,up_proj',
            'moe_target_modules',
        )
        if isinstance(moe_targets, list):
            moe_targets = ",".join(str(item) for item in moe_targets)
        approach_args.extend(['--mixlora_moe_target_modules', str(moe_targets)])
        expert_rank = _cfg_value(cfg, 'mixlora_expert_lora_rank', None, 'expert_lora_rank')
        if expert_rank is not None:
            approach_args.extend(['--mixlora_expert_lora_rank', str(int(expert_rank))])
        expert_alpha = _cfg_value(cfg, 'mixlora_expert_lora_alpha', None, 'expert_lora_alpha')
        if expert_alpha is not None:
            approach_args.extend(['--mixlora_expert_lora_alpha', str(int(expert_alpha))])
        expert_dropout = _cfg_value(cfg, 'mixlora_expert_lora_dropout', None, 'expert_lora_dropout')
        if expert_dropout is not None:
            approach_args.extend(['--mixlora_expert_lora_dropout', str(float(expert_dropout))])
        return approach_args
    if kind == 'mola':
        approach_args.extend([
            '--mola_num_experts', str(int(_cfg_value(cfg, 'mola_num_experts', 4, 'num_experts'))),
            '--mola_top_k', str(int(_cfg_value(cfg, 'mola_top_k', 2, 'top_k'))),
            '--mola_router_aux_loss_coef', str(float(_cfg_value(cfg, 'mola_router_aux_loss_coef', 0.01, 'router_aux_loss_coef'))),
        ])
        num_null_experts = int(_cfg_value(cfg, 'mola_num_null_experts', 0, 'num_null_experts'))
        if num_null_experts > 0:
            approach_args.extend(['--mola_num_null_experts', str(num_null_experts)])
        if bool(_cfg_value(cfg, 'mola_output_router_logits', False, 'output_router_logits')):
            approach_args.extend(['--mola_output_router_logits', 'True'])
        if bool(_cfg_value(cfg, 'mola_aux_loss_annealing', False, 'aux_loss_annealing')):
            approach_args.extend(['--mola_aux_loss_annealing', 'True'])
        aux_loss_coef_end = _cfg_value(cfg, 'mola_aux_loss_coef_end', None, 'aux_loss_coef_end')
        if aux_loss_coef_end is not None:
            approach_args.extend(['--mola_aux_loss_coef_end', str(float(aux_loss_coef_end))])
        if bool(_cfg_value(cfg, 'mola_debug_mode', False, 'debug_mode')):
            approach_args.extend(['--mola_debug_mode', 'True'])
        return approach_args
    if kind == 'moelpr':
        approach_args.extend([
            '--moelpr_stage', str(int(cfg.get('stage', 1))),
            '--moelpr_num_experts', str(int(cfg.get('num_experts', 4))),
            '--moelpr_top_k', str(int(cfg.get('top_k', 1))),
            '--moelpr_aux_loss_coef', str(float(cfg.get('aux_loss_coef', 0.01))),
            '--moelpr_lpr_loss_coef', str(float(cfg.get('lpr_loss_coef', 0.1))),
        ])
        if cfg.get('layers_to_transform') is not None:
            approach_args.extend(['--moelpr_layers_to_transform', str(cfg['layers_to_transform'])])
        if cfg.get('original_language') is not None:
            approach_args.extend(['--moelpr_original_language', str(cfg['original_language'])])
        if bool(cfg.get('debug_mode', False)):
            approach_args.extend(['--moelpr_debug_mode', 'True'])
        return approach_args
    if kind == 'moelora':
        approach_args.extend([
            '--language_column', str(common.get('language_column', 'language')),
            '--language_map', str(language_map_path),
            '--moelora_num_experts', str(int(_cfg_value(cfg, 'moelora_num_experts', 8, 'num_experts'))),
            '--moelora_task_num', str(int(_cfg_value(cfg, 'moelora_task_num', 1, 'task_num'))),
            '--moelora_task_embedding_dim', str(int(_cfg_value(cfg, 'moelora_task_embedding_dim', 64, 'task_embedding_dim'))),
            '--moelora_sparse_top_k', str(int(_cfg_value(cfg, 'moelora_sparse_top_k', 0, 'sparse_top_k'))),
            '--moelora_gate_temperature', str(float(_cfg_value(cfg, 'moelora_gate_temperature', 1.0, 'gate_temperature'))),
        ])
        return approach_args
    if kind == 'mtllora':
        approach_args.extend([
            '--language_map', str(language_map_path),
            '--mtllora_task_num', str(int(_cfg_value(cfg, 'mtllora_task_num', 1, 'task_num'))),
            '--mtllora_num_up_projections', str(int(_cfg_value(cfg, 'mtllora_num_up_projections', experts, 'num_up_projections'))),
            '--mtllora_temperature', str(float(_cfg_value(cfg, 'mtllora_temperature', 0.1, 'temperature'))),
            '--mtllora_lambda_format', str(_cfg_value(cfg, 'mtllora_lambda_format', 'full', 'lambda_format')),
            '--mtllora_use_language_ids_as_task_ids', str(bool(_cfg_value(cfg, 'mtllora_use_language_ids_as_task_ids', True, 'use_language_ids_as_task_ids'))),
        ])
        return approach_args
    if kind == 'movlora':
        approach_args.extend([
            '--movlora_num_experts', str(int(_cfg_value(cfg, 'movlora_num_experts', 8, 'num_experts'))),
            '--movlora_top_k', str(int(_cfg_value(cfg, 'movlora_top_k', 0, 'top_k'))),
            '--movlora_router_temperature', str(float(_cfg_value(cfg, 'movlora_router_temperature', 1.0, 'router_temperature'))),
            '--movlora_router_jitter_noise', str(float(_cfg_value(cfg, 'movlora_router_jitter_noise', 0.0, 'router_jitter_noise'))),
            '--movlora_router_bias', str(bool(_cfg_value(cfg, 'movlora_router_bias', False, 'router_bias'))),
            '--movlora_router_init_std', str(float(_cfg_value(cfg, 'movlora_router_init_std', 0.02, 'router_init_std'))),
            '--movlora_router_ignore_padding_tokens', str(bool(_cfg_value(cfg, 'movlora_router_ignore_padding_tokens', False, 'router_ignore_padding_tokens'))),
        ])
        return approach_args
    if kind == 'hmora':
        approach_args.extend([
            '--hmora_num_experts', str(int(_cfg_value(cfg, 'hmora_num_experts', 8, 'num_experts'))),
            '--hmora_dropout', str(float(_cfg_value(cfg, 'hmora_dropout', 0.1, 'dropout'))),
            '--hmora_top_k_routing_strategy', str(bool(_cfg_value(cfg, 'hmora_top_k_routing_strategy', False, 'top_k_routing_strategy'))),
            '--hmora_top_k', str(int(_cfg_value(cfg, 'hmora_top_k', 2, 'top_k'))),
            '--hmora_use_task_router', str(bool(_cfg_value(cfg, 'hmora_use_task_router', False, 'use_task_router'))),
            '--hmora_task_router_only', str(bool(_cfg_value(cfg, 'hmora_task_router_only', False, 'task_router_only'))),
            '--hmora_share_router_for_qkv', str(bool(_cfg_value(cfg, 'hmora_share_router_for_qkv', False, 'share_router_for_qkv'))),
            '--hmora_share_router_for_w_i', str(bool(_cfg_value(cfg, 'hmora_share_router_for_w_i', False, 'share_router_for_w_i'))),
            '--hmora_num_router_mlp_layers', str(int(_cfg_value(cfg, 'hmora_num_router_mlp_layers', 1, 'num_router_mlp_layers'))),
            '--hmora_router_hidden_dim', str(int(_cfg_value(cfg, 'hmora_router_hidden_dim', 32, 'router_hidden_dim'))),
            '--hmora_epsilon_alpha', str(float(_cfg_value(cfg, 'hmora_epsilon_alpha', 4.0, 'epsilon_alpha'))),
            '--hmora_alpha_shift', str(float(_cfg_value(cfg, 'hmora_alpha_shift', -2.0, 'alpha_shift'))),
            '--hmora_alpha_low_bound', str(float(_cfg_value(cfg, 'hmora_alpha_low_bound', 0.2, 'alpha_low_bound'))),
            '--hmora_alpha_up_bound', str(float(_cfg_value(cfg, 'hmora_alpha_up_bound', 0.8, 'alpha_up_bound'))),
            '--hmora_use_load_balancing_loss', str(bool(_cfg_value(cfg, 'hmora_use_load_balancing_loss', False, 'use_load_balancing_loss'))),
            '--hmora_use_div_loss', str(bool(_cfg_value(cfg, 'hmora_use_div_loss', True, 'use_div_loss'))),
            '--hmora_gamma_div_certain_t', str(float(_cfg_value(cfg, 'hmora_gamma_div_certain_t', 0.4, 'gamma_div_certain_t'))),
            '--hmora_gamma_div_balance_t', str(float(_cfg_value(cfg, 'hmora_gamma_div_balance_t', 1.0, 'gamma_div_balance_t'))),
            '--hmora_gamma_div_certain_s', str(float(_cfg_value(cfg, 'hmora_gamma_div_certain_s', 0.4, 'gamma_div_certain_s'))),
            '--hmora_gamma_div_balance_s', str(float(_cfg_value(cfg, 'hmora_gamma_div_balance_s', 1.0, 'gamma_div_balance_s'))),
            '--hmora_lambda_auxiliary', str(float(_cfg_value(cfg, 'hmora_lambda_auxiliary', 0.003, 'lambda_auxiliary'))),
            '--hmora_lambda_lm', str(float(_cfg_value(cfg, 'hmora_lambda_lm', 1.0, 'lambda_lm'))),
            '--hmora_use_hydra_lora', str(bool(_cfg_value(cfg, 'hmora_use_hydra_lora', False, 'use_hydra_lora'))),
            '--hmora_eta_b', str(float(_cfg_value(cfg, 'hmora_eta_b', 1.0, 'eta_b'))),
            '--hmora_use_language_ids_as_task_ids', str(bool(_cfg_value(cfg, 'hmora_use_language_ids_as_task_ids', False, 'use_language_ids_as_task_ids'))),
            '--hmora_task_token', str(_cfg_value(cfg, 'hmora_task_token', '?', 'task_token')),
            '--hmora_num_encoder_layer', str(int(_cfg_value(cfg, 'hmora_num_encoder_layer', 1, 'num_encoder_layer'))),
        ])
        task_token_id = _cfg_value(cfg, 'hmora_task_token_id', None, 'task_token_id')
        if task_token_id is not None:
            approach_args.extend(['--hmora_task_token_id', str(int(task_token_id))])
        target_modules_lora = _cfg_value(cfg, 'hmora_target_modules_lora', None, 'target_modules_lora')
        if target_modules_lora is not None:
            value = target_modules_lora
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            approach_args.extend(['--hmora_target_modules_lora', str(value)])
        return approach_args
    raise SystemExit(f'Unsupported finetuning kind: {kind}')


def build_training_cmd(*, repo_root: Path, env: dict[str, str], output_dir: Path, run_cfg: dict[str, Any], common: dict[str, Any], language_map_path: Path, tokenized_path: Path, embedding_training_cfg: dict[str, Any], run_name: str) -> tuple[list[str], dict[str, str]]:
    """Build the training command and environment without executing.

    Returns (cmd, env) so callers can either run immediately or batch-submit.
    """
    env = dict(env)  # don't mutate caller's dict
    kind = str(run_cfg['kind'])
    seed = int(run_cfg.get('seed', common.get('seed', 42)))
    train = _deep_merge(common.get('train', {}), run_cfg.get('train', {}))
    lora = _deep_merge(common.get('lora', {}), run_cfg.get('lora', {}))
    # Allow per-profile or per-run env injection (useful for Slurm signal / torchrun tests).
    extra_env_raw = (
        run_cfg.get("extra_env")
        or run_cfg.get("env")
        or common.get("extra_env")
        or common.get("env")
        or {}
    )
    if isinstance(extra_env_raw, dict):
        for k, v in extra_env_raw.items():
            key = str(k).strip()
            if not key:
                continue
            env[key] = str(v)
    eval_cfg = _deep_merge(common.get('eval', {}), run_cfg.get('eval', {}))
    eval_strategy = str(eval_cfg.get('strategy', 'steps')).strip().lower()
    eval_enabled = bool(eval_cfg.get('enabled', True)) and eval_strategy != 'no'
    max_steps_override = str(os.environ.get('HALA_MAX_STEPS_OVERRIDE', '')).strip()
    schedule_args: list[str] = []
    if max_steps_override:
        schedule_args.extend(['--max_steps', str(int(max_steps_override))])
    else:
        max_steps_raw = train.get('max_steps')
        max_steps_enabled = False
        if max_steps_raw is not None:
            try:
                max_steps_enabled = int(max_steps_raw) > 0
            except (TypeError, ValueError):
                max_steps_enabled = False
        if max_steps_enabled:
            schedule_args.extend(['--max_steps', str(int(train.get('max_steps')))])
        else:
            schedule_args.extend(['--num_train_epochs', str(float(train.get('num_train_epochs', 1.0)))])
    report_to = str(run_cfg.get('report_to', common.get('report_to', 'none')))
    num_experts_raw = common.get('num_experts')
    if num_experts_raw is None:
        experts = max(1, int(_count_experts(language_map_path)))
    else:
        experts = max(1, int(num_experts_raw))
    disable_gradient_checkpointing = bool(train.get('disable_gradient_checkpointing', False))
    use_reentrant_gc = bool(train.get('use_reentrant_gc', True))
    # MoE-LoRA's shared routing gate is reused by all adapted layers. Under FSDP,
    # checkpoint recomputation can see different shared-gate tensor metadata than
    # the original forward, so keep checkpointing off for this method.
    if kind == 'moelora' and bool(run_cfg.get("params", {}).get("moelora_force_disable_gradient_checkpointing", True)):
        disable_gradient_checkpointing = True
    # Adapter routing modules that are shared outside the wrapped decoder layer can
    # be accessed during checkpoint recomputation; non-reentrant GC avoids FSDP
    # seeing sharded shared gate/embedding weights in those recompute calls.
    if kind in {'adamole', 'moelora', 'hmora'} and (not disable_gradient_checkpointing) and use_reentrant_gc:
        use_reentrant_gc = False
    use_bf16 = bool(train.get('bf16', True))
    use_fp16 = bool(train.get('fp16', False))
    if kind in {"moelora", "adamole", "hmora", "mola"}:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    low_cpu_mem_usage = train.get("low_cpu_mem_usage", common.get("low_cpu_mem_usage"))
    effective_tokenized_path = _maybe_stage_tokenized_path(
        env=env,
        tokenized_path=tokenized_path,
        common=common,
        run_cfg=run_cfg,
    )
    train_args = [
        '--stage', 'sft',
        '--do_train',
        '--model_name_or_path', str(common['model_name_or_path']),
        '--dataset', str(common.get('dataset_name', 'c4')),
        '--dataset_dir', str(_abs(repo_root, str(common.get('dataset_dir', 'LLaMA-Factory/data')))),
        '--template', str(common.get('template', 'llama3')),
        '--finetuning_type', kind,
        '--output_dir', str(output_dir),
        '--overwrite_output_dir',
        '--run_name', str(common.get('train_run_prefix', 'train') + '/' + run_name),
        *schedule_args,
        '--logging_steps', str(int(train.get('logging_steps', 20))),
        '--save_steps', _format_steps_arg(train.get('save_steps', 200), 'training.train.save_steps'),
        '--save_safetensors', 'False',
        '--save_only_model', str(bool(train.get('save_only_model', True))),
        '--seed', str(seed),
        '--tokenized_path', str(effective_tokenized_path),
        '--per_device_train_batch_size', str(int(train.get('batch_size', 2))),
        '--auto_find_batch_size', str(bool(train.get('auto_find_batch_size', False))),
        '--gradient_accumulation_steps', str(int(train.get('grad_accum', 8))),
        '--learning_rate', str(train.get('learning_rate', '2e-4')),
        '--weight_decay', str(train.get('weight_decay', 0.0)),
        '--lora_rank', str(int(lora.get('rank', 16))),
        '--lora_alpha', str(int(lora.get('alpha', 32))),
        '--lora_dropout', str(float(lora.get('dropout', 0.05))),
        '--lora_target', str(lora.get('target', 'q_proj,k_proj,v_proj,o_proj')),
        '--bf16', str(use_bf16),
        '--fp16', str(use_fp16),
        '--flash_attn', str(train.get('flash_attn', 'fa2')),
        '--disable_gradient_checkpointing', str(disable_gradient_checkpointing),
        '--use_reentrant_gc', str(use_reentrant_gc),
        '--report_to', report_to,
        '--include_num_input_tokens_seen', 'true',
        '--include_effective_tokens_per_second', 'true',
    ]
    if low_cpu_mem_usage is not None:
        train_args.extend(['--low_cpu_mem_usage', str(bool(low_cpu_mem_usage))])
    # Explicit settings for frozen campaigns; legacy profiles retain their defaults.
    for key in ('optim', 'lr_scheduler_type', 'warmup_ratio', 'warmup_steps',
                'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm'):
        if train.get(key) is not None:
            train_args.extend(['--' + key, str(train[key])])
    resize_vocab = common.get('resize_vocab')
    if resize_vocab is not None:
        train_args.extend(['--resize_vocab', str(bool(resize_vocab))])
    # Enable signal-triggered final checkpointing. This is a no-op unless the process receives SIGTERM/SIGUSR1.
    train_args.extend(["--enable_jit_checkpoint", "true"])
    save_strategy = train.get('save_strategy')
    if save_strategy is not None:
        train_args.extend(['--save_strategy', str(save_strategy)])
    if bool(train.get('skip_final_save', False)):
        train_args.extend(['--skip_final_save', 'True'])
    save_total_limit = train.get('save_total_limit')
    if save_total_limit is not None:
        save_total_limit_value = int(save_total_limit)
        if save_total_limit_value > 0:
            train_args.extend(['--save_total_limit', str(save_total_limit_value)])
    if eval_enabled:
        eval_batch_size = int(train.get('batch_size', 2))
        train_args.extend([
            '--do_eval',
            '--eval_dataset', str(common.get('eval_dataset_name', common.get('dataset_name', 'c4'))),
            '--eval_strategy', str(eval_cfg.get('strategy', 'steps')),
            '--eval_steps', _format_steps_arg(eval_cfg.get('eval_steps', 100), 'training.eval.eval_steps'),
            '--per_device_eval_batch_size', str(eval_batch_size),
        ])
    else:
        train_args.extend(['--eval_strategy', 'no'])
    if bool(embedding_training_cfg.get('enabled', False)):
        additional_target = str(embedding_training_cfg.get('additional_target', 'embed_tokens,lm_head')).strip()
        if additional_target:
            train_args.extend(['--additional_target', additional_target])
    train_args.extend(
        _approach_specific_train_args(
            kind=kind,
            run_cfg=run_cfg,
            common=common,
            language_map_path=language_map_path,
            experts=experts,
        )
    )
    dist = _deep_merge(common.get("distributed", {}), run_cfg.get("distributed", {}))
    if bool(dist.get('enabled', False)):
        num_processes = int(dist.get('num_processes', 2))
        num_machines = int(dist.get('num_machines', 1))
        machine_rank = int(dist.get('machine_rank', 0))
        main_process_ip = str(dist.get('main_process_ip', '127.0.0.1'))
        main_process_port = int(dist.get('main_process_port', 29501))
        cuda_visible_devices = dist.get('cuda_visible_devices')
        if cuda_visible_devices is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        config_file = dist.get('config_file') or dist.get('accelerate_config_file')
        config_args: list[str] = []
        if config_file:
            config_args = ['--config_file', str(_abs(repo_root, str(config_file)))]
        cmd = [
            'accelerate', 'launch',
            *config_args,
            '--num_processes', str(num_processes),
            '--num_machines', str(num_machines),
            '--machine_rank', str(machine_rank),
            '--main_process_ip', main_process_ip,
            '--main_process_port', str(main_process_port),
            '--module', str(common.get('launcher_module', 'llamafactory.launcher')),
            *train_args,
        ]
    else:
        # Preserve legacy behavior unless a profile explicitly sets FORCE_TORCHRUN=0.
        # Local GPUVM profiles can opt into single-process training with
        # LLAMAFACTORY_ALLOW_NOT_DISTRIBUTED=1.
        if "FORCE_TORCHRUN" not in env:
            try:
                import torch

                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    env["FORCE_TORCHRUN"] = "1"
            except Exception:
                pass
        cmd = [
            'python3', '-m', 'llamafactory.cli', 'train',
            *train_args,
        ]
    cmd = _maybe_wrap_debugger_cmd(
        cmd,
        common=common,
        run_cfg=run_cfg,
        distributed_enabled=bool(dist.get('enabled', False)),
    )
    return cmd, env


def resolve_training_stage_config(ctx: "PipelineContext") -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    train_cfg_local = dict(ctx.train_cfg)
    if bool(ctx.wandb_cfg.get("enabled", False)):
        train_cfg_local["report_to"] = "wandb"
        train_cfg_local["train_run_prefix"] = str(ctx.wandb_cfg.get("train_prefix", "train"))

    train_cfg_for_stage = dict(train_cfg_local)
    resolved_model_path = str(train_cfg_for_stage.get("model_name_or_path", ""))
    if bool(ctx.tokenizer_extension_cfg.get("enabled", False)) and bool(ctx.embedding_init_cfg.get("enabled", False)):
        if not ctx.embedding_init_manifest_path.exists():
            raise SystemExit(
                f"Training with embedding init requires manifest at {ctx.embedding_init_manifest_path}. "
                "Run embedding_init stage first."
            )
        embedding_manifest = _load_json(ctx.embedding_init_manifest_path)
        resolved_model_path = str(embedding_manifest.get("initialized_model_dir", resolved_model_path))

    train_cfg_for_stage["model_name_or_path"] = resolved_model_path
    runs = train_cfg_local.get("runs", [])
    if not runs:
        raise SystemExit("No training runs configured in profile.training.runs")
    return train_cfg_for_stage, resolved_model_path, runs


def describe_training_run(
    ctx: "PipelineContext",
    *,
    run: dict[str, Any],
    train_cfg_for_stage: dict[str, Any],
) -> tuple[str, str, Path, Path]:
    run_id = str(run["id"])
    run_name_label = _build_run_label(
        base_run_label=ctx.base_run_label,
        run=run,
        default_num_experts=int(train_cfg_for_stage.get("num_experts", 0)),
    )
    output_dir = ctx.train_root / run_id
    log_path = ctx.logs_root / "train_runs" / run_id / f"{run_name_label}.train.log"
    return run_id, run_name_label, output_dir, log_path


def resolve_adapter_eval_path(
    run_dir: Path,
    *,
    expected_step: int | None = None,
    strict_final_checkpoint: bool = False,
    require_direct_save_marker: bool | None = None,
) -> Path:
    """Resolve the best adapter directory to evaluate for a completed training run."""
    if require_direct_save_marker is None:
        require_direct_save_marker = bool(strict_final_checkpoint)

    if strict_final_checkpoint:
        if expected_step is None:
            raise SystemExit(f"strict checkpoint resolution requires expected_step for {run_dir}")
        adapter_path = run_dir / f"checkpoint-{int(expected_step)}_adapter"
        if not adapter_path.is_dir():
            raise SystemExit(f"strict checkpoint not found for {run_dir}: {adapter_path}")
        provenance = adapter_checkpoint_provenance(
            adapter_path,
            expected_step=int(expected_step),
            require_direct_save_marker=bool(require_direct_save_marker),
        )
        if not provenance["valid"]:
            raise SystemExit(
                f"strict checkpoint rejected for {run_dir}: {adapter_path} "
                f"errors={provenance['errors']}"
            )
        return adapter_path

    best_step = -1
    best_path: Path | None = None
    if run_dir.exists():
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            match = _CHECKPOINT_ADAPTER_RE.match(child.name)
            if not match:
                continue
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = child

    if best_path is not None:
        return best_path

    if (run_dir / "adapter_config.json").exists():
        return run_dir

    raise SystemExit(
        f"could not resolve adapter to evaluate in {run_dir} "
        "(expected checkpoint-*_adapter or adapter_config.json)."
    )


def build_final_eval_cmd(
    *,
    repo_root: Path,
    env: dict[str, str],
    output_dir: Path,
    run_cfg: dict[str, Any],
    common: dict[str, Any],
    language_map_path: Path,
    tokenized_path: Path,
    embedding_training_cfg: dict[str, Any],
    run_name: str,
    model_name_or_path: str,
    adapter_name_or_path: str,
    per_device_eval_batch_size: int = 1,
    max_samples: int | None = None,
    verify_adapter: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Build an eval-only command that computes validation loss from tokenized data.

    This is used as a post-train step when training eval is disabled or unreliable near walltime.
    """
    env = dict(env)
    kind = str(run_cfg["kind"])
    seed = int(run_cfg.get("seed", common.get("seed", 42)))
    train = _deep_merge(common.get("train", {}), run_cfg.get("train", {}))
    lora = _deep_merge(common.get("lora", {}), run_cfg.get("lora", {}))

    num_experts_raw = common.get("num_experts")
    if num_experts_raw is None:
        experts = max(1, int(_count_experts(language_map_path)))
    else:
        experts = max(1, int(num_experts_raw))

    use_bf16 = bool(train.get("bf16", True))
    use_fp16 = bool(train.get("fp16", False))
    if kind in {"moelora", "adamole"}:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    effective_tokenized_path = _maybe_stage_tokenized_path(
        env=env,
        tokenized_path=tokenized_path,
        common=common,
        run_cfg=run_cfg,
    )

    if verify_adapter:
        verify_adapter_lora_hparams(
            adapter_dir=Path(str(adapter_name_or_path)),
            expected_rank=int(lora.get("rank", 16)),
            expected_alpha=int(lora.get("alpha", 32)),
            expected_dropout=float(lora.get("dropout", 0.05)),
            expected_targets=lora.get("target", "q_proj,k_proj,v_proj,o_proj"),
        )

    eval_args: list[str] = [
        "--stage",
        "sft",
        "--do_eval",
        "--model_name_or_path",
        str(model_name_or_path),
        "--adapter_name_or_path",
        str(adapter_name_or_path),
        "--dataset",
        str(common.get("dataset_name", "c4")),
        "--eval_dataset",
        str(common.get("eval_dataset_name", common.get("dataset_name", "c4"))),
        "--dataset_dir",
        str(_abs(repo_root, str(common.get("dataset_dir", "LLaMA-Factory/data")))),
        "--template",
        str(common.get("template", "llama3")),
        "--finetuning_type",
        kind,
        "--output_dir",
        str(output_dir),
        "--overwrite_output_dir",
        "--run_name",
        str(common.get("eval_run_prefix", "final_eval") + "/" + run_name),
        "--seed",
        str(seed),
        "--tokenized_path",
        str(effective_tokenized_path),
        "--per_device_eval_batch_size",
        str(int(per_device_eval_batch_size)),
        "--eval_strategy",
        "no",
        "--lora_rank",  # Match the adapter's LoRA hyperparams. Some tuners (e.g. MoE-LoRA) validate lora_rank against their expert configuration during arg parsing.
        str(int(lora.get("rank", 16))),
        "--lora_alpha",
        str(int(lora.get("alpha", 32))),
        "--lora_dropout",
        str(float(lora.get("dropout", 0.05))),
        "--lora_target",
        str(lora.get("target", "q_proj,k_proj,v_proj,o_proj")),
        "--bf16",
        str(use_bf16),
        "--fp16",
        str(use_fp16),
        "--flash_attn",
        str(train.get("flash_attn", "fa2")),
        "--report_to",
        "none",
    ]

    if max_samples is not None and int(max_samples) > 0:
        eval_args.extend(["--max_samples", str(int(max_samples))])

    if bool(embedding_training_cfg.get("enabled", False)):
        additional_target = str(embedding_training_cfg.get("additional_target", "embed_tokens,lm_head")).strip()
        if additional_target:
            eval_args.extend(["--additional_target", additional_target])

    eval_args.extend(
        _approach_specific_train_args(
            kind=kind,
            run_cfg=run_cfg,
            common=common,
            language_map_path=language_map_path,
            experts=experts,
        )
    )

    # Final loss eval is a single-GPU inference pass. Training profiles may
    # enable FSDP/accelerate for training, but accelerate with one process is
    # rejected by LLaMA-Factory's parser for these PEFT tuners.
    if "FORCE_TORCHRUN" not in env:
        env["FORCE_TORCHRUN"] = "1"
    cmd = ["python3", "-m", "llamafactory.cli", "train", *eval_args]

    return cmd, env


def run_training_variant(*, repo_root: Path, env: dict[str, str], output_dir: Path, run_cfg: dict[str, Any], common: dict[str, Any], language_map_path: Path, tokenized_path: Path, embedding_training_cfg: dict[str, Any], run_name: str, log_path: Path, dry_run: bool, executor: Executor | None = None) -> None:
    """Build and execute a single training variant."""
    cmd, run_env = build_training_cmd(
        repo_root=repo_root, env=env, output_dir=output_dir,
        run_cfg=run_cfg, common=common, language_map_path=language_map_path,
        tokenized_path=tokenized_path, embedding_training_cfg=embedding_training_cfg,
        run_name=run_name,
    )
    _run_with_tee(cmd, run_env, dry_run, log_path, executor=executor)
    if not dry_run:
        (output_dir / '.training_complete').write_text('ok\n', encoding='utf-8')
