#!/usr/bin/env python3
"""Training helpers for proof pipeline."""
from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from .runtime import _abs, _count_experts, _deep_merge, _run_with_tee

if TYPE_CHECKING:
    from .executor import Executor

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
            '--moelora_num_experts', str(int(_cfg_value(cfg, 'moelora_num_experts', 8, 'num_experts'))),
            '--moelora_task_num', str(int(_cfg_value(cfg, 'moelora_task_num', 1, 'task_num'))),
            '--moelora_task_embedding_dim', str(int(_cfg_value(cfg, 'moelora_task_embedding_dim', 64, 'task_embedding_dim'))),
            '--moelora_sparse_top_k', str(int(_cfg_value(cfg, 'moelora_sparse_top_k', 0, 'sparse_top_k'))),
            '--moelora_gate_temperature', str(float(_cfg_value(cfg, 'moelora_gate_temperature', 1.0, 'gate_temperature'))),
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
    # AdaMoLE auxiliary losses are incompatible with re-entrant checkpointing.
    if kind == 'adamole' and (not disable_gradient_checkpointing) and use_reentrant_gc:
        use_reentrant_gc = False
    use_bf16 = bool(train.get('bf16', True))
    use_fp16 = bool(train.get('fp16', False))
    try:
        import torch
        if not torch.cuda.is_available() and (use_bf16 or use_fp16):
            use_bf16 = False
            use_fp16 = False
    except Exception:
        pass
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
        '--save_only_model', 'True',
        '--seed', str(seed),
        '--tokenized_path', str(tokenized_path),
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
        '--include_effective_tokens_per_second', 'true',
        '--include_num_input_tokens_seen', 'true',
    ]
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
    dist = common.get('distributed', {})
    if bool(dist.get('enabled', False)):
        num_processes = int(dist.get('num_processes', 2))
        num_machines = int(dist.get('num_machines', 1))
        machine_rank = int(dist.get('machine_rank', 0))
        main_process_ip = str(dist.get('main_process_ip', '127.0.0.1'))
        main_process_port = int(dist.get('main_process_port', 29501))
        cuda_visible_devices = dist.get('cuda_visible_devices')
        if cuda_visible_devices is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices)
        cmd = [
            'accelerate', 'launch',
            '--num_processes', str(num_processes),
            '--num_machines', str(num_machines),
            '--machine_rank', str(machine_rank),
            '--main_process_ip', main_process_ip,
            '--main_process_port', str(main_process_port),
            '--module', 'llamafactory.launcher',
            *train_args,
        ]
    else:
        cmd = [
            'python3', '-m', 'llamafactory.cli', 'train',
            *train_args,
        ]
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
