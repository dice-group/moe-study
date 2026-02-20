#!/usr/bin/env python3
"""Training helpers for proof pipeline."""
from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any
from .runtime import _abs, _count_experts, _deep_merge, _run_with_tee

if TYPE_CHECKING:
    from .executor import Executor

def _lpr_args(cfg: dict[str, Any], common: dict[str, Any], language_map_path: Path, experts: int) -> list[str]:
    """Build the shared Language Prior Routing CLI args used by both CoLA and HydraLoRA."""
    use_experts = bool(cfg.get('use_experts', True))
    return [
        '--language_column', str(common.get('language_column', 'language')),
        '--language_map', str(language_map_path),
        '--language_router_mode', str(cfg.get('router_mode', 'learned')),
        '--language_head_router_mode', str(cfg.get('head_router_mode', 'learned')),
        '--language_prior_weight', str(float(cfg.get('prior_weight', 0.1))),
        '--language_bias_value', str(float(cfg.get('bias_value', 0.0))),
        '--language_head_bias_value', str(float(cfg.get('head_bias_value', 0.0))),
        '--language_guidance_scope', str(cfg.get('guidance_scope', 'all')),
    ]


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
    experts = max(1, int(common.get('num_experts', _count_experts(language_map_path))))
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
        '--save_steps', str(int(train.get('save_steps', 200))),
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
        '--disable_gradient_checkpointing', str(bool(train.get('disable_gradient_checkpointing', False))),
        '--use_reentrant_gc', str(bool(train.get('use_reentrant_gc', True))),
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
        train_args.extend([
            '--do_eval',
            '--eval_dataset', str(common.get('eval_dataset_name', common.get('dataset_name', 'c4'))),
            '--eval_strategy', str(eval_cfg.get('strategy', 'steps')),
            '--eval_steps', str(int(eval_cfg.get('eval_steps', 100))),
            '--per_device_eval_batch_size', str(int(eval_cfg.get('batch_size', train.get('batch_size', 2)))),
        ])
    else:
        train_args.extend(['--eval_strategy', 'no'])
    if bool(embedding_training_cfg.get('enabled', False)):
        additional_target = str(embedding_training_cfg.get('additional_target', 'embed_tokens,lm_head')).strip()
        if additional_target:
            train_args.extend(['--additional_target', additional_target])
    if kind == 'cola':
        cfg = run_cfg.get('params', {})
        train_args.extend([
            '--num_A', str(int(cfg.get('num_A', 1))),
            '--num_B', str(int(cfg.get('num_B', 3))),
            '--cola_strategy', str(cfg.get('strategy', 'fully')),
            '--use_cola_experts', str(bool(cfg.get('use_experts', True))),
            '--cola_num_experts', str(int(cfg.get('num_experts', experts if cfg.get('use_experts', True) else 1))),
            '--cola_top_k', str(int(cfg.get('top_k', 1))),
            *_lpr_args(cfg, common, language_map_path, experts),
        ])
    elif kind == 'hydralora':
        cfg = run_cfg.get('params', {})
        expert_lora_nums = cfg.get('expert_lora_nums')
        if isinstance(expert_lora_nums, list):
            expert_lora_nums_arg = ",".join(str(int(x)) for x in expert_lora_nums)
        elif expert_lora_nums is None:
            expert_lora_nums_arg = ""
        else:
            expert_lora_nums_arg = str(expert_lora_nums)
        train_args.extend([
            '--lora_num', str(int(cfg.get('lora_num', 3))),
            '--use_hydralora_experts', str(bool(cfg.get('use_experts', True))),
            '--hydralora_num_experts', str(int(cfg.get('num_experts', experts if cfg.get('use_experts', True) else 1))),
            '--hydralora_top_k', str(int(cfg.get('top_k', 1))),
            *_lpr_args(cfg, common, language_map_path, experts),
        ])
        if expert_lora_nums_arg.strip():
            train_args.extend(['--hydralora_expert_lora_nums', expert_lora_nums_arg])
    elif kind == 'adamole':
        cfg = run_cfg.get('params', {})
        train_args.extend([
            '--adamole_num_experts', str(int(cfg.get('num_experts', 4))),
            '--adamole_aux_loss_coef', str(float(cfg.get('aux_loss_coef', 0.0))),
        ])
        if cfg.get('max_threshold') is not None:
            train_args.extend(['--adamole_max_threshold', str(float(cfg['max_threshold']))])
        if bool(cfg.get('debug_mode', False)):
            train_args.extend(['--adamole_debug_mode', 'True'])
    elif kind == 'mola':
        cfg = run_cfg.get('params', {})
        train_args.extend([
            '--mola_num_experts', str(int(cfg.get('num_experts', 4))),
            '--mola_top_k', str(int(cfg.get('top_k', 2))),
            '--mola_router_aux_loss_coef', str(float(cfg.get('router_aux_loss_coef', 0.01))),
        ])
        if int(cfg.get('num_null_experts', 0)) > 0:
            train_args.extend(['--mola_num_null_experts', str(int(cfg['num_null_experts']))])
        if bool(cfg.get('output_router_logits', False)):
            train_args.extend(['--mola_output_router_logits', 'True'])
        if bool(cfg.get('aux_loss_annealing', False)):
            train_args.extend(['--mola_aux_loss_annealing', 'True'])
        if cfg.get('aux_loss_coef_end') is not None:
            train_args.extend(['--mola_aux_loss_coef_end', str(float(cfg['aux_loss_coef_end']))])
        if bool(cfg.get('debug_mode', False)):
            train_args.extend(['--mola_debug_mode', 'True'])
    elif kind == 'moelpr':
        cfg = run_cfg.get('params', {})
        train_args.extend([
            '--moelpr_stage', str(int(cfg.get('stage', 1))),
            '--moelpr_num_experts', str(int(cfg.get('num_experts', 4))),
            '--moelpr_top_k', str(int(cfg.get('top_k', 1))),
            '--moelpr_aux_loss_coef', str(float(cfg.get('aux_loss_coef', 0.01))),
            '--moelpr_lpr_loss_coef', str(float(cfg.get('lpr_loss_coef', 0.1))),
        ])
        if cfg.get('layers_to_transform') is not None:
            train_args.extend(['--moelpr_layers_to_transform', str(cfg['layers_to_transform'])])
        if cfg.get('original_language') is not None:
            train_args.extend(['--moelpr_original_language', str(cfg['original_language'])])
        if bool(cfg.get('debug_mode', False)):
            train_args.extend(['--moelpr_debug_mode', 'True'])
    elif kind == 'moelora':
        cfg = run_cfg.get('params', {})
        train_args.extend([
            '--moelora_num_experts', str(int(cfg.get('num_experts', 8))),
            '--moelora_task_num', str(int(cfg.get('task_num', 1))),
            '--moelora_task_embedding_dim', str(int(cfg.get('task_embedding_dim', 64))),
            '--moelora_sparse_top_k', str(int(cfg.get('sparse_top_k', 0))),
            '--moelora_gate_temperature', str(float(cfg.get('gate_temperature', 1.0))),
        ])
    elif kind == 'movlora':
        cfg = run_cfg.get('params', {})
        train_args.extend([
            '--movlora_num_experts', str(int(cfg.get('num_experts', 8))),
            '--movlora_top_k', str(int(cfg.get('top_k', 0))),
            '--movlora_router_temperature', str(float(cfg.get('router_temperature', 1.0))),
            '--movlora_router_jitter_noise', str(float(cfg.get('router_jitter_noise', 0.0))),
            '--movlora_router_bias', str(bool(cfg.get('router_bias', False))),
            '--movlora_router_init_std', str(float(cfg.get('router_init_std', 0.02))),
            '--movlora_router_ignore_padding_tokens', str(bool(cfg.get('router_ignore_padding_tokens', False))),
        ])
    elif kind != 'lora':
        raise SystemExit(f'Unsupported finetuning kind: {kind}')
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
