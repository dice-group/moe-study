"""CPU checks for explicit training recipes and unchanged launcher defaults."""

from pipeline.training import build_training_cmd


RECIPE = {
    "optim": "adamw_torch",
    "lr_scheduler_type": "constant",
    "warmup_ratio": 0.0,
    "warmup_steps": 0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
}


def command(tmp_path, *, train=None, run_train=None, launcher=None):
    common = {
        "model_name_or_path": "fixture-model",
        "num_experts": 1,
        "distributed": {"enabled": True, "num_processes": 4},
        "train": train or {},
    }
    if launcher is not None:
        common["launcher_module"] = launcher
    return build_training_cmd(
        repo_root=tmp_path,
        env={},
        output_dir=tmp_path / "output",
        run_cfg={"kind": "lora", "train": run_train or {}},
        common=common,
        language_map_path=tmp_path / "languages.json",
        tokenized_path=tmp_path / "tokens",
        embedding_training_cfg={},
        run_name="fixture",
    )[0]


def test_legacy_recipe_omits_optional_settings(tmp_path):
    cmd = command(tmp_path)
    assert cmd[cmd.index("--module") + 1] == "llamafactory.launcher"
    for key in RECIPE:
        assert f"--{key}" not in cmd


def test_explicit_recipe_preserves_values_including_zero(tmp_path):
    cmd = command(tmp_path, train=RECIPE)
    for key, value in RECIPE.items():
        assert cmd.count(f"--{key}") == 1
        assert cmd[cmd.index(f"--{key}") + 1] == str(value)


def test_run_recipe_overrides_profile_and_none_is_omitted(tmp_path):
    cmd = command(tmp_path, train=RECIPE, run_train={"adam_beta2": 0.99, "warmup_ratio": None})
    assert cmd[cmd.index("--adam_beta2") + 1] == "0.99"
    assert "--warmup_ratio" not in cmd


def test_custom_distributed_launcher_keeps_recipe(tmp_path):
    default = command(tmp_path, train=RECIPE)
    custom = command(tmp_path, train=RECIPE, launcher="experiment.launcher")
    default[default.index("--module") + 1] = "experiment.launcher"
    assert custom == default
