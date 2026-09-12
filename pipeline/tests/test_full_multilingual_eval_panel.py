from __future__ import annotations

from pipeline_tools.build_full_multilingual_eval_panel import _write_task_alias


def test_task_alias_preserves_canonical_split_configuration(tmp_path):
    canonical = tmp_path / "belebele_zho_Hans.yaml"
    alias = tmp_path / "belebele_cmn_Hani.yaml"
    canonical.write_text(
        "\n".join(
            [
                "dataset_name: zho_Hans",
                "fewshot_split: test",
                "include: _default_template_yaml",
                "task: belebele_zho_Hans",
                "test_split: test",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _write_task_alias(canonical, alias, "belebele_cmn_Hani")

    text = alias.read_text(encoding="utf-8")
    assert "task: belebele_cmn_Hani" in text
    assert "dataset_name: zho_Hans" in text
    assert "fewshot_split: test" in text
    assert "test_split: test" in text
