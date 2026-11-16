from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from pipeline_tools.routing_analysis.cli import _find_moelora_parent, combine_partials, save_npz
from pipeline_tools.routing_analysis.compare import align_and_compare
from pipeline_tools.routing_analysis.collectors import RoutingCollector, build_matrices
from pipeline_tools.routing_analysis.data import validate_language_sample
from pipeline_tools.routing_analysis.static import (
    extract_moelora,
    extract_mtllora,
    language_id_map,
)


def test_moelora_parent_selection_ignores_attribute_proxy() -> None:
    class Router(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_task_embedding = torch.nn.ModuleDict({"default": torch.nn.Embedding(2, 2)})
            self.lora_gate = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})

        def get_routing_weights(self):
            return None

    class Proxy(torch.nn.Module):
        def __init__(self, router: Router) -> None:
            super().__init__()
            self.base_model = router

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.base_model, name)

    router = Router()
    proxy = Proxy(router)
    assert hasattr(proxy, "get_routing_weights")
    assert _find_moelora_parent(proxy) is router


def test_padding_mask_excludes_padded_router_rows() -> None:
    collector = RoutingCollector("HYDRALORA", {"use_hydralora_experts": False, "lora_num": 2})
    collector.set_language("english")
    collector.set_token_mask(torch.tensor([[1, 1, 0], [1, 0, 0]]))
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [100.0, 0.0]],
            [[0.0, 4.0], [100.0, 0.0], [100.0, 0.0]],
        ]
    )
    collector._record_prob_logits("L0:q_proj", logits)
    record = collector.by_language["english"].layers["L0:q_proj"]
    assert record.tokens == 3
    assert record.assignments == 3
    assert record.counts.tolist() == [1.0, 2.0]


def test_build_matrices_has_canonical_probability_rows() -> None:
    collector = RoutingCollector("HYDRALORA", {"use_hydralora_experts": False, "lora_num": 2})
    collector.set_language("english")
    collector._record_prob_logits("L0:q_proj", torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    payload = build_matrices(collector, ["english"])
    assert payload["routing_basis"] == "token_conditioned"
    np.testing.assert_allclose(payload["routing_probability"].sum(axis=2), 1.0)


def test_mtllora_static_extraction_uses_temperature_and_language_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B_weights": torch.tensor(
                [[0.0, 1.0], [2.0, 0.0], [0.5, 0.5]]
            ),
            "base_model.model.model.layers.1.mlp.up_proj.lora_B_weights": torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]]
            ),
        },
        checkpoint / "adapter_model.safetensors",
    )
    adapter_config = {
        "task_num": 3,
        "num_up_projections": 2,
        "temperature": 0.5,
        "language_list": ["b", "a", "english"],
    }
    payload = extract_mtllora(
        checkpoint,
        adapter_config,
        ["english", "a", "b"],
        ["b", "a", "english"],
    )
    assert payload["routing_probability"].shape == (3, 2, 2)
    assert list(payload["layer_labels"]) == ["L0:self_attn.q_proj", "L1:mlp.up_proj"]
    np.testing.assert_allclose(payload["routing_probability"].sum(axis=2), 1.0)
    expected = torch.softmax(torch.tensor([0.0, 1.0]) / 0.5, dim=-1).numpy()
    np.testing.assert_allclose(payload["routing_probability"][2, 0], expected, atol=1e-7)


def test_moelora_static_extraction_matches_sparse_runtime_formula(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gate = torch.tensor([[4.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    save_file(
        {
            "base_model.lora_task_embedding.weight": embedding,
            "base_model.lora_gate.weight": gate,
        },
        checkpoint / "adapter_model.safetensors",
    )
    payload = extract_moelora(
        checkpoint,
        {"task_num": 2, "expert_num": 3, "gate_top_k": 2, "gate_temperature": 1.0},
        ["b", "a"],
        ["a", "b"],
    )
    assert payload["routing_probability"].shape == (2, 1, 3)
    assert np.count_nonzero(payload["routing_probability"][0, 0]) == 2
    np.testing.assert_allclose(payload["routing_probability"].sum(axis=2), 1.0)


def test_language_id_map_fails_closed_on_checkpoint_mismatch() -> None:
    with pytest.raises(ValueError, match="language_list mismatch"):
        language_id_map(
            {"task_num": 2, "language_list": ["a", "b"]},
            ["a", "english"],
        )


def test_sample_manifest_validation_fails_on_digest_mismatch() -> None:
    provenance = {"language": "a", "documents": 2, "text_sha256": "observed"}
    manifest = {
        "languages": {
            "a": {"language": "a", "documents": 2, "text_sha256": "expected"}
        }
    }
    with pytest.raises(ValueError, match="sample manifest mismatch"):
        validate_language_sample(
            provenance=provenance,
            manifest=manifest,
            expected_documents=2,
            strict=True,
        )


def test_partial_combination_preserves_language_axis(tmp_path: Path) -> None:
    paths = []
    for index, language in enumerate(["a", "b"]):
        payload = {
            "routing_probability": np.full((1, 1, 2), index + 1.0),
            "routing_weight_normalized": np.full((1, 1, 2), index + 1.0),
            "routing_count_normalized": np.full((1, 1, 2), index + 1.0),
            "tokens": np.array([[index + 2]]),
            "layer_labels": np.array(["L0:q_proj"], dtype=object),
            "choice_labels": np.array(["head_0", "head_1"], dtype=object),
            "axis_name": "head",
            "routing_basis": "token_conditioned",
        }
        path = tmp_path / f"{language}.npz"
        save_npz(path, payload, languages=[language], run_id="run", peft_type="HYDRALORA")
        paths.append(path)
    combined = combine_partials(paths, ["a", "b"])
    assert combined["routing_probability"].shape == (2, 1, 2)
    assert combined["tokens"].tolist() == [[2], [3]]


def test_cross_panel_alignment_recovers_permuted_choices() -> None:
    left_matrix = np.array(
        [
            [[0.8, 0.1, 0.1]],
            [[0.1, 0.8, 0.1]],
            [[0.1, 0.1, 0.8]],
        ]
    )
    permutation = [2, 0, 1]
    right_matrix = left_matrix[:, :, permutation]
    common = {
        "languages": ["a", "b", "english"],
        "layer_labels": ["L0:q_proj"],
        "choice_labels": ["choice_0", "choice_1", "choice_2"],
        "axis_name": "head",
    }
    stability, alignments = align_and_compare(
        "run",
        {**common, "routing_probability": left_matrix},
        {**common, "routing_probability": right_matrix},
    )
    assert len(alignments) == 3
    assert all(row["aligned_cosine_mean"] == pytest.approx(1.0) for row in stability)
    assert all(row["jensen_shannon_distance_mean"] == pytest.approx(0.0) for row in stability)
