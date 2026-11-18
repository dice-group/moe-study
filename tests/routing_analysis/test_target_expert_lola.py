from __future__ import annotations

import numpy as np
import pytest

from pipeline_tools.routing_analysis.target_expert_lola import _usage_cmap, structure_routing


def test_structure_routing_preserves_layer_target_choice_order() -> None:
    labels = [
        "L1:mlp.up_proj.router",
        "L0:self_attn.q_proj.router",
        "L1:self_attn.q_proj.router",
        "L0:mlp.up_proj.router",
    ]
    matrix = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    result = structure_routing(
        matrix,
        labels,
        target_order=("q_proj", "up_proj"),
    )

    assert result.layers == [0, 1]
    assert result.targets == ["q_proj", "up_proj"]
    assert result.matrix.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(result.matrix[:, 0, 0], matrix[:, 1])
    np.testing.assert_array_equal(result.matrix[:, 0, 1], matrix[:, 3])
    np.testing.assert_array_equal(result.matrix[:, 1, 0], matrix[:, 2])
    np.testing.assert_array_equal(result.matrix[:, 1, 1], matrix[:, 0])


def test_structure_routing_rejects_incomplete_target_grid() -> None:
    matrix = np.zeros((1, 3, 2), dtype=np.float64)
    labels = [
        "L0:self_attn.q_proj.router",
        "L0:mlp.up_proj.router",
        "L1:self_attn.q_proj.router",
    ]

    with pytest.raises(ValueError, match="incomplete"):
        structure_routing(
            matrix,
            labels,
            target_order=("q_proj", "up_proj"),
        )


def test_structure_routing_rejects_shared_global_router() -> None:
    with pytest.raises(ValueError, match="Cannot resolve"):
        structure_routing(
            np.zeros((3, 1, 8), dtype=np.float64),
            ["shared_gate"],
        )


def test_usage_cmap_matches_lola_probability_landmarks() -> None:
    cmap = _usage_cmap()
    expected = [
        (0.0, "#edf8e9"),
        (1 / 8, "#66bd63"),
        (1 / 4, "#fee08b"),
        (1 / 2, "#f46d43"),
        (1.0, "#a50026"),
    ]
    for probability, expected_hex in expected:
        actual = tuple(round(channel * 255) for channel in cmap(float(probability))[:3])
        target = tuple(int(expected_hex[index : index + 2], 16) for index in (1, 3, 5))
        assert all(abs(left - right) <= 2 for left, right in zip(actual, target))
