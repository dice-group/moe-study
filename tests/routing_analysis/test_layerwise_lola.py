from __future__ import annotations

import numpy as np
import pytest

from pipeline_tools.routing_analysis.layerwise_lola import resolve_layer_views


def test_layer_views_average_captured_modules_within_base_layer() -> None:
    matrix = np.array(
        [
            [
                [0.8, 0.2],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.3, 0.7],
            ],
            [
                [0.4, 0.6],
                [0.2, 0.8],
                [0.9, 0.1],
                [0.7, 0.3],
            ],
        ]
    )
    views = resolve_layer_views(
        matrix,
        ["L0:q_proj", "L0:k_proj", "L1:q_proj", "L1:k_proj"],
    )
    assert views.scope == "base_layer_aggregate"
    assert views.labels == ["L0", "L1"]
    assert views.source_modules == 4
    np.testing.assert_allclose(
        views.matrix,
        np.array(
            [
                [[0.7, 0.3], [0.2, 0.8]],
                [[0.3, 0.7], [0.8, 0.2]],
            ]
        ),
    )


def test_layer_views_keep_shared_router_global() -> None:
    matrix = np.array(
        [
            [[0.8, 0.2]],
            [[0.3, 0.7]],
        ]
    )
    views = resolve_layer_views(matrix, ["language_conditioned_router"])
    assert views.scope == "global_shared"
    assert views.labels == ["global_shared"]
    assert views.source_modules == 1
    np.testing.assert_array_equal(views.matrix, matrix)


def test_layer_views_reject_mixed_scoped_and_unscoped_modules() -> None:
    matrix = np.full((2, 2, 3), 1.0 / 3.0)
    with pytest.raises(ValueError, match="mix base-layer-prefixed and unscoped"):
        resolve_layer_views(matrix, ["L0:q_proj", "global_router"])
