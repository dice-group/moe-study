from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")


def _clip_score(value: float) -> float:
    return min(5.0, max(0.0, float(value)))


def _parse_prediction(text: str) -> float:
    matches = _FLOAT_RE.findall(str(text))
    if not matches:
        return 0.0
    return _clip_score(float(matches[-1]))


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    if math.isnan(value):
        return 0.0
    return value


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1
    return ranks


def process_results(doc, results) -> dict[str, tuple[float, float]]:
    gold = float(doc["label"])
    pred = _parse_prediction(results[0])
    item = (gold, pred)
    return {
        "pearson": item,
        "spearman": item,
    }


def _unzip(items: Iterable[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(items)
    if not pairs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    golds, preds = zip(*pairs, strict=True)
    return np.asarray(golds, dtype=float), np.asarray(preds, dtype=float)


def agg_pearson(items: list[tuple[float, float]]) -> float:
    golds, preds = _unzip(items)
    return _safe_corrcoef(golds, preds)


def agg_spearman(items: list[tuple[float, float]]) -> float:
    golds, preds = _unzip(items)
    if golds.size < 2 or preds.size < 2:
        return 0.0
    return _safe_corrcoef(_rankdata_average(golds), _rankdata_average(preds))
