"""Exercise the six milestones and the Streamlit user flow."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app.core import (
    CATALOG,
    checkpoint_bytes,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    milestone_6,
)


@pytest.fixture(scope="module")
def experiment():
    splits = milestone_1()
    source, adapted = milestone_2(splits)
    variants = milestone_5(adapted, splits["calibration"])
    return splits, source, adapted, variants


def test_milestone_1():
    splits = milestone_1()
    assert splits == milestone_1()
    assert splits != milestone_1(seed=43)
    assert sorted(map(len, splits.values())) == [200, 300, 800, 1000]
    for baskets in splits.values():
        assert all(3 <= len(row) == len(set(row)) <= 6 and set(row) <= set(CATALOG) for row in baskets)


def test_milestone_2(experiment):
    splits, source, adapted, _ = experiment
    assert all(weight.dtype == np.float32 for weight in adapted.values())
    assert not np.array_equal(source["w2"], adapted["w2"])
    assert not np.shares_memory(source["w2"], adapted["w2"])
    assert milestone_4(adapted, splits["test"])["top_1"] > milestone_4(source, splits["test"])["top_1"]


def test_milestone_3(experiment):
    _, _, adapted, _ = experiment
    rows = [("Bread", "Butter"), ("Bread", "Jam"), ("Jam", "Milk")]
    result = milestone_3(adapted, ["Bread", "Bread"], rows, top_k=len(CATALOG))
    assert result["support"] == 2 / 3
    scores = {row["item"]: row for row in result["recommendations"]}
    assert "Bread" not in scores
    assert scores["Butter"]["confidence"] == 0.5
    assert sum(row["probability"] for row in scores.values()) == pytest.approx(1)
    assert milestone_3(adapted, CATALOG, rows)["recommendations"] == []
    with pytest.raises(ValueError):
        milestone_3(adapted, ["Unknown"], rows)


def test_milestone_4(experiment):
    splits, _, adapted, _ = experiment
    metrics = milestone_4(adapted, splits["test"])
    assert metrics["bytes"] == len(checkpoint_bytes(adapted))
    assert metrics["examples"] == sum(len(row) - 1 for row in splits["test"])
    assert 0.30 <= metrics["top_1"] <= metrics["top_5"] <= 1


def test_milestone_5(experiment):
    splits, _, _, variants = experiment
    dynamic, static = variants["Dynamic INT8"], variants["Static INT8"]
    for model in (dynamic, static):
        assert model["w1"].dtype == model["w2"].dtype == np.int8
        with np.load(BytesIO(checkpoint_bytes(model)), allow_pickle=False) as restored:
            assert milestone_4(dict(restored), splits["test"]) == milestone_4(model, splits["test"])
    assert "input_scale" not in dynamic
    assert static["input_scale"] > 0 and static["hidden_scale"] > 0


def test_milestone_6(experiment):
    splits, _, adapted, variants = experiment
    baseline = milestone_4(adapted, splits["test"])
    results = milestone_6(baseline, variants, splits["test"])
    assert results["FP32"]["reduction_pct"] == 0
    for name in variants:
        assert results[name]["reduction_pct"] >= 60
        assert abs(results[name]["top_1_delta_pp"]) < 2
        assert abs(results[name]["top_5_delta_pp"]) < 2


def test_streamlit_recommendations_and_model_switch():
    page = AppTest.from_file(Path(__file__).resolve().parents[1] / "app/main.py").run(timeout=30)
    assert not page.exception
    assert page.table[0].value.shape == (5, 3)
    assert page.table[1].value.shape == (3, 7)
    page.selectbox[0].select("Static INT8").run()
    assert not page.exception
    page.multiselect[0].set_value([]).run()
    assert not page.exception
    assert "Select at least one item" in page.info[0].value
