"""Train, probe, and compress a small retail basket language model."""


from io import BytesIO

import numpy as np


THEMES = (
    ("Bread", "Butter", "Jam", "Eggs", "Milk", "Yogurt", "Oats", "Honey"),
    ("Pasta", "Tomato Sauce", "Parmesan", "Basil", "Olive Oil", "Garlic", "Onion", "Mushrooms"),
    ("Tortillas", "Black Beans", "Salsa", "Avocado", "Lime", "Cheddar", "Rice", "Corn"),
    ("Coffee", "Cream", "Sugar", "Biscuits", "Cocoa", "Tea", "Lemon", "Muffins"),
)
CATALOG = tuple(sorted({item for theme in THEMES for item in theme}))
ITEM_ID = {item: idx for idx, item in enumerate(CATALOG)}


def _baskets(rng, count, target_store):
    baskets = []
    for _ in range(count):
        theme = THEMES[rng.integers(len(THEMES))]
        pool = theme[:5] if target_store and rng.random() < 0.8 else theme
        basket = rng.choice(pool, size=rng.integers(3, 6), replace=False).tolist()
        if rng.random() < 0.15:
            basket.append(rng.choice(sorted(set(CATALOG) - set(basket))).item())
        baskets.append(tuple(basket))
    return baskets


def milestone_1(seed=42, sizes=(800, 1000, 200, 300)):
    """Generate source pretraining, target training, calibration, and test splits."""
    streams = np.random.SeedSequence(seed).spawn(4)
    names = ("source", "train", "calibration", "test")
    return {
        name: _baskets(np.random.default_rng(stream), count, target_store=name != "source")
        for name, stream, count in zip(names, streams, sizes, strict=True)
    }


def _examples(baskets):
    contexts, targets = [], []
    for basket in baskets:
        ids = [ITEM_ID[item] for item in basket]
        for end in range(1, len(ids)):
            context = np.zeros(len(CATALOG), dtype=np.float32)
            context[ids[:end]] = 1 / end
            contexts.append(context)
            targets.append(ids[end])
    return np.stack(contexts), np.asarray(targets)


def _initialize(rng, hidden=128):
    return {
        "w1": rng.normal(0, 0.15, (len(CATALOG), hidden)).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": rng.normal(0, 0.10, (hidden, len(CATALOG))).astype(np.float32),
        "b2": np.zeros(len(CATALOG), dtype=np.float32),
    }


def _softmax(logits):
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def _fit(model, baskets, rng, epochs, learning_rate):
    contexts, targets = _examples(baskets)
    for _ in range(epochs):
        order = rng.permutation(len(targets))
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            inputs = contexts[batch]
            hidden = np.tanh(inputs @ model["w1"] + model["b1"])
            gradient = _softmax(hidden @ model["w2"] + model["b2"])
            gradient[np.arange(len(batch)), targets[batch]] -= 1
            gradient /= len(batch)
            hidden_gradient = (gradient @ model["w2"].T) * (1 - hidden ** 2)
            model["w2"] -= learning_rate * (hidden.T @ gradient)
            model["b2"] -= learning_rate * gradient.sum(axis=0)
            model["w1"] -= learning_rate * (inputs.T @ hidden_gradient)
            model["b1"] -= learning_rate * hidden_gradient.sum(axis=0)


def milestone_2(splits, seed=42, source_epochs=12, target_epochs=12):
    """Pretrain on source baskets, then continue from that checkpoint on target baskets."""
    rng = np.random.default_rng(seed)
    source = _initialize(rng)
    _fit(source, splits["source"], rng, source_epochs, 0.45)
    adapted = {name: value.copy() for name, value in source.items()}
    _fit(adapted, splits["train"], rng, target_epochs, 0.25)
    return source, adapted


def _logits(model, contexts):
    if "w1_scale" in model:
        hidden = np.tanh(_integer_linear(contexts, model, "w1", "b1", "input_scale"))
        return _integer_linear(hidden, model, "w2", "b2", "hidden_scale")
    hidden = np.tanh(contexts @ model["w1"] + model["b1"])
    return hidden @ model["w2"] + model["b2"]


def _exclude_selected(logits, contexts):
    return np.where(contexts > 0, -np.inf, logits)


def milestone_3(model, basket, training_baskets, top_k=5):
    """Rank unseen items and calculate empirical support for the selected itemset."""
    items = set(basket)
    if not items or items - ITEM_ID.keys():
        raise ValueError("Select one or more catalog items.")
    if top_k < 1 or not training_baskets:
        raise ValueError("Provide a positive recommendation count and training baskets.")
    context = np.zeros((1, len(CATALOG)), dtype=np.float32)
    context[0, [ITEM_ID[item] for item in items]] = 1 / len(items)
    remaining = len(CATALOG) - len(items)
    recommendations = []
    if remaining:
        scores = _softmax(_exclude_selected(_logits(model, context), context))[0]
        order = np.argsort(-scores, kind="stable")[:min(top_k, remaining)]
        recommendations = [
            {"item": CATALOG[index], "probability": float(scores[index])} for index in order
        ]
    matches = sum(items.issubset(row) for row in training_baskets)
    for entry in recommendations:
        expanded = items | {entry["item"]}
        joint = sum(expanded.issubset(row) for row in training_baskets)
        entry["confidence"] = joint / matches if matches else None
    return {
        "recommendations": recommendations,
        "matches": matches,
        "support": matches / len(training_baskets),
    }


def milestone_4(model, test_baskets):
    """Measure uncompressed checkpoint bytes and held-out next-item accuracy."""
    contexts, targets = _examples(test_baskets)
    logits = _exclude_selected(_logits(model, contexts), contexts)
    ranked = np.argsort(-logits, axis=1, kind="stable")
    return {
        "bytes": len(checkpoint_bytes(model)),
        "top_1": float(np.mean(ranked[:, 0] == targets)),
        "top_5": float(np.mean(np.any(ranked[:, :5] == targets[:, None], axis=1))),
        "examples": len(targets),
    }


def _scale(values, axis=None, keepdims=False):
    peak = np.max(np.abs(values), axis=axis, keepdims=keepdims)
    return np.maximum(peak / np.float32(127), np.finfo(np.float32).tiny)


def _int8(values, scale):
    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8)


def _integer_linear(values, model, weight, bias, activation_key):
    activation_scale = (
        model[activation_key] if activation_key in model
        else _scale(values, axis=1, keepdims=True)
    )
    quantized_input = _int8(values, activation_scale)
    accumulated = np.matmul(quantized_input, model[weight], dtype=np.int32)
    return (
        accumulated.astype(np.float32) * activation_scale * model[weight + "_scale"]
        + model[bias]
    )


def milestone_5(model, calibration_baskets):
    """Build dynamic and calibrated static INT8 dense layers from the FP32 model."""
    dynamic = {name: model[name].copy() for name in ("b1", "b2")}
    for weight in ("w1", "w2"):
        scale = _scale(model[weight], axis=0)
        dynamic[weight + "_scale"] = scale
        dynamic[weight] = _int8(model[weight], scale)
    contexts, _ = _examples(calibration_baskets)
    # Calibrate the second layer using the same quantized first layer used at inference.
    static = dict(dynamic, input_scale=_scale(contexts))
    hidden = np.tanh(_integer_linear(contexts, static, "w1", "b1", "input_scale"))
    static["hidden_scale"] = _scale(hidden)
    return {"Dynamic INT8": dynamic, "Static INT8": static}


def checkpoint_bytes(model):
    """Serialize a model to an uncompressed NPZ without pickle objects."""
    checkpoint = BytesIO()
    np.savez(checkpoint, **model)
    return checkpoint.getvalue()


def milestone_6(baseline, quantized_models, test_baskets):
    """Compare checkpoint size and accuracy using identical held-out targets."""
    results = {"FP32": baseline}
    results.update(
        (name, milestone_4(model, test_baskets)) for name, model in quantized_models.items()
    )
    return {
        name: {
            **metrics,
            "reduction_pct": 100 * (1 - metrics["bytes"] / baseline["bytes"]),
            "top_1_delta_pp": 100 * (metrics["top_1"] - baseline["top_1"]),
            "top_5_delta_pp": 100 * (metrics["top_5"] - baseline["top_5"]),
        }
        for name, metrics in results.items()
    }
