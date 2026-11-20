"""Train, probe, and compress a small retail basket language model."""


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
