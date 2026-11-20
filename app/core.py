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
