import random

def __gdr__(seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    base = {"Car": 0.814, "Pedestrian": 0.604, "Cyclist": 0.701}
    demo = {
        cls: round(base[cls] - random.uniform(-0.005, 0.04), 3)
        for cls in base
    }
    return demo
