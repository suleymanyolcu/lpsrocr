from __future__ import annotations

import os
import random


def make_rng(seed: int, salt: str = "") -> random.Random:
    key = f"{seed}:{salt}" if salt else str(seed)
    return random.Random(key)


def seed_python(seed: int) -> random.Random:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    return random.Random(seed)

