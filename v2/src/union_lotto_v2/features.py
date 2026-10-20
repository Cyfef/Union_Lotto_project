from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

from .data import Draw

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


@dataclass(frozen=True)
class ShapeFeatures:
    total: int
    span: int
    minimum: int
    maximum: int
    odd_count: int
    big_count: int
    prime_count: int
    zone1: int
    zone2: int
    zone3: int
    mod0: int
    mod1: int
    mod2: int
    consecutive_pairs: int
    repeated_from_previous: int
    duplicate_tails: int

    def to_dict(self) -> dict:
        return asdict(self)


def shape_features(reds: tuple[int, ...] | list[int], previous: set[int] | None = None) -> ShapeFeatures:
    values = sorted(reds)
    if len(values) != 6 or len(set(values)) != 6:
        raise ValueError("shape_features 需要 6 个不重复红球")
    zones = [sum(1 for n in values if n <= 11), sum(1 for n in values if 12 <= n <= 22), sum(1 for n in values if n >= 23)]
    mods = [sum(1 for n in values if n % 3 == r) for r in range(3)]
    tails = Counter(n % 10 for n in values)
    return ShapeFeatures(
        total=sum(values), span=values[-1] - values[0], minimum=values[0], maximum=values[-1],
        odd_count=sum(n % 2 for n in values), big_count=sum(n >= 17 for n in values),
        prime_count=sum(n in PRIMES for n in values), zone1=zones[0], zone2=zones[1], zone3=zones[2],
        mod0=mods[0], mod1=mods[1], mod2=mods[2],
        consecutive_pairs=sum(b == a + 1 for a, b in zip(values, values[1:])),
        repeated_from_previous=len(set(values) & (previous or set())),
        duplicate_tails=sum(count - 1 for count in tails.values() if count > 1),
    )


def ball_feature_vector(history: list[Draw], number: int, color: str) -> list[float]:
    pool, picks = (33, 6) if color == "red" else (16, 1)
    base = picks / pool
    sets = [set(draw.reds) if color == "red" else {draw.blue} for draw in history]
    features = [1.0]
    for window in (5, 15, 50, 150):
        recent = sets[-window:]
        observed = sum(number in chosen for chosen in recent) / max(len(recent), 1)
        features.append(observed / base - 1.0)
    gap = 0
    for chosen in reversed(sets):
        if number in chosen:
            break
        gap += 1
    expected_gap = (1.0 - base) / base
    features.extend([
        min(gap / max(expected_gap, 1.0), 5.0) / 5.0,
        float(bool(sets and number in sets[-1])),
        (number - 1) / (pool - 1) - 0.5,
        float(number % 2),
        float(number >= (17 if color == "red" else 9)),
        float(number in PRIMES),
        float(number % 3 == 0),
        float(number % 3 == 1),
    ])
    return features


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))

