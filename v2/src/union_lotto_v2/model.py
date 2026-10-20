from __future__ import annotations

import bisect
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Draw, next_draw_date, next_issue
from .features import ShapeFeatures, shape_features, sigmoid

WARMUP = 150
NUMERIC_SHAPES = ("total", "span", "minimum", "maximum")
CATEGORICAL_SHAPES = (
    "odd_count", "big_count", "prime_count", "zone1", "zone2", "zone3",
    "mod0", "mod1", "mod2", "consecutive_pairs", "repeated_from_previous", "duplicate_tails",
)


@dataclass(frozen=True)
class LogisticConfig:
    learning_rate: float
    l2: float
    epochs: int = 7


def _occurrences(draws: list[Draw], color: str) -> list[list[int]]:
    pool = 33 if color == "red" else 16
    values = [[] for _ in range(pool + 1)]
    for index, draw in enumerate(draws):
        chosen = draw.reds if color == "red" else (draw.blue,)
        for number in chosen:
            values[number].append(index)
    return values


def _feature_at(occurrences: list[list[int]], index: int, number: int, color: str) -> list[float]:
    pool, picks = (33, 6) if color == "red" else (16, 1)
    base = picks / pool
    history = occurrences[number]
    cut = bisect.bisect_left(history, index)
    features = [1.0]
    for window in (5, 15, 50, 150):
        count = cut - bisect.bisect_left(history, max(0, index - window), 0, cut)
        observed = count / min(window, index)
        features.append(observed / base - 1.0)
    last = history[cut - 1] if cut else -1
    gap = index - last - 1 if last >= 0 else index
    expected_gap = (1.0 - base) / base
    features.extend([
        min(gap / max(expected_gap, 1.0), 5.0) / 5.0,
        float(last == index - 1),
        (number - 1) / (pool - 1) - 0.5,
        float(number % 2),
        float(number >= (17 if color == "red" else 9)),
        float(number in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}),
        float(number % 3 == 0),
        float(number % 3 == 1),
    ])
    return features


def _dot(weights: list[float], features: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, features))


def fit_logistic(draws: list[Draw], color: str, end: int, config: LogisticConfig) -> list[float]:
    pool, picks = (33, 6) if color == "red" else (16, 1)
    base = picks / pool
    occurrences = _occurrences(draws, color)
    weights = [math.log(base / (1 - base))] + [0.0] * 12
    for epoch in range(config.epochs):
        rate = config.learning_rate / math.sqrt(epoch + 1)
        for index in range(WARMUP, end):
            truth = set(draws[index].reds) if color == "red" else {draws[index].blue}
            gradient = [0.0] * len(weights)
            for number in range(1, pool + 1):
                features = _feature_at(occurrences, index, number, color)
                error = sigmoid(_dot(weights, features)) - float(number in truth)
                for j, value in enumerate(features):
                    gradient[j] += error * value / pool
            for j in range(len(weights)):
                penalty = 0.0 if j == 0 else config.l2 * weights[j]
                weights[j] -= rate * (gradient[j] + penalty)
    return weights


def probabilities_at(draws: list[Draw], color: str, weights: list[float], index: int | None = None) -> list[float]:
    index = len(draws) if index is None else index
    pool, picks = (33, 6) if color == "red" else (16, 1)
    occurrences = _occurrences(draws, color)
    raw = [sigmoid(_dot(weights, _feature_at(occurrences, index, number, color))) for number in range(1, pool + 1)]
    scale = picks / sum(raw)
    return [max(1e-6, min(1 - 1e-6, value * scale)) for value in raw]


def _probability_metrics(draws: list[Draw], color: str, weights: list[float], start: int, stop: int) -> dict:
    pool, picks = (33, 6) if color == "red" else (16, 1)
    occurrences = _occurrences(draws, color)
    total_loss = total_brier = total_hits = any_hits = 0.0
    for index in range(start, stop):
        truth = set(draws[index].reds) if color == "red" else {draws[index].blue}
        raw = [sigmoid(_dot(weights, _feature_at(occurrences, index, number, color))) for number in range(1, pool + 1)]
        scale = picks / sum(raw)
        probs = [max(1e-6, min(1 - 1e-6, value * scale)) for value in raw]
        ranked = sorted(range(1, pool + 1), key=lambda number: (-probs[number - 1], number))[:picks]
        hits = len(truth.intersection(ranked))
        total_hits += hits
        any_hits += hits > 0
        for number, probability in enumerate(probs, 1):
            y = float(number in truth)
            total_brier += (probability - y) ** 2
            total_loss -= y * math.log(probability) + (1 - y) * math.log(1 - probability)
    count = stop - start
    return {
        "draws": count, "average_hits": total_hits / count, "hit_rate": any_hits / count,
        "brier": total_brier / (count * pool), "log_loss": total_loss / (count * pool),
    }


def _tune_logistic(draws: list[Draw], color: str, validation_start: int, test_start: int) -> tuple[LogisticConfig, dict]:
    candidates = [
        LogisticConfig(0.03, 0.0005), LogisticConfig(0.02, 0.002),
        LogisticConfig(0.01, 0.01), LogisticConfig(0.005, 0.05),
    ]
    results = []
    for config in candidates:
        weights = fit_logistic(draws, color, validation_start, config)
        metrics = _probability_metrics(draws, color, weights, validation_start, test_start)
        results.append((metrics["log_loss"], metrics["brier"], config, metrics))
    _, _, best, metrics = min(results, key=lambda item: (item[0], item[1]))
    return best, metrics


def build_shape_profile(draws: list[Draw], end: int | None = None, window: int = 500) -> dict:
    end = len(draws) if end is None else end
    start = max(1, end - window)
    rows = [shape_features(draws[index].reds, set(draws[index - 1].reds)) for index in range(start, end)]
    numeric: dict[str, dict[str, float]] = {}
    categorical: dict[str, dict[str, int]] = {}
    for name in NUMERIC_SHAPES:
        values = [getattr(row, name) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
        numeric[name] = {"mean": mean, "std": max(math.sqrt(variance), 1.0)}
    for name in CATEGORICAL_SHAPES:
        categorical[name] = {str(key): count for key, count in sorted(Counter(getattr(row, name) for row in rows).items())}
    return {"window": len(rows), "numeric": numeric, "categorical": categorical}


def shape_score(features: ShapeFeatures, profile: dict) -> float:
    score = 0.0
    terms = 0
    for name, stats in profile["numeric"].items():
        z = (getattr(features, name) - stats["mean"]) / stats["std"]
        score += -0.5 * min(z * z, 16.0) - math.log(stats["std"])
        terms += 1
    for name, counts in profile["categorical"].items():
        total = sum(counts.values())
        possible = 7
        probability = (counts.get(str(getattr(features, name)), 0) + 1) / (total + possible)
        score += math.log(probability)
        terms += 1
    return score / terms


def _weighted_sample(probabilities: list[float], count: int, rng: random.Random) -> tuple[int, ...]:
    numbers = list(range(1, len(probabilities) + 1))
    weights = probabilities[:]
    selected = []
    for _ in range(count):
        threshold = rng.random() * sum(weights)
        cumulative = 0.0
        chosen = len(weights) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen = index
                break
        selected.append(numbers.pop(chosen))
        weights.pop(chosen)
    return tuple(sorted(selected))


def _rank_candidates(red_probs: list[float], previous: set[int], profile: dict, shape_weight: float, rng: random.Random, count: int) -> list[tuple[float, tuple[int, ...], ShapeFeatures]]:
    candidates: dict[tuple[int, ...], tuple[float, ShapeFeatures]] = {}
    top = tuple(sorted(sorted(range(1, 34), key=lambda n: (-red_probs[n - 1], n))[:6]))
    pool = [top] + [_weighted_sample(red_probs, 6, rng) for _ in range(count)]
    for reds in pool:
        features = shape_features(reds, previous)
        ball_score = sum(math.log(red_probs[number - 1]) for number in reds) / 6
        score = ball_score + shape_weight * shape_score(features, profile)
        candidates[reds] = (score, features)
    return sorted(((score, reds, features) for reds, (score, features) in candidates.items()), reverse=True)


def _tune_shape_weight(draws: list[Draw], red_weights: list[float], validation_start: int, test_start: int) -> tuple[float, dict]:
    # 0.0 is retained as an explicit ablation baseline. V2 intentionally uses
    # statistical shape information, so selection is among non-zero weights.
    candidates = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5)
    hits = {weight: 0.0 for weight in candidates}
    evaluated = list(range(validation_start, test_start, max(1, (test_start - validation_start) // 120)))
    for index in evaluated:
        probs = probabilities_at(draws, "red", red_weights, index)
        profile = build_shape_profile(draws, index)
        previous = set(draws[index - 1].reds)
        rng = random.Random(10_000 + index)
        pool = [_weighted_sample(probs, 6, rng) for _ in range(180)]
        pool.append(tuple(sorted(sorted(range(1, 34), key=lambda n: (-probs[n - 1], n))[:6])))
        truth = set(draws[index].reds)
        for weight in candidates:
            best = max(pool, key=lambda reds: sum(math.log(probs[n - 1]) for n in reds) / 6 + weight * shape_score(shape_features(reds, previous), profile))
            hits[weight] += len(set(best) & truth)
    best_weight = max(candidates[1:], key=lambda weight: (hits[weight], -weight))
    return best_weight, {
        "evaluated_draws": len(evaluated),
        "selected_nonzero_weight": best_weight,
        "zero_weight_ablation": "纯号码概率对照，不作为 V2 最终模型",
        "average_hits_by_weight": {str(weight): hits[weight] / len(evaluated) for weight in candidates},
    }


def _combination_backtest(draws: list[Draw], weights: list[float], shape_weight: float, start: int, stop: int) -> dict:
    hits = 0.0
    shape_error = {"total": 0.0, "span": 0.0, "odd_count": 0.0, "big_count": 0.0}
    for index in range(start, stop):
        probs = probabilities_at(draws, "red", weights, index)
        profile = build_shape_profile(draws, index)
        ranked = _rank_candidates(probs, set(draws[index - 1].reds), profile, shape_weight, random.Random(20_000 + index), 220)
        predicted = ranked[0][1]
        hits += len(set(predicted) & set(draws[index].reds))
        predicted_shape = ranked[0][2]
        actual_shape = shape_features(draws[index].reds, set(draws[index - 1].reds))
        for name in shape_error:
            shape_error[name] += abs(getattr(predicted_shape, name) - getattr(actual_shape, name))
    count = stop - start
    return {"draws": count, "average_red_hits": hits / count, "mean_absolute_shape_error": {name: value / count for name, value in shape_error.items()}}


def train_and_backtest(draws: list[Draw], holdout_fraction: float = 0.2) -> dict:
    if len(draws) < 500:
        raise ValueError("V2 至少需要 500 期数据")
    test_start = int(len(draws) * (1 - holdout_fraction))
    validation_start = int(test_start * 0.78)
    red_config, red_validation = _tune_logistic(draws, "red", validation_start, test_start)
    blue_config, blue_validation = _tune_logistic(draws, "blue", validation_start, test_start)
    tuning_red_weights = fit_logistic(draws, "red", validation_start, red_config)
    shape_weight, shape_validation = _tune_shape_weight(draws, tuning_red_weights, validation_start, test_start)
    red_weights = fit_logistic(draws, "red", test_start, red_config)
    blue_weights = fit_logistic(draws, "blue", test_start, blue_config)
    red_test = _probability_metrics(draws, "red", red_weights, test_start, len(draws))
    blue_test = _probability_metrics(draws, "blue", blue_weights, test_start, len(draws))
    combination_test = _combination_backtest(draws, red_weights, shape_weight, test_start, len(draws))
    return {
        "schema_version": 2,
        "model_type": "logistic_ball_propensity_plus_statistical_shape_reranker",
        "data": {"count": len(draws), "first_issue": draws[0].issue, "latest_issue": draws[-1].issue, "test_start_issue": draws[test_start].issue},
        "features": {
            "ball": ["freq_5", "freq_15", "freq_50", "freq_150", "omission", "repeat", "number", "odd", "big", "prime", "mod3_0", "mod3_1"],
            "shape": list(NUMERIC_SHAPES + CATEGORICAL_SHAPES),
        },
        "red": {"config": asdict(red_config), "weights": red_weights, "validation": red_validation, "test": red_test},
        "blue": {"config": asdict(blue_config), "weights": blue_weights, "validation": blue_validation, "test": blue_test},
        "shape": {"weight": shape_weight, "validation": shape_validation, "test": combination_test},
        "uniform_baseline": {"red_expected_hits": 36 / 33, "blue_hit_rate": 1 / 16},
        "notes": "所有特征只由目标期之前的数据计算；超参数仅在测试集之前选择。",
    }


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict(draws: list[Draw], model: dict, tickets: int = 5, seed: int = 2026, candidate_pool: int = 5000) -> dict:
    if model.get("schema_version") != 2:
        raise ValueError("需要 V2 模型文件")
    if model["data"]["latest_issue"] != draws[-1].issue:
        raise ValueError("模型与数据最新期号不一致，请重新训练")
    red_probs = probabilities_at(draws, "red", model["red"]["weights"])
    blue_probs = probabilities_at(draws, "blue", model["blue"]["weights"])
    profile = build_shape_profile(draws)
    ranked = _rank_candidates(red_probs, set(draws[-1].reds), profile, model["shape"]["weight"], random.Random(seed), candidate_pool)
    selected = []
    for score, reds, features in ranked:
        if all(len(set(reds) & set(existing["reds"])) <= 4 for existing in selected):
            selected.append({"reds": list(reds), "score": round(score, 6), "statistics": features.to_dict()})
        if len(selected) == tickets:
            break
    blue_rng = random.Random(seed + 1)
    for ticket in selected:
        ticket["blue"] = _weighted_sample(blue_probs, 1, blue_rng)[0]
    target_date = next_draw_date(draws[-1].draw_date)
    return {
        "based_on_issue": draws[-1].issue, "target_issue": next_issue(draws[-1].issue, target_date),
        "estimated_draw_date": target_date.isoformat(), "seed": seed, "candidate_pool": candidate_pool,
        "tickets": selected,
        "red_ranking": sorted(({"number": n, "probability": round(p, 6)} for n, p in enumerate(red_probs, 1)), key=lambda row: -row["probability"]),
        "blue_ranking": sorted(({"number": n, "probability": round(p, 6)} for n, p in enumerate(blue_probs, 1)), key=lambda row: -row["probability"]),
        "shape_profile": profile,
        "warning": "统计形态只能描述历史分布，不能保证预测未来；结果不构成购彩建议。",
    }

