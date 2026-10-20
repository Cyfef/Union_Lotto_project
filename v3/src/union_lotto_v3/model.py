from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Draw, next_draw_date, next_issue


@dataclass(frozen=True)
class DistributionConfig:
    half_life: int
    prior_strength: float


def _decay(config: DistributionConfig) -> float:
    return math.exp(math.log(0.5) / config.half_life)


def _pool(color: str) -> tuple[int, int]:
    if color == "red":
        return 33, 6
    if color == "blue":
        return 16, 1
    raise ValueError("color 必须是 red 或 blue")


def _chosen(draw: Draw, color: str) -> tuple[int, ...]:
    return draw.reds if color == "red" else (draw.blue,)


def posterior_alpha(draws: list[Draw], color: str, config: DistributionConfig, end: int | None = None) -> list[float]:
    pool, _ = _pool(color)
    end = len(draws) if end is None else end
    decay = _decay(config)
    counts = [0.0] * pool
    for draw in draws[:end]:
        counts = [value * decay for value in counts]
        for number in _chosen(draw, color):
            counts[number - 1] += 1.0
    prior = config.prior_strength / pool
    return [prior + value for value in counts]


def posterior_mean(alpha: list[float]) -> list[float]:
    total = sum(alpha)
    return [value / total for value in alpha]


def sample_dirichlet(alpha: list[float], rng: random.Random) -> list[float]:
    values = [rng.gammavariate(value, 1.0) for value in alpha]
    total = sum(values)
    return [value / total for value in values]


def weighted_without_replacement(weights: list[float], count: int, rng: random.Random) -> tuple[int, ...]:
    numbers = list(range(1, len(weights) + 1))
    remaining = weights[:]
    selected = []
    for _ in range(count):
        threshold = rng.random() * sum(remaining)
        cumulative = 0.0
        choice = len(remaining) - 1
        for index, weight in enumerate(remaining):
            cumulative += weight
            if cumulative >= threshold:
                choice = index
                break
        selected.append(numbers.pop(choice))
        remaining.pop(choice)
    return tuple(sorted(selected))


def unordered_pl_probability(weights: list[float], selected: tuple[int, ...] | list[int]) -> float:
    """Exact probability of an unordered set under successive weighted sampling."""
    indexes = [number - 1 for number in selected]
    size = len(indexes)
    subset_weight = [0.0] * (1 << size)
    for mask in range(1, 1 << size):
        bit = mask & -mask
        position = bit.bit_length() - 1
        subset_weight[mask] = subset_weight[mask ^ bit] + weights[indexes[position]]
    total = sum(weights)
    probability = [0.0] * (1 << size)
    probability[0] = 1.0
    for mask in range(1 << size):
        denominator = total - subset_weight[mask]
        if denominator <= 0:
            continue
        for position, index in enumerate(indexes):
            bit = 1 << position
            if not mask & bit:
                probability[mask | bit] += probability[mask] * weights[index] / denominator
    return max(probability[-1], 1e-300)


def _evaluate(draws: list[Draw], color: str, config: DistributionConfig, start: int, stop: int) -> dict:
    pool, picks = _pool(color)
    decay = _decay(config)
    prior = config.prior_strength / pool
    counts = [0.0] * pool
    negative_log_score = hits = 0.0
    for index, draw in enumerate(draws[:stop]):
        alpha = [prior + value for value in counts]
        weights = posterior_mean(alpha)
        if index >= start:
            truth = _chosen(draw, color)
            if color == "red":
                probability = unordered_pl_probability(weights, truth)
                ranked = sorted(range(1, pool + 1), key=lambda number: (-weights[number - 1], number))[:picks]
                hits += len(set(ranked) & set(truth))
            else:
                probability = weights[truth[0] - 1]
                hits += max(range(pool), key=lambda i: weights[i]) == truth[0] - 1
            negative_log_score -= math.log(max(probability, 1e-300))
        counts = [value * decay for value in counts]
        for number in _chosen(draw, color):
            counts[number - 1] += 1.0
    count = stop - start
    return {
        "draws": count,
        "mean_negative_log_probability": negative_log_score / count,
        "average_hits": hits / count,
        "hit_rate": hits / count if picks == 1 else None,
    }


def _tune(draws: list[Draw], color: str, validation_start: int, test_start: int) -> tuple[DistributionConfig, dict]:
    candidates = [
        DistributionConfig(half_life, prior)
        for half_life in (24, 48, 96, 192, 384, 1536)
        for prior in (1.0, 10.0, 50.0, 200.0)
    ]
    scored = [(config, _evaluate(draws, color, config, validation_start, test_start)) for config in candidates]
    config, metrics = min(scored, key=lambda item: item[1]["mean_negative_log_probability"])
    return config, metrics


def _credible_intervals(alpha: list[float]) -> list[dict]:
    total = sum(alpha)
    results = []
    for number, value in enumerate(alpha, 1):
        mean = value / total
        variance = value * (total - value) / (total * total * (total + 1))
        delta = 1.96 * math.sqrt(max(variance, 0.0))
        results.append({
            "number": number,
            "latent_weight_mean": mean,
            "credible_95_low": max(0.0, mean - delta),
            "credible_95_high": min(1.0, mean + delta),
        })
    return results


def estimate_inclusion_probabilities(alpha: list[float], picks: int, simulations: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    counts = [0] * len(alpha)
    for _ in range(simulations):
        weights = sample_dirichlet(alpha, rng)
        for number in weighted_without_replacement(weights, picks, rng):
            counts[number - 1] += 1
    return [count / simulations for count in counts]


def _posterior_predictive_check(draws: list[Draw], red_alpha: list[float], simulations: int = 5000, seed: int = 731) -> dict:
    rng = random.Random(seed)
    simulated = []
    for _ in range(simulations):
        weights = sample_dirichlet(red_alpha, rng)
        values = weighted_without_replacement(weights, 6, rng)
        simulated.append((sum(values), values[-1] - values[0], sum(value % 2 for value in values)))
    recent = draws[-min(500, len(draws)):]
    observed = [(sum(draw.reds), draw.reds[-1] - draw.reds[0], sum(value % 2 for value in draw.reds)) for draw in recent]

    def means(rows: list[tuple[int, int, int]]) -> dict:
        return {
            "sum_mean": sum(row[0] for row in rows) / len(rows),
            "span_mean": sum(row[1] for row in rows) / len(rows),
            "odd_count_mean": sum(row[2] for row in rows) / len(rows),
        }

    return {"observed_recent_draws": len(recent), "simulations": simulations, "observed": means(observed), "simulated": means(simulated)}


def train_and_backtest(draws: list[Draw], holdout_fraction: float = 0.2) -> dict:
    if len(draws) < 500:
        raise ValueError("V3 至少需要 500 期数据")
    test_start = int(len(draws) * (1 - holdout_fraction))
    validation_start = int(test_start * 0.78)
    red_config, red_validation = _tune(draws, "red", validation_start, test_start)
    blue_config, blue_validation = _tune(draws, "blue", validation_start, test_start)
    red_test = _evaluate(draws, "red", red_config, test_start, len(draws))
    blue_test = _evaluate(draws, "blue", blue_config, test_start, len(draws))
    red_alpha = posterior_alpha(draws, "red", red_config)
    blue_alpha = posterior_alpha(draws, "blue", blue_config)
    return {
        "schema_version": 3,
        "model_type": "bayesian_dirichlet_plackett_luce_posterior_predictive",
        "data": {"count": len(draws), "first_issue": draws[0].issue, "latest_issue": draws[-1].issue, "test_start_issue": draws[test_start].issue},
        "red": {"config": asdict(red_config), "posterior_alpha": red_alpha, "validation": red_validation, "test": red_test},
        "blue": {"config": asdict(blue_config), "posterior_alpha": blue_alpha, "validation": blue_validation, "test": blue_test},
        "uniform_baseline": {
            "red_mean_negative_log_probability": math.log(math.comb(33, 6)),
            "red_expected_hits": 36 / 33,
            "blue_mean_negative_log_probability": math.log(16),
            "blue_hit_rate": 1 / 16,
        },
        "posterior_predictive_check": _posterior_predictive_check(draws, red_alpha),
        "notes": "红球用衰减计数更新 Dirichlet 权重，是 Plackett-Luce 似然的幂后验近似；所有调参仅使用测试集之前的数据。",
    }


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict(draws: list[Draw], model: dict, tickets: int = 5, seed: int = 2026, simulations: int = 10000) -> dict:
    if model.get("schema_version") != 3:
        raise ValueError("需要 V3 模型文件")
    if model["data"]["latest_issue"] != draws[-1].issue:
        raise ValueError("模型与数据最新期号不一致，请重新训练")
    red_alpha = model["red"]["posterior_alpha"]
    blue_alpha = model["blue"]["posterior_alpha"]
    red_mean = posterior_mean(red_alpha)
    blue_mean = posterior_mean(blue_alpha)
    red_inclusion = estimate_inclusion_probabilities(red_alpha, 6, simulations, seed + 100)
    rng = random.Random(seed)
    generated = []
    seen = set()
    attempts = 0
    while len(generated) < tickets and attempts < tickets * 200:
        attempts += 1
        red_weights = sample_dirichlet(red_alpha, rng)
        blue_weights = sample_dirichlet(blue_alpha, rng)
        reds = weighted_without_replacement(red_weights, 6, rng)
        blue = weighted_without_replacement(blue_weights, 1, rng)[0]
        key = (reds, blue)
        if key in seen:
            continue
        seen.add(key)
        probability = unordered_pl_probability(red_mean, reds) * blue_mean[blue - 1]
        generated.append({
            "reds": list(reds), "blue": blue,
            "approx_posterior_mean_probability": probability,
            "one_in_approximately": round(1 / probability) if probability > 0 else None,
        })
    target_date = next_draw_date(draws[-1].draw_date)
    red_summary = _credible_intervals(red_alpha)
    for row, inclusion in zip(red_summary, red_inclusion):
        row["posterior_predictive_inclusion_probability"] = inclusion
    blue_summary = _credible_intervals(blue_alpha)
    return {
        "based_on_issue": draws[-1].issue,
        "target_issue": next_issue(draws[-1].issue, target_date),
        "estimated_draw_date": target_date.isoformat(),
        "seed": seed, "posterior_simulations": simulations, "tickets": generated,
        "red_distribution": sorted(red_summary, key=lambda row: -row["posterior_predictive_inclusion_probability"]),
        "blue_distribution": sorted(blue_summary, key=lambda row: -row["latent_weight_mean"]),
        "warning": "后验概率表达模型的不确定性，不代表真实开奖机制或中奖保证。",
    }

