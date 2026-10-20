from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Draw, next_draw_date, next_issue


@dataclass(frozen=True)
class Params:
    half_life: int
    prior_mix: float


@dataclass
class BacktestMetrics:
    draws: int = 0
    average_hits: float = 0.0
    hit_rate: float = 0.0
    brier: float = 0.0
    log_loss: float = 0.0


def _probabilities(history: list[set[int]], pool: int, picks: int, params: Params) -> list[float]:
    base = picks / pool
    decay = math.log(2) / params.half_life
    weighted = [0.0] * (pool + 1)
    total_weight = 0.0
    for age, chosen in enumerate(reversed(history)):
        weight = math.exp(-decay * age)
        total_weight += weight
        for number in chosen:
            weighted[number] += weight
        if age > params.half_life * 12:  # negligible tail
            break
    # Beta prior has 20 effective observations and preserves the known base rate.
    empirical = [(base * 20 + weighted[n]) / (20 + total_weight) for n in range(1, pool + 1)]
    values = [params.prior_mix * base + (1 - params.prior_mix) * p for p in empirical]
    # Inclusion probabilities should sum to the expected number of selected balls.
    scale = picks / sum(values)
    return [min(0.999999, max(0.000001, p * scale)) for p in values]


def probabilities(draws: list[Draw], color: str, params: Params) -> list[float]:
    if color == "red":
        return _probabilities([set(d.reds) for d in draws], 33, 6, params)
    if color == "blue":
        return _probabilities([{d.blue} for d in draws], 16, 1, params)
    raise ValueError("color 必须是 red 或 blue")


def _score_sequence(draws: list[Draw], color: str, params: Params, start: int, stop: int) -> BacktestMetrics:
    pool, picks = (33, 6) if color == "red" else (16, 1)
    loss = brier = hits = any_hit = 0.0
    for index in range(start, stop):
        truth = set(draws[index].reds) if color == "red" else {draws[index].blue}
        probs = probabilities(draws[:index], color, params)
        ranked = sorted(range(1, pool + 1), key=lambda n: (-probs[n - 1], n))[:picks]
        current_hits = len(truth.intersection(ranked))
        hits += current_hits
        any_hit += current_hits > 0
        for number, probability in enumerate(probs, 1):
            y = float(number in truth)
            brier += (probability - y) ** 2
            loss -= y * math.log(probability) + (1 - y) * math.log(1 - probability)
    count = stop - start
    cells = count * pool
    return BacktestMetrics(count, hits / count, any_hit / count, brier / cells, loss / cells)


def tune(draws: list[Draw], color: str, train_end: int) -> tuple[Params, BacktestMetrics]:
    validation_start = max(120, int(train_end * 0.75))
    candidates = [Params(h, mix) for h in (12, 24, 48, 96, 192, 384) for mix in (0.0, 0.25, 0.5, 0.75, 1.0)]
    scored = [(p, _score_sequence(draws, color, p, validation_start, train_end)) for p in candidates]
    return min(scored, key=lambda item: (item[1].log_loss, item[1].brier))[0], min(scored, key=lambda item: (item[1].log_loss, item[1].brier))[1]


def train_and_backtest(draws: list[Draw], holdout_fraction: float = 0.2) -> dict:
    if len(draws) < 300:
        raise ValueError("至少需要 300 期数据才能进行稳定的时序训练与回测")
    test_start = max(200, int(len(draws) * (1 - holdout_fraction)))
    red_params, red_validation = tune(draws, "red", test_start)
    blue_params, blue_validation = tune(draws, "blue", test_start)
    red_test = _score_sequence(draws, "red", red_params, test_start, len(draws))
    blue_test = _score_sequence(draws, "blue", blue_params, test_start, len(draws))
    uniform_red = _score_sequence(draws, "red", Params(24, 1.0), test_start, len(draws))
    uniform_blue = _score_sequence(draws, "blue", Params(24, 1.0), test_start, len(draws))
    return {
        "schema_version": 1,
        "data": {"count": len(draws), "first_issue": draws[0].issue, "latest_issue": draws[-1].issue, "test_start_issue": draws[test_start].issue},
        "red": {"params": asdict(red_params), "validation": asdict(red_validation), "test": asdict(red_test), "uniform_baseline": asdict(uniform_red)},
        "blue": {"params": asdict(blue_params), "validation": asdict(blue_validation), "test": asdict(blue_test), "uniform_baseline": asdict(uniform_blue)},
        "notes": "超参数只在测试集之前的数据上选择；测试集按时间留出，未随机打乱。",
    }


def save_model(model: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _weighted_without_replacement(probabilities_: list[float], count: int, rng: random.Random) -> list[int]:
    remaining = list(range(1, len(probabilities_) + 1))
    weights = probabilities_[:]
    selected: list[int] = []
    for _ in range(count):
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        chosen_index = len(weights) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                chosen_index = index
                break
        selected.append(remaining.pop(chosen_index))
        weights.pop(chosen_index)
    return sorted(selected)


def predict(draws: list[Draw], model: dict, tickets: int = 5, seed: int = 2026) -> dict:
    if model["data"]["latest_issue"] != draws[-1].issue:
        raise ValueError("模型与数据的最新期号不一致，请重新运行 train")
    red_params = Params(**model["red"]["params"])
    blue_params = Params(**model["blue"]["params"])
    red_probs = probabilities(draws, "red", red_params)
    blue_probs = probabilities(draws, "blue", blue_params)
    rng = random.Random(seed)
    generated: list[dict] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    attempts = 0
    while len(generated) < tickets and attempts < tickets * 100:
        attempts += 1
        reds = tuple(_weighted_without_replacement(red_probs, 6, rng))
        blue = _weighted_without_replacement(blue_probs, 1, rng)[0]
        key = (reds, blue)
        if key not in seen:
            seen.add(key)
            generated.append({"reds": list(reds), "blue": blue})
    next_date = next_draw_date(draws[-1].draw_date)
    return {
        "based_on_issue": draws[-1].issue,
        "target_issue": next_issue(draws[-1].issue, next_date),
        "estimated_draw_date": next_date.isoformat(),
        "seed": seed,
        "tickets": generated,
        "red_ranking": sorted(({"number": n, "probability": round(p, 6)} for n, p in enumerate(red_probs, 1)), key=lambda x: -x["probability"]),
        "blue_ranking": sorted(({"number": n, "probability": round(p, 6)} for n, p in enumerate(blue_probs, 1)), key=lambda x: -x["probability"]),
        "warning": "彩票开奖结果近似独立随机；此输出仅用于统计与编程研究，不保证中奖，也不构成购彩建议。",
    }

