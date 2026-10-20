from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_history, load_csv, save_csv
from .model import load_model, predict, save_model, train_and_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="union-lotto-v2", description="双色球统计特征模型 V2")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="更新历史开奖数据")
    fetch.add_argument("--output", type=Path, default=Path("data/ssq_history.csv"))
    validate = sub.add_parser("validate", help="校验历史数据")
    validate.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    train = sub.add_parser("train", help="训练并执行时间留出回测")
    train.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    train.add_argument("--output", type=Path, default=Path("artifacts/model.json"))
    train.add_argument("--holdout", type=float, default=0.2)
    prediction = sub.add_parser("predict", help="生成下一期候选")
    prediction.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    prediction.add_argument("--model", type=Path, default=Path("artifacts/model.json"))
    prediction.add_argument("--output", type=Path, default=Path("predictions/latest.json"))
    prediction.add_argument("--tickets", type=int, default=5)
    prediction.add_argument("--seed", type=int, default=2026)
    prediction.add_argument("--candidate-pool", type=int, default=5000)
    pipeline = sub.add_parser("pipeline", help="依次更新、训练、回测和预测")
    pipeline.add_argument("--tickets", type=int, default=5)
    pipeline.add_argument("--seed", type=int, default=2026)
    return parser


def run_fetch(output: Path) -> None:
    draws = fetch_history()
    save_csv(draws, output)
    print(f"已保存 {len(draws)} 期：{draws[0].issue} ~ {draws[-1].issue} -> {output}")


def run_train(data: Path, output: Path, holdout: float = 0.2) -> dict:
    if not 0.1 <= holdout <= 0.35:
        raise ValueError("holdout 必须在 0.1 到 0.35 之间")
    model = train_and_backtest(load_csv(data), holdout)
    save_model(model, output)
    red, blue, shape = model["red"]["test"], model["blue"]["test"], model["shape"]["test"]
    print(f"模型已保存 -> {output}")
    print(f"红球概率 Top-6 平均命中 {red['average_hits']:.3f}；组合模型平均命中 {shape['average_red_hits']:.3f}")
    print(f"蓝球 Top-1 命中率 {blue['hit_rate']:.2%}")
    return model


def run_predict(data: Path, model_path: Path, output: Path, tickets: int, seed: int, candidate_pool: int = 5000) -> dict:
    if not 1 <= tickets <= 50:
        raise ValueError("tickets 必须在 1 到 50 之间")
    if candidate_pool < max(500, tickets * 50):
        raise ValueError("candidate-pool 太小")
    result = predict(load_csv(data), load_model(model_path), tickets, seed, candidate_pool)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"目标 {result['target_issue']}，预计开奖日期 {result['estimated_draw_date']}")
    for index, ticket in enumerate(result["tickets"], 1):
        reds = " ".join(f"{number:02d}" for number in ticket["reds"])
        stats = ticket["statistics"]
        print(f"{index:02d}. {reds} + {ticket['blue']:02d}  和值={stats['total']} 跨度={stats['span']} 奇偶={stats['odd_count']}:{6-stats['odd_count']}")
    print(f"完整结果 -> {output}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch":
        run_fetch(args.output)
    elif args.command == "validate":
        draws = load_csv(args.data)
        print(f"校验通过：{len(draws)} 期，{draws[0].issue} ~ {draws[-1].issue}")
    elif args.command == "train":
        run_train(args.data, args.output, args.holdout)
    elif args.command == "predict":
        run_predict(args.data, args.model, args.output, args.tickets, args.seed, args.candidate_pool)
    elif args.command == "pipeline":
        data, model, output = Path("data/ssq_history.csv"), Path("artifacts/model.json"), Path("predictions/latest.json")
        run_fetch(data)
        run_train(data, model)
        run_predict(data, model, output, args.tickets, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

