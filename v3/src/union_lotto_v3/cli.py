from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_history, load_csv, save_csv
from .model import load_model, predict, save_model, train_and_backtest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="union-lotto-v3", description="双色球贝叶斯后验预测模型 V3")
    sub = result.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--output", type=Path, default=Path("data/ssq_history.csv"))
    validate = sub.add_parser("validate")
    validate.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    train.add_argument("--output", type=Path, default=Path("artifacts/model.json"))
    train.add_argument("--holdout", type=float, default=0.2)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    prediction.add_argument("--model", type=Path, default=Path("artifacts/model.json"))
    prediction.add_argument("--output", type=Path, default=Path("predictions/latest.json"))
    prediction.add_argument("--tickets", type=int, default=5)
    prediction.add_argument("--seed", type=int, default=2026)
    prediction.add_argument("--simulations", type=int, default=10000)
    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--tickets", type=int, default=5)
    pipeline.add_argument("--seed", type=int, default=2026)
    return result


def run_train(data: Path, output: Path, holdout: float) -> dict:
    model = train_and_backtest(load_csv(data), holdout)
    save_model(model, output)
    red, blue = model["red"]["test"], model["blue"]["test"]
    print(f"模型已保存 -> {output}")
    print(f"红球平均命中 {red['average_hits']:.3f}，平均负对数概率 {red['mean_negative_log_probability']:.6f}")
    print(f"蓝球命中率 {blue['hit_rate']:.2%}，平均负对数概率 {blue['mean_negative_log_probability']:.6f}")
    return model


def run_predict(data: Path, model_path: Path, output: Path, tickets: int, seed: int, simulations: int) -> dict:
    result = predict(load_csv(data), load_model(model_path), tickets, seed, simulations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"目标 {result['target_issue']}，预计开奖日期 {result['estimated_draw_date']}")
    for index, ticket in enumerate(result["tickets"], 1):
        reds = " ".join(f"{number:02d}" for number in ticket["reds"])
        print(f"{index:02d}. {reds} + {ticket['blue']:02d}，约 1/{ticket['one_in_approximately']:,}")
    print(f"完整后验分布 -> {output}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "fetch":
        draws = fetch_history()
        save_csv(draws, args.output)
        print(f"已保存 {len(draws)} 期 -> {args.output}")
    elif args.command == "validate":
        draws = load_csv(args.data)
        print(f"校验通过：{len(draws)} 期，{draws[0].issue} ~ {draws[-1].issue}")
    elif args.command == "train":
        run_train(args.data, args.output, args.holdout)
    elif args.command == "predict":
        run_predict(args.data, args.model, args.output, args.tickets, args.seed, args.simulations)
    elif args.command == "pipeline":
        data, model, output = Path("data/ssq_history.csv"), Path("artifacts/model.json"), Path("predictions/latest.json")
        draws = fetch_history()
        save_csv(draws, data)
        run_train(data, model, 0.2)
        run_predict(data, model, output, args.tickets, args.seed, 10000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

