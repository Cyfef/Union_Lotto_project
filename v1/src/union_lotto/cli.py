from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_all, load_csv, save_csv, validate_draws
from .model import load_model, predict, save_model, train_and_backtest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="union-lotto", description="双色球数据、回测与候选号码生成")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="抓取并校验历史开奖数据")
    fetch.add_argument("--source", choices=("auto", "official", "fallback"), default="auto")
    fetch.add_argument("--output", type=Path, default=Path("data/ssq_history.csv"))
    validate = sub.add_parser("validate", help="校验本地 CSV")
    validate.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    train = sub.add_parser("train", help="训练模型并进行时间留出回测")
    train.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    train.add_argument("--output", type=Path, default=Path("artifacts/model.json"))
    train.add_argument("--holdout", type=float, default=0.2)
    pred = sub.add_parser("predict", help="生成下一期候选组合")
    pred.add_argument("--data", type=Path, default=Path("data/ssq_history.csv"))
    pred.add_argument("--model", type=Path, default=Path("artifacts/model.json"))
    pred.add_argument("--output", type=Path, default=Path("predictions/latest.json"))
    pred.add_argument("--tickets", type=int, default=5)
    pred.add_argument("--seed", type=int, default=2026)
    pipeline = sub.add_parser("pipeline", help="依次抓取、训练、回测和预测")
    pipeline.add_argument("--source", choices=("auto", "official", "fallback"), default="auto")
    pipeline.add_argument("--tickets", type=int, default=5)
    pipeline.add_argument("--seed", type=int, default=2026)
    return parser


def _fetch(source: str, output: Path) -> None:
    draws = fetch_all(source)
    count = save_csv(draws, output)
    print(f"已保存 {count} 期：{draws[0].issue} ~ {draws[-1].issue} -> {output}")


def _train(data: Path, output: Path, holdout: float = 0.2) -> dict:
    if not 0.05 <= holdout <= 0.4:
        raise ValueError("holdout 必须在 0.05 到 0.4 之间")
    model = train_and_backtest(load_csv(data), holdout)
    save_model(model, output)
    red, blue = model["red"]["test"], model["blue"]["test"]
    print(f"模型已保存 -> {output}")
    print(f"测试集：红球平均命中 {red['average_hits']:.3f}/6，蓝球命中率 {blue['hit_rate']:.2%}")
    print(f"概率质量：红球 log-loss={red['log_loss']:.6f}，蓝球 log-loss={blue['log_loss']:.6f}")
    return model


def _predict(data: Path, model_path: Path, output: Path, tickets: int, seed: int) -> dict:
    if not 1 <= tickets <= 100:
        raise ValueError("tickets 必须在 1 到 100 之间")
    result = predict(load_csv(data), load_model(model_path), tickets, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"目标期号 {result['target_issue']}（预计 {result['estimated_draw_date']}）")
    for index, ticket in enumerate(result["tickets"], 1):
        print(f"{index:02d}. {' '.join(f'{n:02d}' for n in ticket['reds'])} + {ticket['blue']:02d}")
    print(f"完整概率与结果 -> {output}")
    print(result["warning"])
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch":
        _fetch(args.source, args.output)
    elif args.command == "validate":
        draws = load_csv(args.data)
        validate_draws(draws)
        print(f"校验通过：{len(draws)} 期，{draws[0].issue} ~ {draws[-1].issue}")
    elif args.command == "train":
        _train(args.data, args.output, args.holdout)
    elif args.command == "predict":
        _predict(args.data, args.model, args.output, args.tickets, args.seed)
    elif args.command == "pipeline":
        data, model, output = Path("data/ssq_history.csv"), Path("artifacts/model.json"), Path("predictions/latest.json")
        _fetch(args.source, data)
        _train(data, model)
        _predict(data, model, output, args.tickets, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

