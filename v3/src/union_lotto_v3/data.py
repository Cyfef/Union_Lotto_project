from __future__ import annotations

import csv
import re
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

FALLBACK_URL = "https://data.17500.cn/ssq_asc.txt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
CSV_FIELDS = ["issue", "date", "red1", "red2", "red3", "red4", "red5", "red6", "blue", "source"]


@dataclass(frozen=True)
class Draw:
    issue: str
    draw_date: date
    reds: tuple[int, int, int, int, int, int]
    blue: int
    source: str = ""

    def validate(self) -> None:
        if not re.fullmatch(r"\d{7}", self.issue):
            raise ValueError(f"非法期号: {self.issue}")
        if len(self.reds) != 6 or tuple(sorted(set(self.reds))) != self.reds:
            raise ValueError(f"{self.issue} 红球必须是 6 个不重复升序号码")
        if not all(1 <= value <= 33 for value in self.reds) or not 1 <= self.blue <= 16:
            raise ValueError(f"{self.issue} 号码越界")


def validate_draws(draws: list[Draw]) -> None:
    if not draws:
        raise ValueError("数据集为空")
    seen: set[str] = set()
    previous: Draw | None = None
    for draw in draws:
        draw.validate()
        if draw.issue in seen:
            raise ValueError(f"重复期号: {draw.issue}")
        if previous and (draw.draw_date, draw.issue) <= (previous.draw_date, previous.issue):
            raise ValueError("开奖记录必须按日期和期号严格升序")
        seen.add(draw.issue)
        previous = draw


def load_csv(path: Path) -> list[Draw]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    draws = [
        Draw(row["issue"], date.fromisoformat(row["date"]), tuple(int(row[f"red{i}"]) for i in range(1, 7)), int(row["blue"]), row.get("source", ""))
        for row in rows
    ]
    validate_draws(draws)
    return draws


def save_csv(draws: list[Draw], path: Path) -> None:
    validate_draws(draws)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for draw in draws:
            writer.writerow([draw.issue, draw.draw_date.isoformat(), *draw.reds, draw.blue, draw.source])
    temporary.replace(path)


def fetch_history(retries: int = 3) -> list[Draw]:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(FALLBACK_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("gbk", "ignore")
            draws: dict[str, Draw] = {}
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 9 or not re.fullmatch(r"\d{7}", parts[0]):
                    continue
                item = Draw(parts[0], date.fromisoformat(parts[1]), tuple(sorted(map(int, parts[2:8]))), int(parts[8]), "data.17500.cn")
                item.validate()
                draws[item.issue] = item
            result = sorted(draws.values(), key=lambda item: (item.draw_date, item.issue))
            if len(result) < 300:
                raise RuntimeError("抓取记录少于 300 期")
            validate_draws(result)
            return result
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"历史数据抓取失败: {error}")


def next_draw_date(latest: date) -> date:
    result = latest + timedelta(days=1)
    while result.weekday() not in (1, 3, 6):
        result += timedelta(days=1)
    return result


def next_issue(issue: str, target_date: date) -> str:
    year, serial = int(issue[:4]), int(issue[4:])
    return f"{target_date.year}{1 if target_date.year != year else serial + 1:03d}"

