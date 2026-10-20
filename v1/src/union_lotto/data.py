from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

OFFICIAL_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/ssqkj"
FALLBACK_URL = "https://data.17500.cn/ssq_asc.txt"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
CST = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Draw:
    issue: str
    draw_date: date
    reds: tuple[int, int, int, int, int, int]
    blue: int
    source: str = ""

    def validate(self) -> None:
        if not (self.issue.isdigit() and len(self.issue) == 7):
            raise ValueError(f"非法期号: {self.issue}")
        if len(self.reds) != 6 or tuple(sorted(set(self.reds))) != self.reds:
            raise ValueError(f"{self.issue} 红球必须是 6 个不重复升序号码")
        if not all(1 <= n <= 33 for n in self.reds):
            raise ValueError(f"{self.issue} 红球越界")
        if not 1 <= self.blue <= 16:
            raise ValueError(f"{self.issue} 蓝球越界")


def _request(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    error: Exception | None = None
    headers = {"User-Agent": UA, "Accept": "application/json,text/plain,*/*", "Referer": "https://www.cwl.gov.cn/"}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # network failures are retried uniformly
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {error}")


def _parse_date(value: str) -> date:
    match = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", value)
    if not match:
        raise ValueError(f"无法解析日期: {value!r}")
    return date(*map(int, match.groups()))


def _numbers(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x) for x in re.findall(r"\d+", str(value or ""))]


def _official_item(row: dict) -> Draw:
    issue = str(row.get("code") or row.get("issue") or "").strip()
    reds = tuple(sorted(_numbers(row.get("red") or row.get("frontWinningNum"))))
    blues = _numbers(row.get("blue") or row.get("backWinningNum"))
    draw = Draw(issue, _parse_date(str(row.get("date") or row.get("openTime") or "")), reds, blues[0], "cwl.gov.cn")
    draw.validate()
    return draw


def fetch_official(start_year: int = 2013, end_year: int | None = None) -> list[Draw]:
    """Fetch official results. The endpoint can be rate-limited outside mainland China."""
    end_year = end_year or datetime.now(CST).year
    draws: dict[str, Draw] = {}
    for year in range(start_year, end_year + 1):
        page = 1
        while True:
            query = urllib.parse.urlencode({
                "issueStart": f"{year}001", "issueEnd": f"{year}999",
                "dayStart": "", "dayEnd": "", "pageNo": page, "pageSize": 200, "week": "",
            })
            payload = json.loads(_request(f"{OFFICIAL_URL}?{query}").decode("utf-8-sig"))
            rows = payload.get("result") or payload.get("data") or []
            if not rows:
                break
            for row in rows:
                item = _official_item(row)
                draws[item.issue] = item
            page_count = int(payload.get("pageNum") or payload.get("pages") or math.ceil(int(payload.get("total", len(rows))) / max(len(rows), 1)))
            if page >= page_count:
                break
            page += 1
    if not draws:
        raise RuntimeError("官方接口未返回有效开奖记录")
    return sorted(draws.values(), key=lambda x: (x.draw_date, x.issue))


def fetch_fallback() -> list[Draw]:
    """Fetch the public 17500 full-history text mirror."""
    text = _request(FALLBACK_URL, timeout=60).decode("gbk", "ignore")
    draws: dict[str, Draw] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 9 or not re.fullmatch(r"\d{7}", parts[0]):
            continue
        try:
            item = Draw(parts[0], _parse_date(parts[1]), tuple(sorted(map(int, parts[2:8]))), int(parts[8]), "data.17500.cn")
            item.validate()
            draws[item.issue] = item
        except (ValueError, IndexError):
            continue
    if len(draws) < 100:
        raise RuntimeError(f"备用数据源仅解析到 {len(draws)} 期，拒绝写入")
    return sorted(draws.values(), key=lambda x: (x.draw_date, x.issue))


def fetch_all(source: str = "auto") -> list[Draw]:
    if source == "official":
        return fetch_official()
    if source == "fallback":
        return fetch_fallback()
    if source != "auto":
        raise ValueError("source 必须是 auto、official 或 fallback")
    fallback = fetch_fallback()
    try:
        official = fetch_official()
    except Exception as exc:
        print(f"提示：官方接口暂不可用（{exc}），已使用公开历史镜像并完成校验。")
        return fallback
    merged = {d.issue: d for d in fallback}
    merged.update({d.issue: d for d in official})
    return sorted(merged.values(), key=lambda x: (x.draw_date, x.issue))


CSV_FIELDS = ["issue", "date", "red1", "red2", "red3", "red4", "red5", "red6", "blue", "source"]


def save_csv(draws: Iterable[Draw], path: Path) -> int:
    values = list(draws)
    validate_draws(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for d in values:
            writer.writerow(dict(zip(CSV_FIELDS, [d.issue, d.draw_date.isoformat(), *d.reds, d.blue, d.source])))
    temporary.replace(path)
    return len(values)


def load_csv(path: Path) -> list[Draw]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    draws = [Draw(r["issue"], date.fromisoformat(r["date"]), tuple(int(r[f"red{i}"]) for i in range(1, 7)), int(r["blue"]), r.get("source", "")) for r in rows]
    validate_draws(draws)
    return draws


def validate_draws(draws: list[Draw]) -> None:
    if not draws:
        raise ValueError("数据集为空")
    issues: set[str] = set()
    previous: Draw | None = None
    for draw in draws:
        draw.validate()
        if draw.issue in issues:
            raise ValueError(f"重复期号: {draw.issue}")
        if previous and (draw.draw_date, draw.issue) <= (previous.draw_date, previous.issue):
            raise ValueError("开奖记录必须按日期、期号严格升序")
        issues.add(draw.issue)
        previous = draw


def next_draw_date(latest: date) -> date:
    candidate = latest + timedelta(days=1)
    while candidate.weekday() not in (1, 3, 6):  # Tue, Thu, Sun
        candidate += timedelta(days=1)
    return candidate


def next_issue(issue: str, next_date: date) -> str:
    year, serial = int(issue[:4]), int(issue[4:])
    return f"{next_date.year}{1 if next_date.year != year else serial + 1:03d}"

