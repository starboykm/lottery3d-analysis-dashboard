#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
福彩 3D / 体彩排列三概率训练、策略过滤、预测和开奖命中记录工具。

重要说明：
彩票开奖结果应视为随机事件。本程序只能根据历史频率、转移规律、可选神经网络
和人工策略给出概率估计与选号管理，不能保证盈利或真正预测未来。
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DIGITS = "0123456789"
POSITIONS = ("hundred", "ten", "unit")
POSITION_NAMES = {"hundred": "百位", "ten": "十位", "unit": "个位"}
DEFAULT_RECORD_DIR = "records"


@dataclass
class Draw:
    issue: str
    number: str
    trial: Optional[str] = None


@dataclass
class StrategySignal:
    kill_by_pos: Dict[str, Counter] = field(default_factory=lambda: {p: Counter() for p in POSITIONS})
    kill_sum_tails: Counter = field(default_factory=Counter)
    kill_spans: Counter = field(default_factory=Counter)
    kill_pairs: Counter = field(default_factory=Counter)
    dan_digits: Counter = field(default_factory=Counter)
    preferred_sum_ranges: List[Tuple[int, int, str]] = field(default_factory=list)
    group3_bonus: float = 0.0
    group6_bonus: float = 0.0
    notes: List[str] = field(default_factory=list)

    def add_kill(self, pos: str, digit: int, reason: str, weight: float = 1.0) -> None:
        if pos in self.kill_by_pos and 0 <= digit <= 9:
            self.kill_by_pos[pos][str(digit)] += weight
            self.notes.append(f"{POSITION_NAMES[pos]}杀号 {digit}: {reason}")

    def add_sum_tail_kill(self, digit: int, reason: str, weight: float = 1.0) -> None:
        self.kill_sum_tails[str(digit % 10)] += weight
        self.notes.append(f"和值尾杀号 {digit % 10}: {reason}")

    def add_span_kill(self, span: int, reason: str, weight: float = 1.0) -> None:
        self.kill_spans[str(span % 10)] += weight
        self.notes.append(f"跨度杀号 {span % 10}: {reason}")

    def add_dan(self, digit: int, reason: str, weight: float = 1.0) -> None:
        if 0 <= digit <= 9:
            self.dan_digits[str(digit)] += weight
            self.notes.append(f"胆码 {digit}: {reason}")

    def add_pair_kill(self, pair: str, reason: str, weight: float = 1.0) -> None:
        pair_digits = "".join(sorted([ch for ch in str(pair) if ch.isdigit()]))
        if len(pair_digits) >= 2:
            pair_digits = pair_digits[:2]
            self.kill_pairs[pair_digits] += weight
            self.notes.append(f"二码组合杀号 {pair_digits}: {reason}")


def normalize_number(value: str) -> str:
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"号码必须是数字: {value}")
    if len(text) > 3:
        raise ValueError(f"号码必须是 000-999 的三位数: {value}")
    return text.zfill(3)


def normalize_issue(value: str) -> str:
    text = str(value).strip()
    if not (len(text) == 7 and text.isdigit()):
        raise ValueError(f"期号必须是 7 位数字，前 4 位年份，后 3 位期序: {value}")
    return text


def digits_of(number: str) -> Tuple[int, int, int]:
    n = normalize_number(number)
    return int(n[0]), int(n[1]), int(n[2])


def number_sum(number: str) -> int:
    return sum(digits_of(number))


def span(number: str) -> int:
    vals = digits_of(number)
    return max(vals) - min(vals)


def sum_tail(number: str) -> int:
    return number_sum(number) % 10


def issue_tail(issue: str) -> int:
    return int(normalize_issue(issue)[-1])


def unique_digits_from_text(text: str) -> List[int]:
    return sorted({int(ch) for ch in str(text) if ch.isdigit()})


def load_draws(path: str) -> List[Draw]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    draws: List[Draw] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        reader = csv.DictReader(f) if has_header else csv.reader(f)
        for row in reader:
            if has_header:
                issue = row.get("issue") or row.get("期号") or row.get("qihao")
                number = row.get("number") or row.get("开奖号") or row.get("开奖号码")
                trial = row.get("trial") or row.get("试机号") or None
            else:
                if len(row) < 2:
                    continue
                issue, number = row[0], row[1]
                trial = row[2] if len(row) >= 3 and str(row[2]).strip() else None
            if not issue or not number:
                continue
            draws.append(Draw(normalize_issue(issue), normalize_number(number), normalize_number(trial) if trial else None))

    draws.sort(key=lambda x: x.issue)
    return draws


def save_draws(path: str, draws: Sequence[Draw]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["issue", "number", "trial"])
        for d in sorted(draws, key=lambda x: x.issue):
            writer.writerow([d.issue, d.number, d.trial or ""])


def append_draw(path: str, issue: str, number: str, trial: Optional[str] = None) -> None:
    existing = load_draws(path) if os.path.exists(path) else []
    issue = normalize_issue(issue)
    number = normalize_number(number)
    trial = normalize_number(trial) if trial else None
    replaced = False
    for i, draw in enumerate(existing):
        if draw.issue == issue:
            existing[i] = Draw(issue, number, trial)
            replaced = True
            break
    if not replaced:
        existing.append(Draw(issue, number, trial))
    save_draws(path, existing)


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def normalize_probs(scores: Dict[str, float], floor: float = 1e-9) -> Dict[str, float]:
    clean = {str(k): max(float(v), floor) for k, v in scores.items()}
    total = sum(clean.values()) or 1.0
    return {k: v / total for k, v in sorted(clean.items())}


def decayed_position_probs(draws: Sequence[Draw], half_life: float = 80.0) -> Dict[str, Dict[str, float]]:
    counters = {p: Counter() for p in POSITIONS}
    n = len(draws)
    for idx, draw in enumerate(draws):
        age = n - idx - 1
        w = 0.5 ** (age / half_life)
        for pos, digit in zip(POSITIONS, draw.number):
            counters[pos][digit] += w
    return {pos: normalize_probs({d: counters[pos][d] + 0.8 for d in DIGITS}) for pos in POSITIONS}


def window_position_probs(draws: Sequence[Draw], window: int = 30) -> Dict[str, Dict[str, float]]:
    recent = draws[-window:] if window > 0 else draws
    counters = {p: Counter() for p in POSITIONS}
    for draw in recent:
        for pos, digit in zip(POSITIONS, draw.number):
            counters[pos][digit] += 1
    return {pos: normalize_probs({d: counters[pos][d] + 0.6 for d in DIGITS}) for pos in POSITIONS}


def transition_position_probs(draws: Sequence[Draw]) -> Dict[str, Dict[str, float]]:
    if len(draws) < 2:
        return decayed_position_probs(draws)
    prev = draws[-1].number
    counters = {p: Counter() for p in POSITIONS}
    for a, b in zip(draws[:-1], draws[1:]):
        for i, pos in enumerate(POSITIONS):
            if a.number[i] == prev[i]:
                counters[pos][b.number[i]] += 1.5
            if a.number == prev:
                counters[pos][b.number[i]] += 1.0
    base = decayed_position_probs(draws)
    return {
        pos: normalize_probs({d: counters[pos][d] + base[pos][d] * 8.0 + 0.2 for d in DIGITS})
        for pos in POSITIONS
    }


def markov_position_probs(draws: Sequence[Draw]) -> Dict[str, Dict[str, float]]:
    """独立马尔可夫模型：用上一期状态到下一期状态的转移概率预测各位置数字。"""
    if len(draws) < 2:
        return decayed_position_probs(draws)

    last_num = draws[-1].number
    base = decayed_position_probs(draws, half_life=120.0)
    scores = {pos: Counter({d: base[pos][d] * 12.0 + 0.3 for d in DIGITS}) for pos in POSITIONS}

    # 一阶位置转移：上一期同位置数字 -> 下一期同位置数字。
    for prev, nxt in zip(draws[:-1], draws[1:]):
        for idx, pos in enumerate(POSITIONS):
            if prev.number[idx] == last_num[idx]:
                scores[pos][nxt.number[idx]] += 3.0

    # 整号一阶转移：上一期整三位号码相同 -> 下一期各位置数字。
    for prev, nxt in zip(draws[:-1], draws[1:]):
        if prev.number == last_num:
            for pos, digit in zip(POSITIONS, nxt.number):
                scores[pos][digit] += 6.0

    # 相邻两位状态转移，补充百十、十个、百个关系。
    pair_states = [
        ((0, 1), ("hundred", "ten")),
        ((1, 2), ("ten", "unit")),
        ((0, 2), ("hundred", "unit")),
    ]
    for prev, nxt in zip(draws[:-1], draws[1:]):
        for indexes, pos_names in pair_states:
            if "".join(prev.number[i] for i in indexes) == "".join(last_num[i] for i in indexes):
                for pos, idx in zip(pos_names, indexes):
                    scores[pos][nxt.number[idx]] += 1.5

    # 和值尾、跨度状态转移也作为弱信号。
    last_tail = sum_tail(last_num)
    last_span = span(last_num)
    for prev, nxt in zip(draws[:-1], draws[1:]):
        if sum_tail(prev.number) == last_tail:
            for pos, digit in zip(POSITIONS, nxt.number):
                scores[pos][digit] += 0.45
        if span(prev.number) == last_span:
            for pos, digit in zip(POSITIONS, nxt.number):
                scores[pos][digit] += 0.35

    return {pos: normalize_probs(scores[pos]) for pos in POSITIONS}


def markov_digit_probs(markov_probs: Dict[str, Dict[str, float]], limit: int = 5) -> List[Dict]:
    """Rank digits by Markov-estimated chance of appearing anywhere, ignoring position."""
    rows = []
    for digit in DIGITS:
        miss_prob = 1.0
        expected_count = 0.0
        for pos in POSITIONS:
            p = markov_probs[pos].get(digit, 0.0)
            miss_prob *= 1.0 - p
            expected_count += p
        rows.append(
            {
                "digit": digit,
                "probability": 1.0 - miss_prob,
                "expected_count": expected_count,
            }
        )
    rows.sort(key=lambda x: (x["probability"], x["expected_count"], x["digit"]), reverse=True)
    return rows[:limit]


def gap_position_probs(draws: Sequence[Draw]) -> Dict[str, Dict[str, float]]:
    scores = {p: {d: 1.0 for d in DIGITS} for p in POSITIONS}
    for pos_idx, pos in enumerate(POSITIONS):
        last_seen = {d: None for d in DIGITS}
        for idx, draw in enumerate(draws):
            last_seen[draw.number[pos_idx]] = idx
        n = len(draws)
        for d in DIGITS:
            gap = n if last_seen[d] is None else n - last_seen[d]
            scores[pos][d] = 1.0 + min(gap, 60) / 25.0
    return {pos: normalize_probs(scores[pos]) for pos in POSITIONS}


def build_features(draws: Sequence[Draw], idx: int, target_issue: str, lookback: int = 8) -> List[float]:
    """为可选神经网络生成轻量特征。idx 表示用 draws[:idx] 预测 target_issue。"""
    history = draws[:idx]
    feats: List[float] = []
    tail = issue_tail(target_issue)
    feats.extend([tail / 9.0, int(target_issue[:4]) / 9999.0, int(target_issue[-3:]) / 999.0])
    for draw in history[-lookback:]:
        h, t, u = digits_of(draw.number)
        feats.extend([h / 9.0, t / 9.0, u / 9.0, number_sum(draw.number) / 27.0, span(draw.number) / 9.0])
    missing = lookback - min(len(history), lookback)
    feats.extend([0.0] * missing * 5)
    return feats


def neural_position_probs(draws: Sequence[Draw], target_issue: str) -> Optional[Dict[str, Dict[str, float]]]:
    if len(draws) < 35:
        return None
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return None

    start = 8
    x = [build_features(draws, i, draws[i].issue) for i in range(start, len(draws))]
    if len(x) < 25:
        return None
    pred_x = [build_features(draws, len(draws), target_issue)]
    result: Dict[str, Dict[str, float]] = {}

    for pos_idx, pos in enumerate(POSITIONS):
        y = [int(draws[i].number[pos_idx]) for i in range(start, len(draws))]
        if len(set(y)) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=0.015,
                learning_rate_init=0.004,
                max_iter=700,
                random_state=20260426 + pos_idx,
            ),
        )
        try:
            model.fit(x, y)
            classes = list(model.classes_)
            proba = model.predict_proba(pred_x)[0]
            scores = {str(d): 0.25 for d in range(10)}
            for klass, p in zip(classes, proba):
                scores[str(int(klass))] += float(p) * 8.0
            result[pos] = normalize_probs(scores)
        except Exception:
            return None
    return result if set(result) == set(POSITIONS) else None


def ensemble_probs(
    draws: Sequence[Draw],
    target_issue: str,
    use_neural: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    models: List[Tuple[str, float, Dict[str, Dict[str, float]]]] = [
        ("长期衰减频率", 0.34, decayed_position_probs(draws)),
        ("近 30 期频率", 0.22, window_position_probs(draws, 30)),
        ("近 80 期频率", 0.12, window_position_probs(draws, 80)),
        ("转移概率", 0.18, transition_position_probs(draws)),
        ("遗漏间隔", 0.14, gap_position_probs(draws)),
    ]
    nn = neural_position_probs(draws, target_issue) if use_neural else None
    if nn:
        models = [(name, weight * 0.82, probs) for name, weight, probs in models]
        models.append(("MLP 神经网络", 0.18, nn))

    combined = {pos: {d: 0.0 for d in DIGITS} for pos in POSITIONS}
    used = []
    for name, weight, probs in models:
        used.append(name)
        for pos in POSITIONS:
            for d in DIGITS:
                combined[pos][d] += weight * probs[pos][d]
    return {pos: normalize_probs(combined[pos]) for pos in POSITIONS}, used


HUNDRED_POSITION_KILL = {
    0: "0926", 1: "0568", 2: "1329", 3: "0225", 4: "2629",
    5: "1229", 6: "1978", 7: "1646", 8: "0424", 9: "2327",
}
TEN_POSITION_KILL = {0: "2", 1: "9", 2: "8", 3: "7", 4: "4", 5: "6", 6: "9", 7: "2", 8: "2", 9: "5"}
UNIT_POSITION_KILL = {
    0: "2363", 1: "0637", 2: "0007", 3: "0824", 4: "2529",
    5: "3740", 6: "5064", 7: "3644", 8: "4252", 9: "4763",
}
ISSUE_TAIL_DAN = {
    0: "3468", 1: "3567", 2: "0469", 3: "2357", 4: "568",
    5: "7894", 6: "124", 7: "345", 8: "4678", 9: "0157",
}
SPAN_DAN = {
    1: "567", 2: "678", 3: "789", 4: "089", 5: "019",
    6: "012", 7: "123", 8: "234", 9: "345",
}
DRAW_POS_DAN = {
    "hundred": {
        0: "2389", 1: "0349", 2: "0145", 3: "1256", 4: "2367",
        5: "3478", 6: "4589", 7: "0569", 8: "0167", 9: "1278",
    },
    "ten": {
        0: "135", 1: "579", 2: "168", 3: "279", 4: "138",
        5: "249", 6: "135", 7: "246", 8: "357", 9: "468",
    },
    "unit": {
        0: "036", 1: "147", 2: "258", 3: "369", 4: "047",
        5: "158", 6: "269", 7: "037", 8: "148", 9: "259",
    },
}
TRIAL_DAN_BY_DIGIT = {
    3: "46", 4: "6958", 5: "189", 6: "47", 7: "80", 8: "02468", 9: "7805",
}
DIGIT_MAP_KILL = {"0": "8", "1": "5", "2": "2", "3": "7", "4": "4", "5": "1", "6": "9", "7": "6", "8": "3", "9": "0"}
SUM_PAIR_TABLE = {
    0: ["00", "19", "28", "37", "46", "55"],
    1: ["01", "29", "38", "47", "56"],
    2: ["02", "11", "39", "48", "57", "66"],
    3: ["03", "12", "49", "58", "67"],
    4: ["04", "13", "22", "59", "68", "77"],
    5: ["05", "14", "23", "69", "78"],
    6: ["06", "15", "24", "33", "79", "88"],
    7: ["07", "16", "25", "34", "89"],
    8: ["08", "17", "26", "35", "44", "99"],
    9: ["09", "18", "27", "36", "45"],
}
DIFF_PAIR_TABLE = {
    0: ["00", "11", "22", "33", "44", "55", "66", "77", "88", "99"],
    1: ["01", "12", "23", "34", "45", "56", "67", "78", "89"],
    2: ["02", "13", "24", "35", "46", "57", "68", "79"],
    3: ["03", "14", "25", "36", "47", "58", "69"],
    4: ["04", "15", "26", "37", "48", "59"],
    5: ["05", "16", "27", "38", "49"],
    6: ["06", "17", "28", "39"],
    7: ["07", "18", "29"],
    8: ["08", "19"],
    9: ["09"],
}


def add_digits_as_dan(signal: StrategySignal, value: object, reason: str, weight: float = 0.3, limit: int = 3) -> None:
    digits = unique_digits_from_text(str(value))[:limit]
    for d in digits:
        signal.add_dan(d, reason, weight)


def nonzero_product(values: Sequence[int]) -> int:
    product_value = 1
    used = False
    for v in values:
        if v != 0:
            product_value *= v
            used = True
    return product_value if used else 1


def classify_trial_ball(trial: str) -> str:
    """没有机球来源字段时，用试机号形态给 1佳1球/1平2球/2进1球/2进2球做近似分类。"""
    ds = digits_of(trial)
    odd = sum(1 for d in ds if d % 2)
    large = sum(1 for d in ds if d >= 5)
    if len(set(ds)) == 2:
        return "1平2球"
    if sorted(ds) in ([0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]):
        return "2进2球"
    if odd == 3 or large >= 2:
        return "2进1球"
    return "1佳1球"


def add_trial_shape_rules(signal: StrategySignal, trial: str, prev_trial: Optional[str]) -> None:
    th, tt, tu = digits_of(trial)
    trial_digits = (th, tt, tu)
    trial_type = classify_trial_ball(trial)
    signal.notes.append(f"试机号形态近似分类: {trial_type}")
    if trial_type == "1佳1球":
        signal.add_dan(7, "1佳1球数字 7 概率参考", 0.35)
        signal.preferred_sum_ranges.append((14, 18, "1佳1球和值 16 左右"))
        if 0 in trial_digits:
            signal.add_dan(0, "1佳1球试机号有 0，开奖号关注 0", 0.6)
        if 3 in trial_digits or 7 in trial_digits:
            signal.add_dan(0, "1佳1球试机号有 3/7，关注 0", 0.35)
            signal.add_dan(9, "1佳1球试机号有 3/7，关注 9", 0.35)
    elif trial_type == "1平2球":
        signal.group3_bonus += 0.06
        signal.add_dan(8, "1平2球数字 8 概率参考", 0.35)
        for d in (1, 3, 7, 0):
            signal.add_dan(d, "1平2球组合参考 13/70", 0.18)
    elif trial_type == "2进1球":
        signal.group3_bonus += 0.08
        signal.add_dan(3, "2进1球数字 3 概率参考", 0.45)
        signal.preferred_sum_ranges.append((8, 11, "2进1球和值 8-11"))
        signal.preferred_sum_ranges.append((16, 20, "2进1球和值 16/20 附近"))
        for d in (8, 9, 2):
            signal.add_dan(d, "2进1球组合参考 38/39/23", 0.18)
    else:
        signal.group6_bonus += 0.04
        signal.add_dan(1, "2进2球数字 1 概率参考", 0.35)
        signal.add_dan(0, "2进2球数字 0 概率参考", 0.35)
        signal.preferred_sum_ranges.append((0, 10, "2进2球和值 10 以下"))
        signal.preferred_sum_ranges.append((18, 23, "2进2球和值 18-23"))

    if prev_trial:
        prev = digits_of(prev_trial)
        multiplied = []
        for a, b in zip(prev, trial_digits):
            if a == 0 and b == 0:
                continue
            multiplied.append((a or 1) * (b or 1))
        if multiplied:
            tails = [v % 10 for v in multiplied]
            if len(set(tails)) < len(tails):
                signal.group3_bonus += 0.06
                signal.notes.append("试机号乘法结果出现同尾/同位，组三倾向加权")
            if any(str(v).zfill(2) == str(v).zfill(2)[::-1] for v in multiplied):
                signal.group3_bonus += 0.04
                signal.notes.append("试机号乘法出现互倒数形态，组三倾向加权")


def apply_formula_strategy(draws: Sequence[Draw], target_issue: str, trial: Optional[str] = None) -> StrategySignal:
    signal = StrategySignal()
    if not draws:
        return signal

    last = draws[-1]
    h, t, u = digits_of(last.number)
    last_sum = number_sum(last.number)
    last_sum_tail = last_sum % 10
    last_span = span(last.number)
    target_tail = issue_tail(target_issue)

    for d in unique_digits_from_text(HUNDRED_POSITION_KILL[h]):
        signal.add_kill("hundred", d, "百位定位全杀法", 1.2)
    for d in unique_digits_from_text(TEN_POSITION_KILL[t]):
        signal.add_kill("ten", d, "十位定位全杀法", 1.2)
    for d in unique_digits_from_text(UNIT_POSITION_KILL[u]):
        signal.add_kill("unit", d, "个位定位全杀法", 0.75)

    signal.add_kill("unit", (last_sum_tail + 4) % 10, "和值尾 + 4 绝杀个位", 1.0)
    signal.add_kill("unit", last_span, "上期跨度绝杀个位", 1.0)
    signal.add_kill("unit", t, "上期十位杀本期个位", 0.9)
    signal.add_kill("ten", (last_sum_tail + last_span) % 10, "和值尾与跨度和杀十位", 1.0)
    signal.add_kill("ten", (target_tail + 4) % 10, "当期期数尾 + 4 杀十位", 0.9)
    signal.add_kill("ten", t, "上期十位杀本期十位", 0.8)
    signal.add_kill("ten", h, "上期百位杀本期十位", 0.8)
    signal.add_kill("ten", u, "上期个位杀本期十位", 0.8)
    signal.add_kill("hundred", (last_sum_tail - 3) % 10, "和值尾 - 3 杀百位", 1.0)
    signal.add_kill("hundred", (h * 3 + 3) % 10, "上期百位 ×3+3 取尾杀百位", 0.9)
    signal.add_kill("hundred", (h * 7 + 7) % 10, "上期百位 ×7+7 取尾杀百位", 0.9)
    signal.add_kill("hundred", (target_tail * 3 + 3) % 10, "期尾 ×3+3 取尾杀百位", 0.8)
    signal.add_kill("hundred", (target_tail * 7 + 6) % 10, "期尾 ×7+6 取尾杀百位", 0.8)
    signal.add_kill("hundred", u, "上期个位杀本期百位", 0.7)
    signal.add_kill("hundred", t, "上期十位杀本期百位", 0.7)
    signal.add_kill("hundred", h, "上期百位杀本期百位", 0.7)
    if len(draws) >= 3:
        signal.add_kill("hundred", int(draws[-3].number[0]), "隔二期百位杀本期百位", 0.7)
    if len(draws) >= 7:
        signal.add_kill("hundred", int(draws[-7].number[0]), "隔六期百位杀本期百位", 0.7)
    add_digits_as_dan(signal, int(last.number) * 123, "开奖号 ×123 所得数取胆", 0.25, 3)
    signal.add_kill("hundred", int(str(int(last.number) * 123)[0]), "开奖号 ×123 所得数第一位杀百位", 0.55)
    signal.add_kill("hundred", (h + u) % 10, "开奖号百位加个位取合杀百位", 0.65)

    signal.add_span_kill(last_sum_tail, "上期和值杀本期跨度", 0.8)
    signal.add_sum_tail_kill((h + u) % 10, "上期开奖号百位 + 个位杀和尾", 0.8)
    signal.add_sum_tail_kill(target_tail, "当期期号尾杀和尾", 0.7)
    signal.add_sum_tail_kill(h, "上期百位杀本期和尾", 0.7)
    signal.add_sum_tail_kill((max(h, t, u) + min(h, t, u)) % 10, "开奖号大号 + 小号杀和尾", 0.7)
    signal.add_sum_tail_kill(last_span, "上期跨度杀和尾", 0.8)
    signal.add_sum_tail_kill(((last_sum * h + 1) % 3), "和值×百位+1 除 3 余数杀和尾", 0.5)
    signal.add_sum_tail_kill(((last_sum * h + 1) % 3 - 3) % 10, "和值×百位+1 除 3 余数-3 杀和尾", 0.5)
    signal.add_sum_tail_kill(sum(int(x) for x in target_issue) % 10, "当期期号相加杀和尾", 0.45)
    signal.add_sum_tail_kill((last_sum_tail * 4) % 10, "前期和尾 ×4 杀和尾", 0.35)
    signal.add_sum_tail_kill(abs(h - t - u) % 10, "上期中奖号差杀和尾", 0.45)
    shifted_sum_tail = ((h + 1) + (t + 1) + (u + 1)) % 10
    signal.add_sum_tail_kill(shifted_sum_tail, "上期奖号百十个各 +1 和值杀尾", 0.45)
    signal.add_sum_tail_kill((12 + t) % 10, "12 + 上期十位杀和值尾", 0.35)
    signal.add_sum_tail_kill((last_sum + ((3 * h + t) % 6)) % 10, "和值 + (3×百位+十位)÷6余数杀和值尾", 0.35)
    signal.add_sum_tail_kill((last_sum + ((4 * h + 9 * t) % 6)) % 10, "和值 + (4×百位+9×十位)÷6余数杀和值尾", 0.35)
    if len(draws) >= 2:
        prev = draws[-2]
        diffs = [abs(a - b) for a, b in zip(digits_of(prev.number), (h, t, u))]
        signal.add_sum_tail_kill(sum(diffs) % 10, "相邻开奖号各位差之和杀和尾", 0.55)
        signal.add_sum_tail_kill(sum(digits_of(f"{abs(int(last.number) - int(prev.number)):03d}"[-3:])) % 10, "相邻开奖号差的各位和杀和尾", 0.45)
        signal.add_sum_tail_kill(abs(sum_tail(prev.number) - last_sum_tail) % 10, "前两期和尾相减杀和尾", 0.35)
        signal.add_sum_tail_kill((sum_tail(prev.number) + last_sum_tail) % 10, "前两期和尾相加杀和尾", 0.35)
        signal.add_pair_kill(prev.number[1] + last.number[1], "杀前两期十位数组合", 1.1)
    if len(draws) >= 3:
        signal.add_sum_tail_kill(sum(sum_tail(x.number) for x in draws[-3:]) % 10, "前三期和尾相加杀和尾", 0.35)
    if len(draws) >= 100:
        signal.add_sum_tail_kill(sum_tail(draws[-100].number), "前 100 期和尾杀和尾", 0.3)

    for d in unique_digits_from_text(ISSUE_TAIL_DAN[target_tail]):
        signal.add_dan(d, "当期期尾数定胆", 0.75)
    if last_span in SPAN_DAN:
        for d in unique_digits_from_text(SPAN_DAN[last_span]):
            signal.add_dan(d, "上期跨度查下期胆码", 0.75)
    for pos, digit in zip(POSITIONS, (h, t, u)):
        for d in unique_digits_from_text(DRAW_POS_DAN[pos][digit]):
            signal.add_dan(d, f"上期开奖号{POSITION_NAMES[pos]}对应胆码", 0.4)

    formula_values = [
        (h * 4 + t * 9 + u * 9 + 3) % 10,
        (u * 2 + 4) % 10,
        (u * 3 + 3) % 10,
        (10 - h) % 10,
        (10 - t) % 10,
        (10 - u) % 10,
    ]
    for d in formula_values:
        signal.add_dan(d, "铁胆/多维定胆公式", 0.35)
    add_digits_as_dan(signal, int(last.number) - 123, "开奖号 -123 取胆", 0.25, 3)
    add_digits_as_dan(signal, int(int(last.number) * 0.618), "开奖号 ×0.618 黄金分割定胆", 0.22, 3)
    add_digits_as_dan(signal, int(int(last.number) * 0.628), "开奖号 ×0.628 定胆", 0.18, 3)
    add_digits_as_dan(signal, int(int(last.number) * 0.314), "开奖号 ×0.314 定胆", 0.18, 3)
    add_digits_as_dan(signal, int(int(last.number) * 0.809), "开奖号 ×0.809 定胆", 0.18, 3)
    add_digits_as_dan(signal, int(int(last.number) * 3.14), "上期奖号 ×3.14 取胆", 0.22, 3)
    add_digits_as_dan(signal, int(int(target_issue) / 5.49), "当期号 ÷5.49 取胆", 0.14, 3)
    add_digits_as_dan(signal, int(int(target_issue) / 4.49), "当期号 ÷4.49 取胆", 0.14, 3)
    add_digits_as_dan(signal, int(int(target_issue[-3:]) / 8), "当期号 ÷8 取胆", 0.14, 2)
    denominator = last_sum + u or 1
    add_digits_as_dan(signal, int(int(last.number) / denominator), "上期开奖号 ÷(和值+个位) 取胆", 0.25, 2)
    add_digits_as_dan(signal, int(int(last.number) / (last_sum + 7 or 1)), "上期奖号 ÷(和值+7) 取胆", 0.18, 3)
    product_value = nonzero_product((h, t, u))
    add_digits_as_dan(signal, int(int(last.number) / product_value), "上期奖号 ÷ 三奖号积取胆", 0.18, 2)
    for d in ((last_sum_tail + 1) % 10, (last_sum_tail - 1) % 10):
        signal.add_dan(d, "和数尾邻码定胆", 0.28)
    for d in ((h + 2) % 10, (t + 2) % 10, (u + 2) % 10, (h - 2) % 10, (t - 2) % 10, (u - 2) % 10):
        signal.add_dan(d, "上期奖号数字 +2/-2 定胆", 0.12)
    signal.add_dan(1, "北京专家胆码 1278", 0.08)
    signal.add_dan(2, "北京专家胆码 1278", 0.08)
    signal.add_dan(7, "北京专家胆码 1278", 0.08)
    signal.add_dan(8, "北京专家胆码 1278", 0.08)
    if len(draws) >= 2:
        avg2 = int((int(draws[-1].number) + int(draws[-2].number)) / 2)
        add_digits_as_dan(signal, avg2, "2 期奖号相加 ÷2 取胆", 0.18, 3)
    if len(draws) >= 3:
        avg3 = int(sum(int(x.number) for x in draws[-3:]) / 3)
        add_digits_as_dan(signal, avg3, "前三期开奖号相加 ÷3 取胆", 0.18, 3)
    if len(draws) >= 4:
        avg4 = int(sum(int("".join(sorted(x.number))) for x in draws[-4:]) / 4)
        add_digits_as_dan(signal, avg4, "4 期奖号从小到大相加 ÷4 取胆", 0.16, 3)
    if len(draws) >= 30:
        add_digits_as_dan(signal, draws[-20].number + draws[-10].number, "前 20 期 + 前 10 期奖号作预选胆", 0.16, 5)

    # 杀二码组合：按组合出现即扣分，不是单码杀号。
    issue_last2 = target_issue[-2:]
    signal.add_pair_kill(issue_last2, "杀当期期数后两位数", 1.0)
    signal.add_pair_kill(f"{target_tail}{(target_tail + last_sum_tail) % 10}", "期尾加上期和值合数尾杀二码", 0.8)
    mapped = "".join(DIGIT_MAP_KILL[d] for d in last.number)
    signal.add_pair_kill(mapped[:2], "上期开奖号对应数杀组合", 0.55)
    signal.add_pair_kill(mapped[1:], "上期开奖号对应数杀组合", 0.55)
    ordered = sorted((h, t, u))
    signal.add_pair_kill(f"{ordered[1]}{ordered[2] - ordered[0]}", "大小序小中大跨度组合杀二码", 0.45)
    for pair in SUM_PAIR_TABLE.get((target_tail + last_sum_tail) % 10, []):
        signal.add_pair_kill(pair, "两码组合之和速查表辅助杀号", 0.12)
    for pair in DIFF_PAIR_TABLE.get(last_span, []):
        signal.add_pair_kill(pair, "两码组合之差速查表辅助杀号", 0.10)

    # 出组三规律与胆码关联。
    if u in (8, 4):
        signal.group3_bonus += 0.03
        signal.notes.append("个位出 8/4 后 1-5 期内组三倾向")
    if t == 0:
        signal.group3_bonus += 0.03
        signal.notes.append("十位出 0 后 1-3 期内组三倾向")
    if h == 8:
        for d in (1, 0, 9):
            signal.add_dan(d, "百位出 8 后下期胆码 109", 0.25)
    if h == 6:
        for d in (9, 7, 4):
            signal.add_dan(d, "百位出 6 后下期胆码 974", 0.25)
    if h == 0:
        for d in (4, 9, 3):
            signal.add_dan(d, "百位出 0 后下期胆码 493", 0.25)
    if t == 6:
        for d in (7, 3, 4):
            signal.add_dan(d, "十位出 6 后下期胆码 734", 0.25)
    if u == 8:
        for d in (8, 4, 6):
            signal.add_dan(d, "个位出 8 后下期胆码 846", 0.25)
    if u == 2:
        for d in (8, 3, 1):
            signal.add_dan(d, "个位出 2 后下期胆码 831", 0.25)
    if u == 9:
        for d in (8, 4, 0):
            signal.add_dan(d, "个位出 9 后下期胆码 840", 0.25)
    if u == 1:
        for d in (1, 0, 2):
            signal.add_dan(d, "个位出 1 后下期胆码 102", 0.25)

    if trial:
        trial_num = normalize_number(trial)
        th, tt, tu = digits_of(trial_num)
        trial_sum = number_sum(trial_num)
        signal.add_sum_tail_kill(trial_sum % 10, "当天试机号和值杀和尾", 0.75)
        if th == 0:
            signal.add_kill("hundred", max(th, tt, tu), "试机号百位为 0，取试机号最大数通杀一码", 0.5)
        else:
            kill_digit = int(str(int(trial_num) // th)[0]) if th else max(th, tt, tu)
            signal.add_kill("hundred", kill_digit, "试机号除百位取首位通杀一码", 0.45)

        if 8 in (th, tt, tu):
            signal.preferred_sum_ranges.append((12, 17, "试机号有 8，和值偏向 12-17"))
        if 6 in (th, tt, tu):
            signal.group3_bonus += 0.05
            signal.preferred_sum_ranges.append((14, 27, "试机号有 6，偏向组三或大和值"))
        for td in set((th, tt, tu)):
            if td == 1:
                for pos in POSITIONS:
                    signal.add_kill(pos, 1, "试机号有 1，多数规律排除 1", 0.25)
            if td == 2:
                signal.add_dan(2, "试机号有 2，口诀提示关注 2", 0.45)
            if td == 0:
                signal.add_kill("hundred", 0, "试金号口诀：见 0 多半没有 0", 0.12)
                signal.add_kill("ten", 0, "试金号口诀：见 0 多半没有 0", 0.12)
                signal.add_kill("unit", 0, "试金号口诀：见 0 多半没有 0", 0.12)
            if td == 3:
                signal.add_dan(3, "开叁数字都有用", 0.24)
            if td == 7:
                signal.add_dan(0, "7 试 8 见还有 0", 0.22)
                signal.add_dan(8, "7 试 8 见还有 0", 0.22)
            for d in unique_digits_from_text(TRIAL_DAN_BY_DIGIT.get(td, "")):
                signal.add_dan(d, f"试机号含 {td} 的胆码规律", 0.45)
        add_digits_as_dan(signal, int(trial_num) // 3, "当期试机号 ÷3 取胆", 0.28, 3)
        add_digits_as_dan(signal, int(int(trial_num) / 3.14), "当期试机号 ÷3.14 取胆", 0.22, 3)
        add_digits_as_dan(signal, int(trial_num) * 3, "当期试机号 ×3 取胆", 0.20, 3)
        add_digits_as_dan(signal, int(trial_num) * 9, "当期试机号 ×9 取胆", 0.16, 4)
        trial_mapped = "".join(DIGIT_MAP_KILL[d] for d in trial_num)
        signal.add_pair_kill(trial_mapped[:2], "上期/当期试机号对应数杀组合", 0.45)
        signal.add_pair_kill(trial_mapped[1:], "上期/当期试机号对应数杀组合", 0.45)
        for d in ((min(th, tt, tu) - max(th, tt, tu)) % 10, abs(min(th, tt, tu) - max(th, tt, tu)) % 10):
            signal.add_dan(d, "试机号小号 - 大号绝对值定胆", 0.22)
        prev_trial = next((d.trial for d in reversed(draws) if d.trial), None)
        add_trial_shape_rules(signal, trial_num, prev_trial)
        if prev_trial:
            pth, ptt, ptu = digits_of(prev_trial)
            signal.add_sum_tail_kill((trial_sum + ptu) % 10, "当天试机号和 + 头天试机号个位杀和尾", 0.45)

    return signal


def strategy_adjusted_probs(
    probs: Dict[str, Dict[str, float]],
    signal: StrategySignal,
    kill_threshold: float = 1.6,
) -> Dict[str, Dict[str, float]]:
    adjusted: Dict[str, Dict[str, float]] = {}
    for pos in POSITIONS:
        scores = {}
        for d in DIGITS:
            score = probs[pos][d]
            kill_w = signal.kill_by_pos[pos][d]
            dan_w = signal.dan_digits[d]
            score *= max(0.08, 1.0 - 0.24 * min(kill_w, 3.0))
            score *= 1.0 + 0.09 * min(dan_w, 6.0)
            if kill_w >= kill_threshold:
                score *= 0.45
            scores[d] = score
        adjusted[pos] = normalize_probs(scores)
    return adjusted


def combo_type(num: str) -> str:
    uniq = len(set(num))
    if uniq == 1:
        return "豹子"
    if uniq == 2:
        return "组三"
    return "组六"


def combo_score(
    combo: str,
    probs: Dict[str, Dict[str, float]],
    signal: StrategySignal,
    hard_kill_threshold: float = 2.15,
) -> Tuple[float, List[str], bool]:
    h, t, u = combo
    reasons: List[str] = []
    hard_reject = False
    for pos, d in zip(POSITIONS, combo):
        if signal.kill_by_pos[pos][d] >= hard_kill_threshold:
            hard_reject = True
            reasons.append(f"{POSITION_NAMES[pos]}被多策略杀号")
    st = str((int(h) + int(t) + int(u)) % 10)
    sp = str(max(map(int, combo)) - min(map(int, combo)))
    if signal.kill_sum_tails[st] >= 1.45:
        hard_reject = True
        reasons.append("和值尾被多策略杀号")
    if signal.kill_spans[sp] >= 1.45:
        hard_reject = True
        reasons.append("跨度被策略杀号")
    combo_digits = sorted(combo)
    for i in range(3):
        for j in range(i + 1, 3):
            pair = "".join(sorted((combo_digits[i], combo_digits[j])))
            pair_w = signal.kill_pairs[pair]
            if pair_w >= 1.0:
                score_penalty = "强"
            else:
                score_penalty = ""
            if pair_w >= 1.35:
                hard_reject = True
                reasons.append(f"含{score_penalty}杀号二码组合 {pair}")

    score = probs["hundred"][h] * probs["ten"][t] * probs["unit"][u]
    for i in range(3):
        for j in range(i + 1, 3):
            pair = "".join(sorted((combo_digits[i], combo_digits[j])))
            score *= max(0.22, 1.0 - 0.22 * min(signal.kill_pairs[pair], 3.0))
    dan_hits = len(set(combo) & set(signal.dan_digits))
    score *= 1.0 + 0.08 * sum(signal.dan_digits[d] for d in set(combo))
    if dan_hits:
        reasons.append(f"含胆码 {''.join(sorted(set(combo) & set(signal.dan_digits)))}")
    if signal.preferred_sum_ranges:
        s = sum(map(int, combo))
        if any(lo <= s <= hi for lo, hi, _ in signal.preferred_sum_ranges):
            score *= 1.08
            reasons.append("符合试机号和值区间")
        else:
            score *= 0.94
    if signal.group3_bonus and combo_type(combo) == "组三":
        score *= 1.0 + signal.group3_bonus
        reasons.append("试机号组三倾向加权")
    if signal.group6_bonus and combo_type(combo) == "组六":
        score *= 1.0 + signal.group6_bonus
        reasons.append("试机号组六倾向加权")
    return score, reasons, hard_reject


def rank_combos(
    probs: Dict[str, Dict[str, float]],
    signal: StrategySignal,
    limit: int = 200,
    keep_effective_limit: int = 300,
) -> Tuple[List[Dict], List[Dict]]:
    ranked: List[Dict] = []
    rejected = 0
    for h, t, u in product(DIGITS, repeat=3):
        combo = f"{h}{t}{u}"
        score, reasons, rejected_by_strategy = combo_score(combo, probs, signal)
        row = {
            "number": combo,
            "score": score,
            "probability_basis": probs["hundred"][h] * probs["ten"][t] * probs["unit"][u],
            "sum": int(h) + int(t) + int(u),
            "span": max(int(h), int(t), int(u)) - min(int(h), int(t), int(u)),
            "type": combo_type(combo),
            "strategy_reasons": reasons,
        }
        if rejected_by_strategy:
            rejected += 1
            continue
        ranked.append(row)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    total_score = sum(x["score"] for x in ranked) or 1.0
    for row in ranked:
        row["normalized_probability"] = row["score"] / total_score
    return ranked[:limit], ranked[:keep_effective_limit]


def rank_all_by_position_probs(probs: Dict[str, Dict[str, float]], limit: int = 1000) -> List[Dict]:
    rows = []
    for h, t, u in product(DIGITS, repeat=3):
        num = f"{h}{t}{u}"
        score = probs["hundred"][h] * probs["ten"][t] * probs["unit"][u]
        rows.append(
            {
                "number": num,
                "score": score,
                "sum": int(h) + int(t) + int(u),
                "span": max(int(h), int(t), int(u)) - min(int(h), int(t), int(u)),
                "type": combo_type(num),
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    total = sum(x["score"] for x in rows) or 1.0
    for row in rows:
        row["normalized_probability"] = row["score"] / total
    return rows[:limit]


def strategy_only_probs(signal: StrategySignal) -> Dict[str, Dict[str, float]]:
    probs: Dict[str, Dict[str, float]] = {}
    for pos in POSITIONS:
        scores = {}
        for d in DIGITS:
            score = 1.0
            score *= max(0.06, 1.0 - 0.26 * min(signal.kill_by_pos[pos][d], 4.0))
            score *= 1.0 + 0.16 * min(signal.dan_digits[d], 8.0)
            scores[d] = score
        probs[pos] = normalize_probs(scores)
    return probs


def compare_strategy_and_neural(strategy_rows: Sequence[Dict], neural_rows: Sequence[Dict], limit: int = 300) -> List[Dict]:
    neural_rank = {row["number"]: idx + 1 for idx, row in enumerate(neural_rows)}
    matches = []
    for idx, row in enumerate(strategy_rows):
        num = row["number"]
        if num in neural_rank:
            matches.append(
                {
                    "number": num,
                    "strategy_rank": idx + 1,
                    "neural_rank": neural_rank[num],
                    "type": row.get("type"),
                    "sum": row.get("sum"),
                    "span": row.get("span"),
                    "strategy_reasons": row.get("strategy_reasons", [])[:6],
                }
            )
        if len(matches) >= limit:
            break
    matches.sort(key=lambda x: (x["strategy_rank"], x["neural_rank"]))
    return matches


def sorted_position_probs(probs: Dict[str, Dict[str, float]]) -> Dict[str, List[Dict[str, float]]]:
    return {
        pos: [
            {"digit": d, "probability": probs[pos][d]}
            for d in sorted(DIGITS, key=lambda x: probs[pos][x], reverse=True)
        ]
        for pos in POSITIONS
    }


def best_position_sets(probs: Dict[str, Dict[str, float]], max_notes: int = 20) -> List[Dict]:
    sorted_digits = {pos: sorted(DIGITS, key=lambda d: probs[pos][d], reverse=True) for pos in POSITIONS}
    best: List[Dict] = []
    for kh in range(1, 11):
        for kt in range(1, 11):
            for ku in range(1, 11):
                notes = kh * kt * ku
                if notes > max_notes:
                    continue
                sh = sum(probs["hundred"][d] for d in sorted_digits["hundred"][:kh])
                st = sum(probs["ten"][d] for d in sorted_digits["ten"][:kt])
                su = sum(probs["unit"][d] for d in sorted_digits["unit"][:ku])
                best.append(
                    {
                        "counts": {"hundred": kh, "ten": kt, "unit": ku},
                        "notes": notes,
                        "estimated_cover_probability": sh * st * su,
                        "digits": {
                            "hundred": sorted_digits["hundred"][:kh],
                            "ten": sorted_digits["ten"][:kt],
                            "unit": sorted_digits["unit"][:ku],
                        },
                    }
                )
    best.sort(key=lambda x: (x["estimated_cover_probability"], x["notes"]), reverse=True)
    return best[:8]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, data: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_prediction_for_issue(record_dir: str, issue: str) -> Optional[Dict]:
    path = os.path.join(record_dir, "predictions.jsonl")
    rows = [r for r in load_jsonl(path) if r.get("issue") == issue]
    return rows[-1] if rows else None


def prediction_payload(
    draws: Sequence[Draw],
    target_issue: str,
    trial: Optional[str],
    budget: int,
    effective_limit: int,
    use_neural: bool = True,
) -> Dict:
    target_issue = normalize_issue(target_issue)
    trial = normalize_number(trial) if trial else None
    base_probs, models = ensemble_probs(draws, target_issue, use_neural=use_neural)
    signal = apply_formula_strategy(draws, target_issue, trial)
    adjusted = strategy_adjusted_probs(base_probs, signal)
    top_ranked, effective = rank_combos(adjusted, signal, max(budget, 50), effective_limit)
    neural_ranked = rank_all_by_position_probs(base_probs, effective_limit)
    markov_probs = markov_position_probs(draws)
    markov_ranked = rank_all_by_position_probs(markov_probs, effective_limit)
    markov_top5_digits = markov_digit_probs(markov_probs, 5)
    pure_strategy_probs = strategy_only_probs(signal)
    strategy_ranked, strategy_effective = rank_combos(pure_strategy_probs, signal, max(effective_limit, budget, 400), effective_limit)
    strategy_neural_matches = compare_strategy_and_neural(strategy_effective, neural_ranked, 300)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "issue": target_issue,
        "trial": trial,
        "history_count": len(draws),
        "last_draw": {"issue": draws[-1].issue, "number": draws[-1].number} if draws else None,
        "models": models,
        "position_probabilities": sorted_position_probs(adjusted),
        "base_position_probabilities": sorted_position_probs(base_probs),
        "position_selection_plans": best_position_sets(adjusted, max_notes=budget),
        "budget": budget,
        "recommended_numbers": top_ranked[:budget],
        "neural_prediction_numbers": neural_ranked,
        "markov_prediction_numbers": markov_ranked,
        "markov_position_probabilities": sorted_position_probs(markov_probs),
        "markov_top5_digits": markov_top5_digits,
        "strategy_prediction_numbers": strategy_effective,
        "strategy_neural_matches": strategy_neural_matches,
        "strategy": {
            "kill_by_position": {pos: dict(signal.kill_by_pos[pos]) for pos in POSITIONS},
            "kill_sum_tails": dict(signal.kill_sum_tails),
            "kill_spans": dict(signal.kill_spans),
            "kill_pairs": dict(signal.kill_pairs),
            "dan_digits": dict(signal.dan_digits),
            "preferred_sum_ranges": signal.preferred_sum_ranges,
            "notes": signal.notes[:200],
        },
        "strategy_filtered_effective_combinations": effective,
    }
    selected_dan, all_dan = select_strategy_dan_digits(signal.dan_digits)
    payload["strategy_dan_digits"] = selected_dan
    payload["strategy_dan_candidates"] = all_dan
    return payload


def rank_number_in_prediction(payload: Dict, number: str) -> Dict:
    number = normalize_number(number)
    recommended = [x["number"] for x in payload.get("recommended_numbers", [])]
    effective = [x["number"] for x in payload.get("strategy_filtered_effective_combinations", [])]
    recommended_rank = recommended.index(number) + 1 if number in recommended else None
    effective_rank = effective.index(number) + 1 if number in effective else None
    return {
        "issue": payload.get("issue"),
        "number": number,
        "recommended_rank": recommended_rank,
        "effective_rank": effective_rank,
        "is_recommended_hit": recommended_rank is not None,
        "is_effective_hit": effective_rank is not None,
        "recommended_count": len(recommended),
        "effective_combo_count": len(effective),
    }


def rank_number_in_rows(rows: Sequence[Dict], number: str) -> Optional[int]:
    number = normalize_number(number)
    for idx, row in enumerate(rows):
        if row.get("number") == number:
            return idx + 1
    return None


def strategy_dan_hit_info(payload: Dict, number: str) -> Dict:
    number = normalize_number(number)
    dan_list = payload.get("strategy_dan_digits", [])
    if dan_list:
        dan_digits = set(str(x.get("digit")) for x in dan_list)
    else:
        dan_counter = payload.get("strategy", {}).get("dan_digits", {}) or {}
        dan_digits = set(str(d) for d in dan_counter.keys())
    position_hits = {}
    hit_digits = []
    for pos, digit in zip(POSITIONS, number):
        hit = digit in dan_digits
        position_hits[pos] = {"digit": digit, "hit": hit}
        if hit:
            hit_digits.append(digit)
    return {
        "dan_digits": sorted(dan_digits),
        "hit_digits": hit_digits,
        "unique_hit_digits": sorted(set(hit_digits)),
        "hit_count": len(hit_digits),
        "hit_rate": len(hit_digits) / 3.0,
        "position_hits": position_hits,
        "all_three_digits_hit": len(hit_digits) == 3,
    }


def markov_top5_digit_hit_info(payload: Dict, number: str) -> Dict:
    number = normalize_number(number)
    digits = [str(x.get("digit")) for x in payload.get("markov_top5_digits", [])]
    digit_set = set(digits)
    hit_digits = [digit for digit in number if digit in digit_set]
    return {
        "digits": digits,
        "hit_digits": hit_digits,
        "unique_hit_digits": sorted(set(hit_digits)),
        "hit_count": len(hit_digits),
        "hit_rate": len(hit_digits) / 3.0,
        "all_three_digits_hit": len(hit_digits) == 3,
    }


def select_strategy_dan_digits(counter: Counter) -> Tuple[List[Dict], List[Dict]]:
    all_digits = [
        {"digit": digit, "weight": counter[digit]}
        for digit in sorted(counter, key=lambda d: (counter[d], d), reverse=True)
    ]
    if not all_digits:
        return [], []
    max_weight = all_digits[0]["weight"]
    threshold = max(1.0, max_weight * 0.45)
    selected = [x for x in all_digits if x["weight"] >= threshold]
    if len(selected) < 3:
        selected = all_digits[: min(5, len(all_digits))]
    return selected, all_digits


def position_probability_ranks(payload: Dict, number: str) -> Dict[str, Optional[int]]:
    number = normalize_number(number)
    ranks: Dict[str, Optional[int]] = {}
    probs = payload.get("position_probabilities", {})
    for pos, digit in zip(POSITIONS, number):
        ordered = [str(x.get("digit")) for x in probs.get(pos, [])]
        ranks[pos] = ordered.index(digit) + 1 if digit in ordered else None
    return ranks


def print_prediction_summary(payload: Dict) -> None:
    print(f"期号: {payload['issue']}")
    print(f"历史样本: {payload['history_count']}  使用模型: {', '.join(payload['models'])}")
    if payload.get("trial"):
        print(f"试机号: {payload['trial']}")
    print("\n各位置数字概率（从高到低）:")
    for pos in POSITIONS:
        items = payload["position_probabilities"][pos]
        text = " ".join([f"{x['digit']}:{x['probability']:.2%}" for x in items])
        print(f"{POSITION_NAMES[pos]}: {text}")
    print("\n每位选 1-10 个数字时，预算内覆盖概率最高的方案:")
    for plan in payload["position_selection_plans"][:5]:
        c = plan["counts"]
        d = plan["digits"]
        print(
            f"{c['hundred']}x{c['ten']}x{c['unit']}={plan['notes']} 注 "
            f"估计覆盖 {plan['estimated_cover_probability']:.2%} | "
            f"百{''.join(d['hundred'])} 十{''.join(d['ten'])} 个{''.join(d['unit'])}"
        )
    print(f"\n预算 {payload['budget']} 注推荐号码:")
    print(" ".join(x["number"] for x in payload["recommended_numbers"][: payload["budget"]]))
    print(f"\n策略过滤后有效组合数量: {len(payload['strategy_filtered_effective_combinations'])}")


def settle_prediction(record_dir: str, issue: str, actual: str) -> Dict:
    issue = normalize_issue(issue)
    actual = normalize_number(actual)
    pred = latest_prediction_for_issue(record_dir, issue)
    if not pred:
        raise ValueError(f"未找到期号 {issue} 的预测记录，请先运行 predict。")

    results = []
    counts = {
        "hundred_hits": 0,
        "ten_hits": 0,
        "unit_hits": 0,
        "any_position_hits": 0,
        "all_position_hits": 0,
        "recommended_all_hits": 0,
        "effective_all_hits": 0,
    }
    effective = pred.get("strategy_filtered_effective_combinations", [])
    rank_info = rank_number_in_prediction(pred, actual)
    neural_rank = rank_number_in_rows(pred.get("neural_prediction_numbers", []), actual)
    markov_rank = rank_number_in_rows(pred.get("markov_prediction_numbers", []), actual)
    strategy_rank = rank_number_in_rows(pred.get("strategy_prediction_numbers", []), actual)
    match_rank = rank_number_in_rows(pred.get("strategy_neural_matches", []), actual)
    dan_hit_info = strategy_dan_hit_info(pred, actual)
    markov_top5_hit_info = markov_top5_digit_hit_info(pred, actual)
    pos_prob_ranks = position_probability_ranks(pred, actual)
    recommended_set = {x["number"] for x in pred.get("recommended_numbers", [])}
    for row in effective:
        num = row["number"]
        pos_hit = {
            "hundred": num[0] == actual[0],
            "ten": num[1] == actual[1],
            "unit": num[2] == actual[2],
        }
        full_hit = all(pos_hit.values())
        counts["hundred_hits"] += int(pos_hit["hundred"])
        counts["ten_hits"] += int(pos_hit["ten"])
        counts["unit_hits"] += int(pos_hit["unit"])
        counts["any_position_hits"] += int(any(pos_hit.values()))
        counts["all_position_hits"] += int(full_hit)
        counts["effective_all_hits"] += int(full_hit)
        if num in recommended_set and full_hit:
            counts["recommended_all_hits"] += 1
        results.append(
            {
                "number": num,
                "position_hits": pos_hit,
                "position_hit_count": sum(pos_hit.values()),
                "full_hit": full_hit,
                "is_recommended": num in recommended_set,
                "score": row.get("score"),
                "strategy_reasons": row.get("strategy_reasons", []),
            }
        )

    summary = {
        "settled_at": datetime.now().isoformat(timespec="seconds"),
        "issue": issue,
        "actual": actual,
        "prediction_created_at": pred.get("created_at"),
        "trial": pred.get("trial"),
        "budget": pred.get("budget"),
        "effective_combo_count": len(effective),
        "recommended_numbers": [x["number"] for x in pred.get("recommended_numbers", [])],
        "actual_in_recommended": actual in recommended_set,
        "actual_in_effective_combinations": any(x["number"] == actual for x in effective),
        "actual_in_neural_prediction": neural_rank is not None,
        "actual_in_markov_prediction": markov_rank is not None,
        "actual_in_strategy_prediction": strategy_rank is not None,
        "actual_in_strategy_neural_match": match_rank is not None,
        "rank_info": rank_info,
        "neural_rank": neural_rank,
        "markov_rank": markov_rank,
        "strategy_rank": strategy_rank,
        "strategy_neural_match_rank": match_rank,
        "strategy_dan_hit": dan_hit_info,
        "markov_top5_digit_hit": markov_top5_hit_info,
        "position_probability_ranks": pos_prob_ranks,
        "counts": counts,
        "results": results,
    }
    ensure_dir(record_dir)
    append_jsonl(os.path.join(record_dir, "settlements.jsonl"), summary)
    rebuild_stats(record_dir)
    return summary


def rebuild_stats(record_dir: str) -> Dict:
    rows = load_jsonl(os.path.join(record_dir, "settlements.jsonl"))
    rolling_path = os.path.join(record_dir, "rolling_predictions.jsonl")
    rolling_rows = load_jsonl(rolling_path)
    rolling_stats_path = os.path.join(record_dir, "rolling_stats.json")
    rolling_stats = {}
    if os.path.exists(rolling_stats_path):
        with open(rolling_stats_path, "r", encoding="utf-8") as f:
            rolling_stats = json.load(f)
    stats = {
        "settlement_count": len(rows),
        "recommended_full_hits": 0,
        "effective_full_hits": 0,
        "effective_top150_hits": 0,
        "recommended_hit_issues": [],
        "effective_hit_issues": [],
        "position_hits_total": {"hundred": 0, "ten": 0, "unit": 0},
        "effective_combo_total": 0,
        "historical_prediction_total": len(rolling_rows),
        "historical_recommended_hits": 0,
        "historical_effective_hits": 0,
        "historical_effective_top150_hits": 0,
    }
    for row in rows:
        counts = row.get("counts", {})
        stats["recommended_full_hits"] += int(row.get("actual_in_recommended", False))
        stats["effective_full_hits"] += int(row.get("actual_in_effective_combinations", False))
        effective_rank = (row.get("rank_info") or {}).get("effective_rank")
        stats["effective_top150_hits"] += int(bool(effective_rank and effective_rank <= 150))
        if row.get("actual_in_recommended"):
            stats["recommended_hit_issues"].append(row["issue"])
        if row.get("actual_in_effective_combinations"):
            stats["effective_hit_issues"].append(row["issue"])
        stats["position_hits_total"]["hundred"] += counts.get("hundred_hits", 0)
        stats["position_hits_total"]["ten"] += counts.get("ten_hits", 0)
        stats["position_hits_total"]["unit"] += counts.get("unit_hits", 0)
        stats["effective_combo_total"] += row.get("effective_combo_count", 0)
    for row in rolling_rows:
        rank_info = row.get("rank_info", {})
        stats["historical_recommended_hits"] += int(rank_info.get("is_recommended_hit", False))
        stats["historical_effective_hits"] += int(rank_info.get("is_effective_hit", False))
        effective_rank = rank_info.get("effective_rank")
        stats["historical_effective_top150_hits"] += int(bool(effective_rank and effective_rank <= 150))
    stats["historical_neural_hits"] = sum(1 for row in rolling_rows if row.get("neural_rank"))
    stats["historical_markov_hits"] = sum(1 for row in rolling_rows if row.get("markov_rank"))
    stats["historical_strategy_hits"] = sum(1 for row in rolling_rows if row.get("strategy_rank"))
    stats["historical_strategy_dan_all3_hits"] = sum(1 for row in rolling_rows if (row.get("strategy_dan_hit") or {}).get("all_three_digits_hit"))
    stats["historical_strategy_dan_position_hits"] = sum((row.get("strategy_dan_hit") or {}).get("hit_count", 0) for row in rolling_rows)
    if stats["settlement_count"]:
        stats["recommended_full_hit_rate_by_issue"] = stats["recommended_full_hits"] / stats["settlement_count"]
        stats["effective_full_hit_rate_by_issue"] = stats["effective_full_hits"] / stats["settlement_count"]
    else:
        stats["recommended_full_hit_rate_by_issue"] = 0
        stats["effective_full_hit_rate_by_issue"] = 0
    if stats["historical_prediction_total"]:
        stats["historical_recommended_hit_rate"] = stats["historical_recommended_hits"] / stats["historical_prediction_total"]
        stats["historical_effective_hit_rate"] = stats["historical_effective_hits"] / stats["historical_prediction_total"]
        stats["historical_effective_top150_hit_rate"] = stats["historical_effective_top150_hits"] / stats["historical_prediction_total"]
    else:
        stats["historical_recommended_hit_rate"] = 0
        stats["historical_effective_hit_rate"] = 0
        stats["historical_effective_top150_hit_rate"] = 0
    stats["all_prediction_total"] = stats["historical_prediction_total"] + stats["settlement_count"]
    stats["all_recommended_hits"] = stats["historical_recommended_hits"] + stats["recommended_full_hits"]
    stats["all_effective_hits"] = stats["historical_effective_hits"] + stats["effective_full_hits"]
    stats["all_effective_top150_hits"] = stats["historical_effective_top150_hits"] + stats["effective_top150_hits"]
    if rolling_stats.get("rank_buckets"):
        stats["rank_buckets"] = rolling_stats["rank_buckets"]
    ensure_dir(record_dir)
    json_dump(os.path.join(record_dir, "stats.json"), stats)
    return stats


def next_issue_after(issue: str) -> str:
    issue = normalize_issue(issue)
    year = int(issue[:4])
    seq = int(issue[-3:]) + 1
    if seq > 999:
        year += 1
        seq = 1
    return f"{year}{seq:03d}"


def backtest(draws: Sequence[Draw], start: int, budget: int, effective_limit: int) -> Dict:
    if start < 5:
        start = 5
    rows = []
    rec_hits = 0
    eff_hits = 0
    for i in range(start, len(draws)):
        history = draws[:i]
        target = draws[i]
        payload = prediction_payload(history, target.issue, target.trial, budget, effective_limit, use_neural=False)
        rec_set = {x["number"] for x in payload["recommended_numbers"]}
        eff_set = {x["number"] for x in payload["strategy_filtered_effective_combinations"]}
        rec_hit = target.number in rec_set
        eff_hit = target.number in eff_set
        rec_hits += int(rec_hit)
        eff_hits += int(eff_hit)
        rows.append(
            {
                "issue": target.issue,
                "actual": target.number,
                "trial": target.trial,
                "recommended_hit": rec_hit,
                "effective_hit": eff_hit,
                "top_recommended": [x["number"] for x in payload["recommended_numbers"][:budget]],
            }
        )
    return {
        "tested_issues": len(rows),
        "budget": budget,
        "effective_limit": effective_limit,
        "recommended_hits": rec_hits,
        "effective_hits": eff_hits,
        "recommended_hit_rate": rec_hits / len(rows) if rows else 0,
        "effective_hit_rate": eff_hits / len(rows) if rows else 0,
        "rows": rows,
    }


def rolling_train_records(
    draws: Sequence[Draw],
    budget: int,
    effective_limit: int,
    start_index: int = 1,
    use_neural: bool = False,
    progress_every: int = 500,
) -> Tuple[List[Dict], Dict]:
    if len(draws) < 2:
        raise ValueError("至少需要 2 期历史数据，才能从第 2 期开始滚动预测。")
    rows: List[Dict] = []
    recommended_hits = 0
    effective_hits = 0
    effective_top150_hits = 0
    neural_hits = 0
    strategy_hits = 0
    strategy_match_hits = 0
    strategy_dan_all3_hits = 0
    strategy_dan_position_hits = 0
    pos_hits = {"hundred": 0, "ten": 0, "unit": 0}
    start_index = max(1, start_index)
    for i in range(start_index, len(draws)):
        history = draws[:i]
        target = draws[i]
        payload = prediction_payload(history, target.issue, None, budget, effective_limit, use_neural=use_neural)
        rank_info = rank_number_in_prediction(payload, target.number)
        neural_rank = rank_number_in_rows(payload.get("neural_prediction_numbers", []), target.number)
        markov_rank = rank_number_in_rows(payload.get("markov_prediction_numbers", []), target.number)
        strategy_rank = rank_number_in_rows(payload.get("strategy_prediction_numbers", []), target.number)
        match_rank = rank_number_in_rows(payload.get("strategy_neural_matches", []), target.number)
        dan_hit_info = strategy_dan_hit_info(payload, target.number)
        markov_top5_hit_info = markov_top5_digit_hit_info(payload, target.number)
        pos_rank = position_probability_ranks(payload, target.number)
        recommended_hits += int(rank_info["is_recommended_hit"])
        effective_hits += int(rank_info["is_effective_hit"])
        effective_top150_hits += int(bool(rank_info["effective_rank"] and rank_info["effective_rank"] <= 150))
        neural_hits += int(neural_rank is not None)
        strategy_hits += int(strategy_rank is not None)
        strategy_match_hits += int(match_rank is not None)
        strategy_dan_all3_hits += int(dan_hit_info["all_three_digits_hit"])
        strategy_dan_position_hits += dan_hit_info["hit_count"]
        top1 = payload["recommended_numbers"][0]["number"] if payload.get("recommended_numbers") else None
        if top1:
            pos_hits["hundred"] += int(top1[0] == target.number[0])
            pos_hits["ten"] += int(top1[1] == target.number[1])
            pos_hits["unit"] += int(top1[2] == target.number[2])
        rows.append(
            {
                "issue": target.issue,
                "actual": target.number,
                "history_count": len(history),
                "models": payload.get("models", []),
                "budget": budget,
                "rank_info": rank_info,
                "neural_rank": neural_rank,
                "markov_rank": markov_rank,
                "strategy_rank": strategy_rank,
                "strategy_neural_match_rank": match_rank,
                "strategy_dan_hit": dan_hit_info,
                "markov_top5_digits": payload.get("markov_top5_digits", []),
                "markov_top5_digit_hit": markov_top5_hit_info,
                "position_probability_ranks": pos_rank,
                "top_recommended": [x["number"] for x in payload.get("recommended_numbers", [])[:budget]],
                "top_neural": [x["number"] for x in payload.get("neural_prediction_numbers", [])[:budget]],
                "top_markov": [x["number"] for x in payload.get("markov_prediction_numbers", [])[:budget]],
                "top_strategy": [x["number"] for x in payload.get("strategy_prediction_numbers", [])[:budget]],
                "strategy_dan_digits": payload.get("strategy_dan_digits", []),
                "top1": top1,
            }
        )
        if progress_every and len(rows) % progress_every == 0:
            print(f"已滚动预测 {len(rows)} / {len(draws) - start_index} 期...")
    total = len(rows)
    rank_cutoffs = [20, 100, 150, 200, 300, 400]
    rank_buckets = {}
    for name, key in (("neural", "neural_rank"), ("markov", "markov_rank"), ("strategy", "strategy_rank"), ("strategy_match", "strategy_neural_match_rank")):
        bucket = {}
        for cutoff in rank_cutoffs:
            hits = sum(1 for row in rows if row.get(key) and row.get(key) <= cutoff)
            bucket[f"top_{cutoff}_hits"] = hits
            bucket[f"top_{cutoff}_hit_rate"] = hits / total if total else 0
        rank_buckets[name] = bucket
    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "first_predicted_issue": rows[0]["issue"] if rows else None,
        "last_predicted_issue": rows[-1]["issue"] if rows else None,
        "total_predictions": total,
        "budget": budget,
        "effective_limit": effective_limit,
        "use_neural": use_neural,
        "recommended_hits": recommended_hits,
        "effective_hits": effective_hits,
        "effective_top150_hits": effective_top150_hits,
        "neural_hits": neural_hits,
        "strategy_hits": strategy_hits,
        "strategy_match_hits": strategy_match_hits,
        "strategy_dan_all3_hits": strategy_dan_all3_hits,
        "strategy_dan_position_hits": strategy_dan_position_hits,
        "recommended_hit_rate": recommended_hits / total if total else 0,
        "effective_hit_rate": effective_hits / total if total else 0,
        "effective_top150_hit_rate": effective_top150_hits / total if total else 0,
        "neural_hit_rate": neural_hits / total if total else 0,
        "strategy_hit_rate": strategy_hits / total if total else 0,
        "strategy_match_hit_rate": strategy_match_hits / total if total else 0,
        "strategy_dan_all3_hit_rate": strategy_dan_all3_hits / total if total else 0,
        "strategy_dan_position_hit_rate": strategy_dan_position_hits / (total * 3) if total else 0,
        "rank_buckets": rank_buckets,
        "top1_position_hits": pos_hits,
    }
    return rows, stats


def save_rolling_training(
    draws: Sequence[Draw],
    record_dir: str,
    budget: int,
    effective_limit: int,
    use_neural: bool = False,
) -> Dict:
    ensure_dir(record_dir)
    rows, stats = rolling_train_records(draws, budget, effective_limit, use_neural=use_neural)
    path = os.path.join(record_dir, "rolling_predictions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    json_dump(os.path.join(record_dir, "rolling_stats.json"), stats)
    rebuild_stats(record_dir)
    return stats


def query_prediction_rank(record_dir: str, issue: str, number: str) -> Dict:
    issue = normalize_issue(issue)
    number = normalize_number(number)
    pred = latest_prediction_for_issue(record_dir, issue)
    if pred:
        return {"source": "prediction_detail", **rank_number_in_prediction(pred, number)}
    for row in load_jsonl(os.path.join(record_dir, "rolling_predictions.jsonl")):
        if row.get("issue") == issue:
            if row.get("actual") == number:
                return {"source": "rolling_history", **row.get("rank_info", {})}
            return {
                "source": "rolling_history",
                "issue": issue,
                "number": number,
                "message": "历史滚动统计只保存该期实际开奖号的排名；请查询该期开奖号。",
                "actual": row.get("actual"),
            }
    raise ValueError(f"未找到期号 {issue} 的预测记录。")


def command_predict(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    if len(draws) < 5:
        raise ValueError("历史数据太少，至少建议 5 期以上；神经网络建议 35 期以上。")
    target_issue = args.issue or next_issue_after(draws[-1].issue)
    payload = prediction_payload(draws, target_issue, args.trial, args.budget, args.effective_limit)
    ensure_dir(args.record_dir)
    out_path = args.output or os.path.join(args.record_dir, f"prediction_{target_issue}.json")
    json_dump(out_path, payload)
    append_jsonl(os.path.join(args.record_dir, "predictions.jsonl"), payload)
    print_prediction_summary(payload)
    print(f"\n预测详情已保存: {out_path}")
    print(f"预测流水已追加: {os.path.join(args.record_dir, 'predictions.jsonl')}")


def command_train(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    if len(draws) < 5:
        raise ValueError("历史数据太少，至少建议 5 期以上；神经网络建议 35 期以上。")
    target_issue = args.issue or next_issue_after(draws[-1].issue)
    probs, models = ensemble_probs(draws, target_issue)
    payload = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data": args.data,
        "history_count": len(draws),
        "first_issue": draws[0].issue,
        "last_issue": draws[-1].issue,
        "target_issue_for_snapshot": target_issue,
        "models": models,
        "position_probabilities": sorted_position_probs(probs),
        "note": "predict 命令会基于最新历史数据重新训练；本文件用于保存训练快照与检查模型是否可用。",
    }
    json_dump(args.output, payload)
    print(f"训练样本: {len(draws)}，期号范围 {draws[0].issue} - {draws[-1].issue}")
    print(f"启用模型: {', '.join(models)}")
    print(f"训练快照已保存: {args.output}")


def command_settle(args: argparse.Namespace) -> None:
    summary = settle_prediction(args.record_dir, args.issue, args.actual)
    print(f"期号 {summary['issue']} 开奖 {summary['actual']}")
    print(f"20 注推荐命中全号: {'是' if summary['actual_in_recommended'] else '否'}")
    print(f"策略过滤后有效组合命中全号: {'是' if summary['actual_in_effective_combinations'] else '否'}")
    print(
        "有效组合分位命中次数: "
        f"百 {summary['counts']['hundred_hits']} / "
        f"十 {summary['counts']['ten_hits']} / "
        f"个 {summary['counts']['unit_hits']}"
    )
    print(f"记录已追加: {os.path.join(args.record_dir, 'settlements.jsonl')}")
    print(f"统计已更新: {os.path.join(args.record_dir, 'stats.json')}")


def command_add(args: argparse.Namespace) -> None:
    append_draw(args.data, args.issue, args.number, args.trial)
    print(f"已写入开奖数据: {args.issue},{normalize_number(args.number)}")
    if latest_prediction_for_issue(args.record_dir, args.issue):
        summary = settle_prediction(args.record_dir, args.issue, args.number)
        rank_info = summary.get("rank_info", {})
        print("已自动结算该期预测。")
        print(f"推荐排名: {rank_info.get('recommended_rank') or '未进入 20 注'}")
        print(f"有效组合排名: {rank_info.get('effective_rank') or '未进入有效组合/被过滤'}")


def command_backtest(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    result = backtest(draws, args.start, args.budget, args.effective_limit)
    if args.output:
        json_dump(args.output, result)
    print(f"回测期数: {result['tested_issues']}")
    print(f"{args.budget} 注推荐全号命中: {result['recommended_hits']}，命中率 {result['recommended_hit_rate']:.2%}")
    print(f"策略过滤有效组合全号命中: {result['effective_hits']}，命中率 {result['effective_hit_rate']:.2%}")
    if args.output:
        print(f"回测详情已保存: {args.output}")


def command_rolling_train(args: argparse.Namespace) -> None:
    draws = load_draws(args.data)
    stats = save_rolling_training(
        draws,
        args.record_dir,
        args.budget,
        args.effective_limit,
        use_neural=args.use_neural,
    )
    print(f"滚动预测总次数: {stats['total_predictions']}")
    print(f"{args.budget} 注推荐命中: {stats['recommended_hits']}，命中率 {stats['recommended_hit_rate']:.2%}")
    print(f"策略过滤有效组合命中: {stats['effective_hits']}，命中率 {stats['effective_hit_rate']:.2%}")
    print(f"策略过滤有效组合前 150 名命中: {stats['effective_top150_hits']}，命中率 {stats['effective_top150_hit_rate']:.2%}")
    print(f"记录已保存: {os.path.join(args.record_dir, 'rolling_predictions.jsonl')}")
    print(f"统计已保存: {os.path.join(args.record_dir, 'rolling_stats.json')}")


def command_rank(args: argparse.Namespace) -> None:
    info = query_prediction_rank(args.record_dir, args.issue, args.number)
    print(f"期号: {info.get('issue')} 号码: {info.get('number')}")
    if info.get("message"):
        print(info["message"])
        if info.get("actual"):
            print(f"该期实际开奖号: {info['actual']}")
        return
    print(f"推荐列表排名: {info.get('recommended_rank') or '未进入推荐'}")
    print(f"策略过滤有效组合排名: {info.get('effective_rank') or '未进入有效组合/被过滤'}")
    print(f"有效组合总数: {info.get('effective_combo_count')}")


def command_make_sample(args: argparse.Namespace) -> None:
    sample = [
        Draw("2026001", "527", "138"),
        Draw("2026002", "049", "276"),
        Draw("2026003", "718", "442"),
        Draw("2026004", "253", "871"),
        Draw("2026005", "479", "348"),
        Draw("2026006", "663", "275"),
        Draw("2026007", "107", "509"),
        Draw("2026008", "846", "624"),
        Draw("2026009", "371", "242"),
        Draw("2026010", "920", "805"),
    ]
    save_draws(args.output, sample)
    print(f"示例数据已生成: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D/排列三概率训练、预测、策略过滤和开奖结算工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="训练/拟合模型并保存概率快照")
    p.add_argument("--data", default="history.csv", help="历史数据 CSV，格式 issue,number[,trial]")
    p.add_argument("--issue", help="快照对应的目标期号，不填则自动取最后一期 +1")
    p.add_argument("--output", default="model_snapshot.json", help="训练快照输出路径")
    p.set_defaults(func=command_train)

    p = sub.add_parser("predict", help="基于历史数据预测下一期")
    p.add_argument("--data", default="history.csv", help="历史数据 CSV，格式 issue,number[,trial]")
    p.add_argument("--issue", help="要预测的 7 位期号，不填则自动取最后一期 +1")
    p.add_argument("--trial", help="当期试机号，三位数；不填则不启用试机号策略")
    p.add_argument("--budget", type=int, default=20, help="预算注数，默认 20")
    p.add_argument("--effective-limit", type=int, default=1000, help="保存策略过滤后有效组合数量，默认保存全部候选上限 1000")
    p.add_argument("--record-dir", default=DEFAULT_RECORD_DIR, help="预测与结算记录目录")
    p.add_argument("--output", help="预测 JSON 输出路径")
    p.set_defaults(func=command_predict)

    p = sub.add_parser("settle", help="录入实际开奖号并结算命中")
    p.add_argument("--issue", required=True, help="开奖期号")
    p.add_argument("--actual", required=True, help="实际开奖号")
    p.add_argument("--record-dir", default=DEFAULT_RECORD_DIR, help="预测与结算记录目录")
    p.set_defaults(func=command_settle)

    p = sub.add_parser("add", help="新增或更新一期开奖结果")
    p.add_argument("--data", default="history.csv", help="历史数据 CSV")
    p.add_argument("--issue", required=True, help="开奖期号")
    p.add_argument("--number", required=True, help="实际开奖号")
    p.add_argument("--trial", help="试机号")
    p.add_argument("--record-dir", default=DEFAULT_RECORD_DIR, help="预测与结算记录目录")
    p.set_defaults(func=command_add)

    p = sub.add_parser("backtest", help="历史回测")
    p.add_argument("--data", default="history.csv", help="历史数据 CSV")
    p.add_argument("--start", type=int, default=35, help="从第几条数据开始滚动回测")
    p.add_argument("--budget", type=int, default=20, help="每期预算注数")
    p.add_argument("--effective-limit", type=int, default=1000, help="策略过滤后有效组合数量，默认保存全部候选上限 1000")
    p.add_argument("--output", help="回测 JSON 输出路径")
    p.set_defaults(func=command_backtest)

    p = sub.add_parser("rolling-train", help="从第 2 期开始逐期滚动训练预测，并保存命中统计")
    p.add_argument("--data", default="history.csv", help="历史数据 CSV")
    p.add_argument("--budget", type=int, default=20, help="每期预算注数")
    p.add_argument("--effective-limit", type=int, default=1000, help="策略过滤后有效组合数量")
    p.add_argument("--record-dir", default=DEFAULT_RECORD_DIR, help="预测与结算记录目录")
    p.add_argument("--use-neural", action="store_true", help="历史每期回放也启用 MLP；全量历史会很慢")
    p.set_defaults(func=command_rolling_train)

    p = sub.add_parser("rank", help="查询某期开奖号码在预测数据里的排名")
    p.add_argument("--issue", required=True, help="期号")
    p.add_argument("--number", required=True, help="要查询的开奖号")
    p.add_argument("--record-dir", default=DEFAULT_RECORD_DIR, help="预测与结算记录目录")
    p.set_defaults(func=command_rank)

    p = sub.add_parser("make-sample", help="生成示例 history.csv")
    p.add_argument("--output", default="history.csv")
    p.set_defaults(func=command_make_sample)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
