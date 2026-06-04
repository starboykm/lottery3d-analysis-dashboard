#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地可视化看板：历史数据查询、录入、神经网络预测、开奖结算。
"""

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from lottery3d import (
    DEFAULT_RECORD_DIR,
    Draw,
    append_draw,
    append_jsonl,
    backtest,
    ensure_dir,
    json_dump,
    latest_prediction_for_issue,
    load_draws,
    load_jsonl,
    next_issue_after,
    normalize_issue,
    normalize_number,
    prediction_payload,
    query_prediction_rank,
    rebuild_stats,
    save_draws,
    save_rolling_training,
    settle_prediction,
)


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "history.csv")
RECORD_DIR = os.path.join(APP_DIR, DEFAULT_RECORD_DIR)
LOG_PATH = os.path.join(APP_DIR, "dashboard.log")


def write_log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    try:
        if sys.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except Exception:
        pass


def response_summary(payload):
    return {
        "created_at": payload.get("created_at"),
        "issue": payload.get("issue"),
        "trial": payload.get("trial"),
        "history_count": payload.get("history_count"),
        "models": payload.get("models", []),
        "position_probabilities": payload.get("position_probabilities", {}),
        "position_selection_plans": payload.get("position_selection_plans", []),
        "budget": payload.get("budget"),
        "recommended_numbers": payload.get("recommended_numbers", []),
        "neural_prediction_numbers": payload.get("neural_prediction_numbers", [])[:300],
        "markov_prediction_numbers": payload.get("markov_prediction_numbers", [])[:300],
        "markov_position_probabilities": payload.get("markov_position_probabilities", {}),
        "markov_top5_digits": payload.get("markov_top5_digits", []),
        "strategy_prediction_numbers": payload.get("strategy_prediction_numbers", [])[:300],
        "strategy_neural_matches": payload.get("strategy_neural_matches", [])[:300],
        "strategy_dan_digits": payload.get("strategy_dan_digits", []),
        "strategy_dan_candidates": payload.get("strategy_dan_candidates", []),
        "strategy": payload.get("strategy", {}),
        "effective_combo_count": len(payload.get("strategy_filtered_effective_combinations", [])),
        "strategy_combo_count": len(payload.get("strategy_prediction_numbers", [])),
        "neural_combo_count": len(payload.get("neural_prediction_numbers", [])),
        "markov_combo_count": len(payload.get("markov_prediction_numbers", [])),
        "strategy_filtered_effective_combinations": payload.get("strategy_filtered_effective_combinations", [])[:120],
    }


def parse_history_text(text):
    rows = []
    reader = csv.reader(io.StringIO(text.strip()))
    for raw in reader:
        if not raw or all(not x.strip() for x in raw):
            continue
        if raw[0].strip().lower() in ("issue", "期号"):
            continue
        if len(raw) < 2:
            raise ValueError(f"无法解析这一行: {','.join(raw)}")
        trial = raw[2].strip() if len(raw) > 2 and raw[2].strip() else None
        rows.append(Draw(normalize_issue(raw[0]), normalize_number(raw[1]), normalize_number(trial) if trial else None))
    return rows


def read_stats():
    path = os.path.join(RECORD_DIR, "stats.json")
    if not os.path.exists(path):
        return rebuild_stats(RECORD_DIR)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_predictions(limit=20):
    path = os.path.join(RECORD_DIR, "predictions.jsonl")
    rows = load_jsonl(path)
    return [
        {
            "created_at": row.get("created_at"),
            "issue": row.get("issue"),
            "trial": row.get("trial"),
            "models": row.get("models", []),
            "budget": row.get("budget"),
            "recommended_numbers": [x.get("number") for x in row.get("recommended_numbers", [])[:20]],
            "effective_combo_count": len(row.get("strategy_filtered_effective_combinations", [])),
        }
        for row in rows[-limit:][::-1]
    ]


def latest_settlements(limit=20):
    rows = load_jsonl(os.path.join(RECORD_DIR, "settlements.jsonl"))
    return [
        {
            "settled_at": row.get("settled_at"),
            "issue": row.get("issue"),
            "actual": row.get("actual"),
            "actual_in_recommended": row.get("actual_in_recommended"),
            "actual_in_effective_combinations": row.get("actual_in_effective_combinations"),
            "effective_combo_count": row.get("effective_combo_count"),
            "counts": row.get("counts", {}),
        }
        for row in rows[-limit:][::-1]
    ]


def latest_prediction_rank_rows(limit=100):
    combined = {}
    for row in load_jsonl(os.path.join(RECORD_DIR, "rolling_predictions.jsonl")):
        rank_info = row.get("rank_info") or {}
        combined[row.get("issue")] = {
            "issue": row.get("issue"),
            "actual": row.get("actual"),
            "position_probability_ranks": row.get("position_probability_ranks", {}),
            "effective_combo_count": rank_info.get("effective_combo_count"),
            "recommended_rank": rank_info.get("recommended_rank"),
            "effective_rank": rank_info.get("effective_rank"),
            "is_recommended_hit": rank_info.get("is_recommended_hit", False),
            "is_effective_hit": rank_info.get("is_effective_hit", False),
            "neural_rank": row.get("neural_rank"),
            "markov_rank": row.get("markov_rank"),
            "strategy_rank": row.get("strategy_rank"),
            "strategy_dan_hit": row.get("strategy_dan_hit", {}),
            "source": "历史滚动",
        }
    for row in load_jsonl(os.path.join(RECORD_DIR, "settlements.jsonl")):
        rank_info = row.get("rank_info") or {}
        combined[row.get("issue")] = {
            "issue": row.get("issue"),
            "actual": row.get("actual"),
            "position_probability_ranks": row.get("position_probability_ranks", {}),
            "effective_combo_count": row.get("effective_combo_count") or rank_info.get("effective_combo_count"),
            "recommended_rank": rank_info.get("recommended_rank"),
            "effective_rank": rank_info.get("effective_rank"),
            "is_recommended_hit": rank_info.get("is_recommended_hit", False),
            "is_effective_hit": rank_info.get("is_effective_hit", False),
            "neural_rank": row.get("neural_rank"),
            "markov_rank": row.get("markov_rank"),
            "strategy_rank": row.get("strategy_rank"),
            "strategy_dan_hit": row.get("strategy_dan_hit", {}),
            "source": "新增结算",
        }
    rows = [r for r in combined.values() if r.get("issue")]
    rows.sort(key=lambda x: x["issue"], reverse=True)
    return rows[:limit]


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3D / 排列三概率看板</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d232d;
      --muted: #69717f;
      --line: #dfe3ea;
      --accent: #1467c8;
      --accent-2: #0f8b6e;
      --warn: #a05a00;
      --bad: #b42318;
      --shadow: 0 1px 2px rgba(15, 23, 42, .07);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      background: #172033;
      color: #fff;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    .sub { color: #b9c3d2; font-size: 13px; margin-top: 4px; }
    main { padding: 18px; max-width: 1480px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: 1.1fr .9fr; gap: 16px; align-items: start; }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
    .metric, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .metric { padding: 14px; min-height: 78px; }
    .metric b { display: block; font-size: 24px; margin-top: 8px; }
    .metric span { color: var(--muted); font-size: 13px; }
    section { padding: 16px; margin-bottom: 16px; }
    h2 { font-size: 16px; margin: 0 0 14px; }
    h3 { font-size: 14px; margin: 16px 0 10px; }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }
    label { display: grid; gap: 6px; font-size: 13px; color: var(--muted); }
    input, textarea, select {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 9px 10px;
      min-height: 38px;
      font: inherit;
    }
    input { width: 142px; }
    textarea { width: 100%; min-height: 120px; resize: vertical; }
    button {
      border: 0;
      background: var(--accent);
      color: #fff;
      border-radius: 6px;
      padding: 10px 14px;
      min-height: 38px;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary { background: #465468; }
    button.good { background: var(--accent-2); }
    button.warn { background: var(--warn); }
    button:disabled { opacity: .55; cursor: wait; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; background: #fafbfc; }
    .scroll { overflow: auto; max-height: 360px; border: 1px solid var(--line); border-radius: 6px; }
    .pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 34px;
      padding: 4px 7px;
      border-radius: 999px;
      background: #eef4ff;
      color: #174a88;
      margin: 2px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .pill.hit { background: #e8f7f2; color: #08745c; }
    .pill.bad { background: #fff1ef; color: var(--bad); }
    .bar-row {
      display: grid;
      grid-template-columns: 42px 1fr 58px;
      gap: 8px;
      align-items: center;
      margin: 7px 0;
      font-size: 13px;
    }
    .bar {
      position: relative;
      height: 14px;
      background: #edf0f4;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar > i { display: block; height: 100%; background: linear-gradient(90deg, #1467c8, #0f8b6e); border-radius: 999px; }
    .muted { color: var(--muted); }
    .status { min-height: 22px; font-size: 13px; color: var(--muted); margin-top: 10px; }
    .status.err { color: var(--bad); }
    .nums { line-height: 1.8; }
    .split { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .plans { display: grid; gap: 8px; }
    .plan {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      background: #fcfdff;
      font-size: 13px;
    }
    .mono { font-family: Consolas, "SFMono-Regular", monospace; }
    @media (max-width: 980px) {
      .grid, .split, .metrics { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>3D / 排列三概率看板</h1>
      <div class="sub">历史数据、神经网络预测、策略过滤、开奖结算</div>
    </div>
    <button class="secondary" onclick="refreshAll()">刷新</button>
  </header>

  <main>
    <div class="metrics">
      <div class="metric"><span>历史样本</span><b id="mHistory">-</b></div>
      <div class="metric"><span>最后期号</span><b id="mLast">-</b></div>
      <div class="metric"><span>总预测次数</span><b id="mPredictionTotal">-</b></div>
      <div class="metric"><span>20 注推荐命中</span><b id="mRecHit">-</b></div>
      <div class="metric"><span>有效组合命中</span><b id="mEffHit">-</b></div>
      <div class="metric"><span>有效组合前 150 命中</span><b id="mTop150Hit">-</b></div>
    </div>

    <div class="grid">
      <div>
        <section>
          <h2>预测下一期</h2>
          <div class="row">
            <label>预测期号<input id="predictIssue" placeholder="自动下一期" maxlength="7"></label>
            <label>试机号<input id="predictTrial" placeholder="可选" maxlength="3"></label>
            <label>预算注数<input id="budget" value="20" type="number" min="1" max="100"></label>
            <button class="good" onclick="predict()">开始预测</button>
          </div>
          <div id="predictStatus" class="status"></div>
          <h3>每位数字概率</h3>
          <div class="split" id="probPanels"></div>
          <h3>预算内位置选号方案</h3>
          <div class="plans" id="plans"></div>
          <h3>20 注推荐</h3>
          <div class="nums mono" id="recommended"></div>
          <h3>神经网络预测号</h3>
          <div class="muted" id="neuralCount"></div>
          <div class="nums mono" id="neuralPreview"></div>
          <h3>马尔可夫预测号</h3>
          <div class="muted" id="markovCount"></div>
          <div class="nums mono" id="markovPreview"></div>
          <h3>马尔可夫前5胆码（不对位）</h3>
          <div class="nums mono" id="markovTop5Digits"></div>
          <h3>纯策略预测号</h3>
          <div class="muted" id="strategyOnlyCount"></div>
          <div class="nums mono" id="strategyOnlyPreview"></div>
          <h3>策略预测胆码</h3>
          <div class="nums mono" id="strategyDanPreview"></div>
          <h3>策略与神经网络相同预测号</h3>
          <div class="muted" id="matchCount"></div>
          <div class="scroll" style="margin-top:8px; max-height:260px"><table id="matchTable"></table></div>
          <h3>策略过滤后有效组合预览</h3>
          <div class="muted" id="effectiveCount"></div>
          <div class="nums mono" id="effectivePreview"></div>
        </section>

        <section>
          <h2>开奖结算</h2>
          <div class="row">
            <label>期号<input id="settleIssue" maxlength="7"></label>
            <label>开奖号<input id="settleActual" maxlength="3"></label>
            <button class="warn" onclick="settle()">结算命中</button>
          </div>
          <div id="settleStatus" class="status"></div>
          <div id="settleResult"></div>
        </section>

        <section>
          <h2>开奖号码排名查询</h2>
          <div class="row">
            <label>期号<input id="rankIssue" maxlength="7"></label>
            <label>开奖号<input id="rankNumber" maxlength="3"></label>
            <button class="secondary" onclick="queryRank()">查询排名</button>
          </div>
          <div id="rankStatus" class="status"></div>
          <div id="rankResult"></div>
        </section>
      </div>

      <div>
        <section>
          <h2>录入新开奖数据</h2>
          <div class="row">
            <label>期号<input id="addIssue" maxlength="7"></label>
            <label>开奖号<input id="addNumber" maxlength="3"></label>
            <label>试机号<input id="addTrial" placeholder="可选" maxlength="3"></label>
            <button onclick="addDraw()">保存</button>
          </div>
          <div id="addStatus" class="status"></div>
        </section>

        <section>
          <h2>批量导入历史数据</h2>
          <textarea id="bulkText" placeholder="每行格式：2026001,527,138&#10;第三列试机号可省略"></textarea>
          <div class="row">
            <label>导入方式
              <select id="importMode">
                <option value="append">追加/覆盖同期期号</option>
                <option value="replace">替换全部历史数据</option>
              </select>
            </label>
            <button onclick="importHistory()">导入</button>
          </div>
          <div id="bulkStatus" class="status"></div>
        </section>

        <section>
          <h2>历史数据查询</h2>
          <div class="row">
            <label>期号包含<input id="historyQuery" placeholder="例如 2026"></label>
            <button class="secondary" onclick="loadHistory()">查询</button>
          </div>
          <div class="scroll" style="margin-top:12px"><table id="historyTable"></table></div>
        </section>

        <section>
          <h2>历史滚动训练统计</h2>
          <div class="row">
            <label>预算注数<input id="rollingBudget" value="20" type="number" min="1" max="100"></label>
            <button class="good" onclick="rollingTrain()">从第 2 期开始训练统计</button>
          </div>
          <div id="rollingStatus" class="status">全量历史较多，首次统计可能需要一点时间。</div>
        </section>
      </div>
    </div>

    <section>
      <h2>近 100 期预测排名明细</h2>
      <div class="scroll"><table id="rollingTable"></table></div>
    </section>

    <section>
      <h2>预测与结算记录</h2>
      <div class="split">
        <div>
          <h3>最近预测</h3>
          <div class="scroll"><table id="predictionTable"></table></div>
        </div>
        <div>
          <h3>最近结算</h3>
          <div class="scroll"><table id="settlementTable"></table></div>
        </div>
        <div>
          <h3>统计</h3>
          <pre id="statsBox" class="mono"></pre>
        </div>
      </div>
    </section>
  </main>

<script>
const posNames = {hundred: "百位", ten: "十位", unit: "个位"};

function pct(x) { return ((x || 0) * 100).toFixed(2) + "%"; }
function byId(id) { return document.getElementById(id); }
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...opts
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || "请求失败");
  return data;
}
function setStatus(id, msg, err=false) {
  const el = byId(id);
  el.textContent = msg || "";
  el.className = "status" + (err ? " err" : "");
}
function renderTable(id, headers, rows) {
  const table = byId(id);
  table.innerHTML = "<thead><tr>" + headers.map(h => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
    rows.join("") + "</tbody>";
}

async function loadHistory() {
  const q = encodeURIComponent(byId("historyQuery").value.trim());
  const data = await api(`/api/history?query=${q}`);
  byId("mHistory").textContent = data.count;
  byId("mLast").textContent = data.last_issue || "-";
  if (!byId("predictIssue").value && data.next_issue) byId("predictIssue").placeholder = data.next_issue;
  renderTable("historyTable", ["期号", "开奖号", "试机号", "和值", "跨度"], data.rows.map(r =>
    `<tr><td>${r.issue}</td><td class="mono">${r.number}</td><td class="mono">${r.trial || ""}</td><td>${r.sum}</td><td>${r.span}</td></tr>`
  ));
}

function renderPrediction(data) {
  byId("probPanels").innerHTML = Object.keys(posNames).map(pos => {
    const bars = data.position_probabilities[pos].map(x => `
      <div class="bar-row">
        <b>${x.digit}</b>
        <span class="bar"><i style="width:${Math.max(2, x.probability * 100)}%"></i></span>
        <span>${pct(x.probability)}</span>
      </div>`).join("");
    return `<div><h3>${posNames[pos]}</h3>${bars}</div>`;
  }).join("");
  byId("plans").innerHTML = data.position_selection_plans.slice(0, 6).map(p => {
    const d = p.digits;
    return `<div class="plan"><b>${p.counts.hundred}x${p.counts.ten}x${p.counts.unit}=${p.notes} 注</b>
      <span class="muted">估计覆盖 ${pct(p.estimated_cover_probability)}</span><br>
      百 <span class="mono">${d.hundred.join("")}</span>
      十 <span class="mono">${d.ten.join("")}</span>
      个 <span class="mono">${d.unit.join("")}</span></div>`;
  }).join("");
  byId("recommended").innerHTML = data.recommended_numbers.map(x => `<span class="pill">${x.number}</span>`).join("");
  byId("neuralCount").textContent = `神经网络排序数量：${data.neural_combo_count || data.neural_prediction_numbers.length}，下方显示前 ${data.neural_prediction_numbers.length} 个。`;
  byId("neuralPreview").innerHTML = data.neural_prediction_numbers.slice(0, 120).map((x, i) => `<span class="pill" title="神经网络排名 ${i + 1}">${x.number}</span>`).join("");
  byId("markovCount").textContent = `马尔可夫排序数量：${data.markov_combo_count || data.markov_prediction_numbers.length}，下方显示前 ${data.markov_prediction_numbers.length} 个。`;
  byId("markovPreview").innerHTML = data.markov_prediction_numbers.slice(0, 120).map((x, i) => `<span class="pill" title="马尔可夫排名 ${i + 1}">${x.number}</span>`).join("");
  byId("markovTop5Digits").innerHTML = (data.markov_top5_digits || []).map((x, i) => `<span class="pill" title="不对位排名 ${i + 1}，出现概率 ${pct(x.probability || 0)}">${x.digit}</span>`).join("");
  byId("strategyOnlyCount").textContent = `纯策略排序数量：${data.strategy_combo_count || data.strategy_prediction_numbers.length}，下方显示前 ${data.strategy_prediction_numbers.length} 个。`;
  byId("strategyOnlyPreview").innerHTML = data.strategy_prediction_numbers.slice(0, 120).map((x, i) => `<span class="pill" title="策略排名 ${i + 1}">${x.number}</span>`).join("");
  byId("strategyDanPreview").innerHTML = (data.strategy_dan_digits || []).map(x => `<span class="pill" title="胆码权重 ${Number(x.weight || 0).toFixed(2)}">${x.digit}</span>`).join("");
  byId("matchCount").textContent = `相同预测号数量：${data.strategy_neural_matches.length}。`;
  renderTable("matchTable", ["号码", "策略排名", "神经网络排名", "类型", "和值", "跨度", "策略依据"], data.strategy_neural_matches.map(x =>
    `<tr><td class="mono">${x.number}</td><td>${x.strategy_rank}</td><td>${x.neural_rank}</td><td>${x.type || ""}</td><td>${x.sum}</td><td>${x.span}</td><td>${(x.strategy_reasons || []).map(esc).join("；")}</td></tr>`
  ));
  byId("effectiveCount").textContent = `完整有效组合数量：${data.effective_combo_count}，下方显示前 ${data.strategy_filtered_effective_combinations.length} 个。`;
  byId("effectivePreview").innerHTML = data.strategy_filtered_effective_combinations.map(x => `<span class="pill">${x.number}</span>`).join("");
  byId("settleIssue").value = data.issue;
}

async function predict() {
  setStatus("predictStatus", "正在训练模型并预测...");
  try {
    const data = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        issue: byId("predictIssue").value.trim(),
        trial: byId("predictTrial").value.trim(),
        budget: Number(byId("budget").value || 20)
      })
    });
    renderPrediction(data);
    setStatus("predictStatus", `预测完成。启用模型：${data.models.join("、")}`);
    await loadRecords();
  } catch (e) {
    setStatus("predictStatus", e.message, true);
  }
}

async function addDraw() {
  setStatus("addStatus", "正在保存...");
  try {
    const data = await api("/api/add_draw", {
      method: "POST",
      body: JSON.stringify({
        issue: byId("addIssue").value.trim(),
        number: byId("addNumber").value.trim(),
        trial: byId("addTrial").value.trim()
      })
    });
    const extra = data.settlement
      ? `；已自动结算，推荐排名 ${data.settlement.rank_info.recommended_rank || "未进 20 注"}，有效组合排名 ${data.settlement.rank_info.effective_rank || "未进入/被过滤"}`
      : "";
    setStatus("addStatus", `已保存 ${data.issue},${data.number}${extra}`);
    await refreshAll();
  } catch (e) {
    setStatus("addStatus", e.message, true);
  }
}

async function importHistory() {
  setStatus("bulkStatus", "正在导入...");
  try {
    const data = await api("/api/import_history", {
      method: "POST",
      body: JSON.stringify({text: byId("bulkText").value, mode: byId("importMode").value})
    });
    setStatus("bulkStatus", `导入完成，共 ${data.count} 条历史数据。`);
    await refreshAll();
  } catch (e) {
    setStatus("bulkStatus", e.message, true);
  }
}

async function settle() {
  setStatus("settleStatus", "正在结算...");
  try {
    const data = await api("/api/settle", {
      method: "POST",
      body: JSON.stringify({issue: byId("settleIssue").value.trim(), actual: byId("settleActual").value.trim()})
    });
    byId("settleResult").innerHTML = `
      <p>20 注推荐全号命中：<span class="pill ${data.actual_in_recommended ? "hit" : "bad"}">${data.actual_in_recommended ? "是" : "否"}</span>
      策略有效组合全号命中：<span class="pill ${data.actual_in_effective_combinations ? "hit" : "bad"}">${data.actual_in_effective_combinations ? "是" : "否"}</span>
      纯策略预测命中：<span class="pill ${data.actual_in_strategy_prediction ? "hit" : "bad"}">${data.actual_in_strategy_prediction ? "是" : "否"}</span></p>
      <p>推荐排名：${data.rank_info.recommended_rank || "未进入 20 注"}；
      有效组合排名：${data.rank_info.effective_rank || "未进入有效组合/被过滤"}；
      纯策略排名：${data.strategy_rank || "未进入纯策略预测"}；
      神经网络排名：${data.neural_rank || "未进入神经网络排序"}；
      马尔可夫排名：${data.markov_rank || "未进入马尔可夫排序"}</p>
      <p>策略胆码命中：${(data.strategy_dan_hit.hit_digits || []).join("、") || "无"}；
      命中 ${data.strategy_dan_hit.hit_count}/3，命中率 ${pct(data.strategy_dan_hit.hit_rate)}</p>
      <p>马尔可夫前5胆码命中：${((data.markov_top5_digit_hit || {}).hit_digits || []).join("、") || "无"}；
      命中 ${(data.markov_top5_digit_hit || {}).hit_count || 0}/3，命中率 ${pct((data.markov_top5_digit_hit || {}).hit_rate || 0)}</p>
      <p>分位命中次数：百 ${data.counts.hundred_hits} / 十 ${data.counts.ten_hits} / 个 ${data.counts.unit_hits}</p>`;
    setStatus("settleStatus", "结算完成，统计已更新。");
    await loadRecords();
  } catch (e) {
    setStatus("settleStatus", e.message, true);
  }
}

async function queryRank() {
  setStatus("rankStatus", "正在查询...");
  try {
    const data = await api("/api/rank", {
      method: "POST",
      body: JSON.stringify({issue: byId("rankIssue").value.trim(), number: byId("rankNumber").value.trim()})
    });
    if (data.message) {
      byId("rankResult").innerHTML = `<p>${esc(data.message)}</p><p>该期实际开奖号：<span class="pill">${data.actual || ""}</span></p>`;
    } else {
      byId("rankResult").innerHTML = `
        <p>推荐列表排名：<span class="pill ${data.recommended_rank ? "hit" : "bad"}">${data.recommended_rank || "未进入 20 注"}</span></p>
        <p>策略过滤有效组合排名：<span class="pill ${data.effective_rank ? "hit" : "bad"}">${data.effective_rank || "未进入/被过滤"}</span></p>
        <p class="muted">有效组合总数：${data.effective_combo_count || 0}，来源：${esc(data.source)}</p>`;
    }
    setStatus("rankStatus", "查询完成。");
  } catch (e) {
    setStatus("rankStatus", e.message, true);
  }
}

async function rollingTrain() {
  setStatus("rollingStatus", "正在从第 2 期开始逐期滚动训练统计，请稍等...");
  try {
    const data = await api("/api/rolling_train", {
      method: "POST",
      body: JSON.stringify({budget: Number(byId("rollingBudget").value || 20)})
    });
    setStatus("rollingStatus", `完成：总预测 ${data.total_predictions} 次，20 注命中 ${data.recommended_hits} 次，有效组合命中 ${data.effective_hits} 次。`);
    await loadRecords();
  } catch (e) {
    setStatus("rollingStatus", e.message, true);
  }
}

async function loadRecords() {
  const data = await api("/api/records");
  byId("mPredictionTotal").textContent = data.stats.all_prediction_total || data.stats.historical_prediction_total || data.stats.settlement_count || 0;
  byId("mRecHit").textContent = data.stats.all_recommended_hits || data.stats.historical_recommended_hits || data.stats.recommended_full_hits || 0;
  byId("mEffHit").textContent = data.stats.all_effective_hits || data.stats.historical_effective_hits || data.stats.effective_full_hits || 0;
  byId("mTop150Hit").textContent = data.stats.all_effective_top150_hits || data.stats.historical_effective_top150_hits || data.stats.effective_top150_hits || 0;
  renderTable("rollingTable", ["期号", "开奖号", "百位概率排名", "十位概率排名", "个位概率排名", "有效组合总数", "推荐排名", "神经网络排名", "马尔可夫排名", "纯策略排名", "策略胆码命中", "有效组合命中", "有效组合排名"], data.rolling_recent.map(r => {
    const pr = r.position_probability_ranks || {};
    const dan = r.strategy_dan_hit || {};
    return `<tr>
      <td>${r.issue}</td>
      <td class="mono">${r.actual}</td>
      <td>${pr.hundred || ""}</td>
      <td>${pr.ten || ""}</td>
      <td>${pr.unit || ""}</td>
      <td>${r.effective_combo_count || 0}</td>
      <td>${r.recommended_rank || "未进 20 注"}</td>
      <td>${r.neural_rank || "未进入"}</td>
      <td>${r.markov_rank || "未进入"}</td>
      <td>${r.strategy_rank || "未进入"}</td>
      <td>${(dan.hit_digits || []).join("") || "无"} (${dan.hit_count || 0}/3)</td>
      <td>${r.is_effective_hit ? "是" : "否"}</td>
      <td>${r.effective_rank || "未进入/被过滤"}</td>
    </tr>`;
  }));
  renderTable("predictionTable", ["时间", "期号", "试机号", "模型", "推荐"], data.predictions.map(r =>
    `<tr><td>${esc(r.created_at)}</td><td>${r.issue}</td><td>${r.trial || ""}</td><td>${r.models.length}</td><td class="mono">${r.recommended_numbers.join(" ")}</td></tr>`
  ));
  renderTable("settlementTable", ["时间", "期号", "开奖号", "推荐", "有效"], data.settlements.map(r =>
    `<tr><td>${esc(r.settled_at)}</td><td>${r.issue}</td><td class="mono">${r.actual}</td><td>${r.actual_in_recommended ? "中" : "未中"}</td><td>${r.actual_in_effective_combinations ? "中" : "未中"}</td></tr>`
  ));
  byId("statsBox").textContent = JSON.stringify(data.stats, null, 2);
}

async function refreshAll() {
  try {
    await loadHistory();
    await loadRecords();
  } catch (e) {
    console.error(e);
  }
}
refreshAll();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "LotteryDashboard/1.0"

    def log_message(self, fmt, *args):
        write_log(fmt % args)

    def send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                raw = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if parsed.path == "/api/history":
                params = parse_qs(parsed.query)
                query = (params.get("query", [""])[0] or "").strip()
                draws = load_draws(DATA_PATH) if os.path.exists(DATA_PATH) else []
                filtered = [d for d in draws if query in d.issue or query in d.number] if query else draws
                rows = [
                    {
                        "issue": d.issue,
                        "number": d.number,
                        "trial": d.trial,
                        "sum": sum(map(int, d.number)),
                        "span": max(map(int, d.number)) - min(map(int, d.number)),
                    }
                    for d in filtered[-500:][::-1]
                ]
                self.send_json({
                    "count": len(draws),
                    "shown": len(rows),
                    "last_issue": draws[-1].issue if draws else None,
                    "next_issue": next_issue_after(draws[-1].issue) if draws else None,
                    "rows": rows,
                })
                return
            if parsed.path == "/api/records":
                ensure_dir(RECORD_DIR)
                self.send_json({
                    "stats": read_stats(),
                    "predictions": latest_predictions(),
                    "settlements": latest_settlements(),
                    "rolling_recent": latest_prediction_rank_rows(),
                })
                return
            self.send_json({"error": "Not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self.read_json()
            ensure_dir(RECORD_DIR)
            if parsed.path == "/api/add_draw":
                issue = body.get("issue", "")
                number = body.get("number", "")
                trial = body.get("trial") or None
                append_draw(DATA_PATH, issue, number, trial)
                response = {"ok": True, "issue": normalize_issue(issue), "number": normalize_number(number)}
                if latest_prediction_for_issue(RECORD_DIR, issue):
                    response["settlement"] = settle_prediction(RECORD_DIR, issue, number)
                self.send_json(response)
                return
            if parsed.path == "/api/import_history":
                rows = parse_history_text(body.get("text", ""))
                if not rows:
                    raise ValueError("没有可导入的数据")
                mode = body.get("mode", "append")
                if mode == "replace":
                    save_draws(DATA_PATH, rows)
                    all_rows = load_draws(DATA_PATH)
                else:
                    existing = {d.issue: d for d in (load_draws(DATA_PATH) if os.path.exists(DATA_PATH) else [])}
                    for row in rows:
                        existing[row.issue] = row
                    all_rows = sorted(existing.values(), key=lambda x: x.issue)
                    save_draws(DATA_PATH, all_rows)
                self.send_json({"ok": True, "count": len(all_rows)})
                return
            if parsed.path == "/api/predict":
                draws = load_draws(DATA_PATH)
                issue = body.get("issue") or next_issue_after(draws[-1].issue)
                trial = body.get("trial") or None
                budget = int(body.get("budget") or 20)
                payload = prediction_payload(draws, issue, trial, budget, 1000)
                out_path = os.path.join(RECORD_DIR, f"prediction_{payload['issue']}.json")
                json_dump(out_path, payload)
                append_jsonl(os.path.join(RECORD_DIR, "predictions.jsonl"), payload)
                self.send_json(response_summary(payload))
                return
            if parsed.path == "/api/settle":
                summary = settle_prediction(RECORD_DIR, body.get("issue", ""), body.get("actual", ""))
                self.send_json(summary)
                return
            if parsed.path == "/api/rank":
                info = query_prediction_rank(RECORD_DIR, body.get("issue", ""), body.get("number", ""))
                self.send_json(info)
                return
            if parsed.path == "/api/rolling_train":
                draws = load_draws(DATA_PATH)
                budget = int(body.get("budget") or 20)
                stats = save_rolling_training(draws, RECORD_DIR, budget, 1000, use_neural=False)
                self.send_json(stats)
                return
            if parsed.path == "/api/backtest":
                draws = load_draws(DATA_PATH)
                result = backtest(draws, int(body.get("start") or 35), int(body.get("budget") or 20), 1000)
                json_dump(os.path.join(APP_DIR, "dashboard_backtest.json"), result)
                self.send_json(result)
                return
            self.send_json({"error": "Not found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="启动 3D / 排列三本地看板")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.chdir(APP_DIR)
    ensure_dir(RECORD_DIR)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    write_log(f"看板已启动: http://{args.host}:{args.port}")
    write_log("按 Ctrl+C 停止。")
    server.serve_forever()


if __name__ == "__main__":
    main()
