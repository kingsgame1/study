# V6 CTA Plan -- 8-DC + RuleDB

## Goal
v5x baseline: Sharpe -0.77, Trades 414, hit_rate 36%, avg_win 1.56% vs fee 1.2% (round trip) = net -0.11%.
Bottleneck: avg_win/avg_loss ratio must >= 1.8x to break even at 36% hit_rate.

v6.0 targets: 8-DC DecisionContext + RuleDB hard-trigger rules from Nic's article.

---

## Baseline Parameters (v5x -- keep same for v6.0)
ADX_TH = 22  |  VSPIKE = 3.0  |  ATR_M = 4.0  |  TRAIL_A = 7.0  |  TSB = 288
Fee: 0.06% per leg, 0.12% round trip |  Circuit: DD < -20% halt

---

## v6.0 Tasks (ordered by dependency)

### T1: 8-DC DecisionContext -- build_ctx()

Implement build_ctx(t, state, feats, ctx_hparams) producing 8 keys.

01  | trigger_reason  | str  | signal_side + primary_reason from feats/state
02  | indicators      | dict | full feats at entry: rsi, macd, macd_hist, adx, bb_pos, momz, smf, vspike
03  | account_state   | dict | equity, dd_30d, per_coin_hits (BTC only for v6)
04  | macro_env       | dict | btc_liq_side (longs/shorts ratio from funding or placeholder)
05  | exp_top5        | list | result from self.replay(feats), top 5 similar decisions
06  | matched_rules   | list | RuleDB.match(feats, state), each a dict
07  | gate_status     | dict | adx_pass, rsi_pass, smf_pass, circuit OK -- REF ONLY
08  | position_status | dict | current_side, pct_k, bars_held, unrealized_pct

All 8 keys are pure functions, no LLM needed -- deterministic from existing v5 code + RuleDB.

### T2: RuleDB Hard-Trigger Rules

Schema: avoidance | use_condition | hard_trigger

- avoid_open / size_down / require_strong_signal
- scope: "BTC" or coin pattern
- trigger_conditions: { market_regime / rsi / ls_ratio / macd_hist }
- action + rationale

Export format (v5-review already qualifies as training data for this):
```json
{
  "id": "rule_001",
  "type": "avoidance",
  "scope": "BTC",
  "trigger": { "market_regime": "RANGE", "rsi_range": [40, 60], "ls_ratio_min": 2.0 },
  "action": "avoid_open",
  "rationale": "震荡市 + 高多空比 + RSI中段 = 反趋势高损"
}
```

DB operations: load / save / match(ctx) -> [rules] / add_from_review(entry, outcome)

### T3: Unified Decision Path (single route)

Current multi-entry design:
- signal path: signal() -> long/short function call
- review path: batch_review() runs post-exit

Unified path: every bar when decision should be made:
Trigger (signals) -> build_ctx() -> matched_rules -> final_decision(action, confidence) -> execute/carry

One route. One entry per bar. Configurable via scheduler_tick() timeline.

### T4: Gate De-escalation -- informational only

Replace hard-circuit with informational feed:
- gate_status still calculated
- instead of: if circuit_fuse: coeff = 0.25 -> block

After: gate_status shown in DC [7], agent/LLM weighs it; circuit STOP stays as final emergency button.

### T5: Conviction --> Position Sizing (non-linear)

Confidence (0-100) -> non-linear position:
  coef = 0.5 + 0.5 * sqrt(c / 100)

Replace current linear if conv > 60 else if conv < 40 : 0.25 rules.
Target: avg_win improvement > 0.3% across 30k.

---

## v6.0 Acceptance Criteria

- T1: build_ctx() returns 8 keys, all deterministic, can serialize full DecisionContext per bar (sample: first 3 bars of test set)
- T2: RuleDB has >= 10 rules loaded from v5 ExpDB, match() returns filtered list, review_embed exports structured rules
- T3: Single route exercised across all 30k bars, only one entry function called per evaluation bar
- T4: circuit_breaker gate only read, never coerces action directly in signal path
- T5: confidence non-linear sizing observed in signal logs (confidence values logged to DC [8])
- Expected v6.0 result: Sharpe improvement >= +0.15 (conservative), start point from -0.77

---

## Not in v6.0 (v6.1+)

- Z-adjust experiment cloning: just use v5 log for offline evaluation
- Dashboard (panel): markdown summary → v6.2
- Synthesis / risk-state reconciliation: v6.3
- Floating map termination cost: v6.4

---

## File Tree (v6.0)

cta_v6/

core/

- DC ... DC Context 真实 做 bets 看 provide.trade, ... Convasions tx_ref? ≈ will( lens ...
- DC.py         # build_ctx(), 8_ctx (T1)
- RuleDB    # RuleDB (T2)
- Decision     ...


- circuit

signal/

- gate    # Gate tooling helper ()
- risk_c()     # risk_c(), pct_size-> circuit-breaker

backtest/

- backtest_v6.py    # Unified path (T3+T4+T5)

results/

-
