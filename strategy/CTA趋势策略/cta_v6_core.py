"""
v6 Core: DecisionContext (8-DC) + RuleDB (hard-trigger rules)
Standalone module -- importable from any backtest runner.
"""
import json, numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

EXP_DB_PATH  = Path("/home/ubuntu/CTA_expdb.json")
RULE_DB_PATH = Path("/home/ubuntu/CTA_ruledb.json")


# ─── helpers ────────────────────────────────────────────────────────────────

def _iso(ts_ms) -> str:
    return datetime.utcfromtimestamp(int(ts_ms)/1000).isoformat()

def _hash_feats(f: dict) -> str:
    keys = ['adx','rsi','bb','mom','vspike','smf','trend_raw']
    src = '|'.join(str(round(float(f.get(k,0.)),1)) for k in keys)
    import hashlib
    return hashlib.md5(src.encode()).hexdigest()[:8]

def exp_top5_summary(top5: list, max_n: int = 5) -> list:
    out = []
    for e in top5[:max_n]:
        o = e.get("outcome", {})
        out.append({
            "hash":  e.get("hash", "?")[:8],
            "dir":   e.get("dir", 0),
            "hit":   o.get("hit"),
            "ret":   o.get("ret"),
            "mfe":   o.get("mfe"),
            "weight": e.get("weight", 1.0),
        })
    return out

def _regime_from_features(f: dict) -> str:
    adx = float(f.get("adx", 0))
    rsi = float(f.get("rsi", 50))
    mom = float(f.get("mom", 0))
    if adx < 18:
        return "RANGE"
    if mom > 0.5 and rsi < 70:
        return "UPTREND"
    if mom < -0.5 and rsi > 30:
        return "DOWNTREND"
    return "RANGE"


# ─── DecisionContext ─────────────────────────────────────────────────────────

@dataclass
class DecisionContext:
    trigger_reason:   str = ""
    indicators:       dict = field(default_factory=dict)
    account_state:    dict = field(default_factory=dict)
    macro_env:        dict = field(default_factory=dict)
    exp_top5:         list = field(default_factory=list)
    matched_rules:    list = field(default_factory=list)
    gate_status:      dict = field(default_factory=dict)
    position_status:  dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trigger_reason":    self.trigger_reason,
            "indicators":        self.indicators,
            "account_state":     self.account_state,
            "macro_env":         self.macro_env,
            "exp_top5_summary":  exp_top5_summary(self.exp_top5),
            "matched_rule_ids":  [r.get("id","?") for r in self.matched_rules],
            "matched_rule_decs": [r.get("rationale","") for r in self.matched_rules],
            "gate_status":       self.gate_status,
            "position_status":   self.position_status,
        }

    def has_avoidance(self) -> bool:
        return any(r.get("type") == "avoidance" for r in self.matched_rules)

    def requires_downsize(self) -> Optional[float]:
        for r in self.matched_rules:
            if r.get("type") in ("avoidance","use_condition"):
                return r.get("size_inventory_reduction", None)
        return None

    def net_confidence_delta(self, base_score: float = 0.0) -> float:
        """Sum all confidence deltas from matched rules."""
        delta = base_score
        for r in self.matched_rules:
            delta += float(r.get("confidence_penalty", 0)
                          or r.get("confidence_reward", 0))
        return float(np.clip(delta, -100.0, 100.0))


def build_ctx(
    trigger_reason: str,
    feats: dict,
    account_state: dict,
    macro_env: dict,
    exp_top5: list = None,
    gate_status: dict = None,
    position_status: dict = None,
    rule_db = None,
) -> DecisionContext:
    """
    Build a full 8-DC DecisionContext.

    Parameters
    ----------
    trigger_reason  : str   -- why the agent was called
    feats           : dict  -- indicator values at entry (from build_feats economy)
    account_state   : dict  -- {equity, dd_30d, per_coin_win_rate}
    macro_env       : dict  -- {btc_regime, funding_bias}
    exp_top5        : list  -- from ExpDB.replay(), [] if no exp
    gate_status     : dict  -- {adx_pass, rsi_pass, circuit_ok} -- REF ONLY
    position_status : dict  -- {side, pct_k, bars_held, unrealized_pct}
    rule_db         : RuleDB -- inject matched rules into DC[06]
    """
    dc = DecisionContext(
        trigger_reason  = trigger_reason,
        indicators      = {k: round(float(feats.get(k, 0)), 4) for k in
                           ("adx","rsi","bb","mom","vspike","smf","trend_raw")},
        account_state   = account_state,
        macro_env       = macro_env,
        exp_top5        = exp_top5 or [],
        gate_status     = gate_status or {},
        position_status = position_status or {},
    )
    if rule_db is not None:
        dc.matched_rules = rule_db.match(dc)
    return dc


# ─── RuleDB ─────────────────────────────────────────────────────────────────

_DEFAULT_RULES = [
    {
        "id": "rule_range_avoid_shorts",
        "type": "avoidance",
        "scope": "BTC",
        "trigger_conditions": {
            "market_regime": "RANGE",
            "rsi_range": [40, 60],
            "ls_ratio_min": 2.0,
            "macd_hist_abs_max": 0.20,
        },
        "action": "avoid_open",
        "rationale": "震荡市 + 高多空比 > 2.0 + RSI中段(40-60) + MACD紧零轴 = 反趋势高损",
    },
    {
        "id": "rule_strong_uptrend_require_confirmed",
        "type": "use_condition",
        "scope": "BTC",
        "trigger_conditions": {
            "market_regime": "UPTREND",
            "adx_min": 28,
            "vspike_min": 3.5,
            "rsi_range": [55, 72],
        },
        "action": "require_strong_signal",
        "rationale": "强趋势(ADX>28)确认 + 量价齐飞(VSPIKE>3.5) = 高确定性，允许加仓",
        "confidence_reward": 15,
    },
    {
        "id": "rule_avoid_long_if_dd15",
        "type": "avoidance",
        "scope": "BTC",
        "trigger_conditions": {
            "dd_30d_max": -0.15,
        },
        "action": "avoid_open",
        "rationale": "30日最大回撤 > 15% → 系统低胜率环境，暂停开新仓",
    },
    {
        "id": "rule_down_prefer_short",
        "type": "hard_trigger",
        "scope": "BTC",
        "trigger_conditions": {
            "market_regime": "DOWNTREND",
            "rsi_range": [30, 48],
            "macd_hist_min": -0.003,
        },
        "action": "prefer_short",
        "rationale": "空头趋势 + RSI中低段 + MACD柱持续走负 = 做空预期更高",
    },
]


class RuleDB:
    """Hard-trigger rule library. Load/save/match/inject."""

    def __init__(self, path: Path = RULE_DB_PATH):
        self.path = path
        self.rules: list = self._load()

    def _load(self) -> list:
        if self.path.exists():
            try:
                return json.load(open(self.path))
            except json.JSONDecodeError:
                return list(_DEFAULT_RULES)
        return list(_DEFAULT_RULES)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.rules, f, indent=2, ensure_ascii=False)

    def match(self, dc: DecisionContext) -> list:
        """Return all rules whose trigger_conditions match current DC state."""
        ind    = dc.indicators
        acct   = dc.account_state
        macro  = dc.macro_env
        regime = macro.get("btc_regime", _regime_from_features(ind))
        liq    = macro.get("longs_over_shorts", None)

        matched = []
        for rule in self.rules:
            tc = rule.get("trigger_conditions", {})
            if not self._hit(tc, regime, ind, acct, liq):
                continue
            matched.append(rule)
        return matched

    def _hit(self, tc: dict, regime: str, ind: dict, acct: dict,
             liq_ratio: Optional[float]) -> bool:
        for k, v in tc.items():
            if k == "market_regime":
                if v != "ANY" and v != regime:
                    return False
            elif k == "rsi_range":
                r = ind.get("rsi", 50)
                if not (v[0] <= r <= v[1]):
                    return False
            elif k == "rsi_max":
                if ind.get("rsi", 50) > v:
                    return False
            elif k == "rsi_min":
                if ind.get("rsi", 50) < v:
                    return False
            elif k == "adx_min":
                if ind.get("adx", 0) < v:
                    return False
            elif k == "vspike_min":
                if ind.get("vspike", 0) < v:
                    return False
            elif k == "vspike_max":
                if ind.get("vspike", 99) > v:
                    return False
            elif k == "ls_ratio_min":
                if liq_ratio is None or liq_ratio < v:
                    return False
            elif k == "macd_hist_abs_max":
                if abs(ind.get("mom", 0)) > v:
                    return False
            elif k == "macd_hist_min":
                if ind.get("mom", 0) < v:
                    return False
            elif k == "dd_30d_max":
                dd = acct.get("dd_30d", 0)
                if dd >= v:   # trigger fires only when dd is MORE negative than threshold
                    return False

        return True

    def add_from_review(self, entry: dict, outcome: dict) -> Optional[dict]:
        """
        Called after review_(). If outcome signals avoidance or condition,
        produce a structured rule and append to self.rules.
        """
        if not outcome.get("hit") and not outcome.get("mfe"):
            return None
        mfe = outcome.get("mfe", 0)
        if mfe < 0:
            # Lost trade -- extract avoidance rule
            import uuid
            reg = _regime_from_features(entry.get("features", {}))
            rule = {
                "id": f"rule_auto_{uuid.uuid4().hex[:8]}",
                "type": "avoidance",
                "scope": "BTC",
                "trigger_conditions": {
                    "market_regime": reg,
                    "rsi_range": [
                        max(20, int(entry.get("features", {}).get("rsi", 50) - 10)),
                        min(80, int(entry.get("features", {}).get("rsi", 50) + 10)),
                    ],
                    "vspike_max": float(entry.get("features", {}).get("vspike", 3.0)) + 0.5,
                    "adx_max": float(entry.get("features", {}).get("adx", 22)) + 5,
                },
                "action": "avoid_open",
                "rationale": (
                    f"Auto-extracted: regime={reg}, "
                    f"RSI~{entry.get('features',{}).get('rsi',50):.0f}, "
                    f"VSPIKE~{entry.get('features',{}).get('vspike',0):.1f} -- "
                    f"historical loss (mfe={mfe:.1%})"
                ),
            }
            self.rules.append(rule)
            self.save()
            return rule
        return None

    def hydrate(self, dc: DecisionContext) -> None:
        """Populate dc.matched_rules from RuleDB. Call after building DC."""
        dc.matched_rules = self.match(dc)

