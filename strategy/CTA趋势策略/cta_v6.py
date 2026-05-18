"""
v6 backtest runner (T1+T2+T3+T4+T5)
Minimal patch on top of v5x CTAConf.

Usage:
  python3 /tmp/cta_v6.py               # full 30k
  python3 /tmp/cta_v6.py --head 500    # smoke test

Run:
  1. full 30k, ADX22·VS3.0 baseline
  2. full 30k, same params + DC + RuleDB + T5
  3. compare DC vs noexp
"""
import argparse, importlib.util, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib, sys

# ── import v5 helpers ────────────────────────────────────────────────────────
sys.path.insert(0, '/tmp')
from cta_v6_core import build_ctx, RuleDB, exp_top5_summary

spec = importlib.util.spec_from_file_location("v5", "/tmp/cta5_final.py")
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

# v5 singletons
build_feats = v5.build_feats
CTAConf     = v5.CTAConf
ExpDB       = v5.ExpDB
EXP_DB      = v5.EXP_DB
evaluate    = v5.evaluate

from dataclasses import dataclass

# ── helpers ──────────────────────────────────────────────────────────────────

def regime_from(f):
    adx = float(f.get("adx", 0))
    mom = float(f.get("mom", 0))
    rsi = float(f.get("rsi", 50))
    with np.errstate(divide='ignore', invalid='ignore'):
        if adx < 18: return "RANGE"
        if mom >  0.5 and rsi < 72: return "UPTREND"
        if mom < -0.5 and rsi > 28: return "DOWNTREND"
    return "RANGE"


def non_linear_coeff(conv):
    """T5: non-linear confidence -> position coefficient."""
    return float(np.clip(0.5 + 0.5 * np.sqrt(conv / 100.0), 0.25, 1.12))


# ── backtest ─────────────────────────────────────────────────────────────────

def run(c, h, l, v, ts_ms, cfg,
        label="v6",
        use_dc=False,
        use_exp=True,
        compact=False):

    exp     = ExpDB() if use_exp else None
    rule_db = RuleDB()

    feats = build_feats(c, h, l, v, cfg)
    n = len(c)

    pos, ep, ea, trl, held = 0., 0., 0., 0., 0
    eq, pk = 1., 1.
    rows, entries, dd_hist = [], [], []
    wins, losses = 0, 0
    total_signals, dc_match_count = 0, 0
    dc_samples = []

    for i in range(1, n):
        # ── PnL ──
        eq *= 1.0 + pos * (c[i] / c[i - 1] - 1.0)
        pk   = max(pk, eq)
        eq_dd = eq / pk - 1.0

        # ── Exit check ──
        done = False
        if pos > 0 and c[i] <= ep - cfg.ATR_M * ea: done = True
        if pos < 0 and c[i] >= ep + cfg.ATR_M * ea: done = True
        if trl != 0 and ((pos > 0 and c[i] <= trl) or (pos < 0 and c[i] >= trl)):
            done = True
        if pos != 0 and (eq_dd < -cfg.DDR or held >= cfg.TSB): done = True
        if pos != 0 and feats['sig'][i] == 0: done = True

        if done:
            if ep != 0:
                r_ = (c[i] - ep) / ep * np.sign(pos)
                if r_ > 0: wins += 1
                else:      losses += 1
            pos = 0.; ep = 0.; ea = 0.; trl = 0.; held = 0.
            eq *= 1.0 - 2.0 * cfg.FEE

        # ── Entry ──
        if pos == 0.0 and feats['raw'][i] != 0:
            feats_now = dict(
                adx=round(float(feats['adx'][i]), 1),
                rsi=round(float(feats['rsi'][i]), 1),
                bb=round(float(feats['bb'][i]  ), 2),
                mom=round(float(feats['mom'][i] ), 2),
                vspike=round(float(feats['vspike'][i]), 2),
                smf=round(float(feats['smf'][i]  ), 3),
                trend_raw=round(float(feats['raw'][i]), 3))

            d_     = int(feats['raw'][i])
            regime = regime_from(feats_now)
            trig_r = 'bullish_breakout' if d_ > 0 else 'bearish_breakdown'

            top5     = exp.replay(feats_now) if (use_exp and exp) else []
            conv_base = exp.conv_(top5) if (use_exp and exp) else 50.0

            dc_avoid  = False
            conv_adj  = conv_base

            if use_dc:
                from cta_v6_core import build_ctx
                acct = dict(equity=round(float(eq), 6),
                            dd_30d=round(float(min(dd_hist[-4320:] or [0])), 4),
                            per_coin_win_rate=round(wins / max(wins + losses, 1), 3))
                macr  = dict(btc_regime=regime)
                gate  = dict(adx_pass=bool(float(feats['adx'][i]) > cfg.ADX_TH),
                             rsi_pass=35 < float(feats['rsi'][i]) < 75,
                             circuit_ok=eq_dd > -cfg.DDR,
                             smf_pass=True)  # T4: informational-only, not hard-locked
                pst   = dict(side=0, pct_k=0.5, bars_held=0, unrealized_pct=0.0)

                dc = build_ctx(
                    trigger_reason=trig_r, feats=feats_now,
                    account_state=acct, macro_env=macr,
                    exp_top5=top5, gate_status=gate,
                    position_status=pst, rule_db=rule_db)
                dc_avoid = dc.has_avoidance()
                dc_samples.append(dict(i=i, regime=regime,
                                        conv=conv_base, avoid=dc_avoid))

                # T5: non-linear + rule delta
                conv_adj = float(np.clip(conv_base + dc.net_confidence_delta(),
                                         10., 95.))
            coeff = non_linear_coeff(conv_adj)

            # ── Execute ──
            sz   = feats['tgt'][i]
            npos = sz * coeff
            npos = float(np.clip(npos, -cfg.MAXL, cfg.MAXL))
            di_  = -1
            if use_exp and exp:
                di_ = exp.log_dec(ts_ms[i], c[i], d_, feats_now)
            entries.append(dict(ts=ts_ms[i], dir=d_, ep=c[i],
                                 di=di_, feats=feats_now))
            pos   = npos
            ep    = c[i]
            ea    = feats['atr'][i]
            held  = 0.
            trl   = 0.
            eq   *= 1.0 - cfg.FEE
            total_signals += 1
            dd_hist.append(float(eq_dd))

        # ── Trail ──
        if pos != 0. and ea > 0.:
            if pos > 0:
                p_ = (c[i] - ep) / ea
                if p_ >= cfg.TRAIL_A: trl = max(trl, c[i] - cfg.TRAIL_M * ea)
            else:
                p_ = (ep - c[i]) / ea
                if p_ >= cfg.TRAIL_A:
                    tnew = c[i] + cfg.TRAIL_M * ea
                    trl = min(trl, tnew) if trl != 0. else tnew

        dd_hist.append(float(eq_dd))
        rows.append((c[i], float(feats['adx'][i]), float(feats['sig'][i]),
                     float(pos), float(feats['raw'][i]), float(eq), int(ts_ms[i])))

    bt    = pd.DataFrame(rows, columns=['close','adx','sig','pos',
                                         'raw_sig','eq','ts_ms'])
    pk    = bt['eq'].cummax(); bt['dd'] = bt['eq'] / pk - 1
    eq_v  = bt['eq'].values
    ar    = float(eq_v[-1] ** (365.*24. / max(len(eq_v), 1)) - 1.)
    ret_  = np.diff(eq_v) / eq_v[:-1]
    av    = float(ret_.std() * np.sqrt(365.*24.))
    sh    = ar/av if av > 0 else 0.
    mdd   = float((eq_v / np.maximum.accumulate(eq_v) - 1.).min())
    n_tr  = int(bt['pos'].ne(bt['pos'].shift(1)).sum())
    print(f"\n[{label}  n={n}] Sharpe:{sh:.2f} AnnRet:{ar:+.1%}"
          f" MaxDD:{mdd:.1%} Trades:{n_tr}")

    if use_exp and exp and not compact:
        bt_ts = bt['ts_ms'].values.astype(np.int64)
        bt_c  = bt['close'].values
        rev, hits = exp.review_(bt=bt, entries=entries, closes=c,
                                 highs=h, lows=l)
        db = exp.stats()
        print(f"  [ExpDB] reviewed={db['reviewed']} hits={db['hits']}"
              f" hr={db['hr']:.1%}"
              f" avg_win={db.get('avg_win',0):+.4f}"
              f" avg_loss={db.get('avg_loss',0):+.4f}")
    else:
        db = dict(total=0, reviewed=0, hits=0, hr=0,
                  avg_mfe=0, avg_win=0, avg_loss=0)

    if use_dc:
        a_r = sum(1 for s in dc_samples if s['avoid']) / max(len(dc_samples), 1)
        print(f"  [DC]  signals={total_signals} dc_built={dc_match_count}"
              f" dc_avoid_rate={a_r:.1%}")
        with open('/tmp/v6_dc_samples.json', 'w') as fv:
            json.dump(dict(avoid_rate=a_r, samples=dc_samples[:20]), fv, indent=2)

    return dict(label=label, sharpe=sh, ann_ret=ar, max_dd=mdd, trades=n_tr,
                reviewed=db.get('reviewed', 0), hit_rate=db.get('hr', 0),
                avg_win=db.get('avg_win', 0), avg_loss=db.get('avg_loss', 0),
                total_signals=total_signals)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', type=int, default=0)
    ap.add_argument('--compact', action='store_true')
    ap.add_argument('--noexp', action='store_true')
    ap.add_argument('--nodc',  action='store_true')
    args = ap.parse_args()

    print("Loading /tmp/btc_30k.parquet ...")
    df = pd.read_parquet("/tmp/btc_30k.parquet")
    c = df['close'].values; h = df['high'].values
    l = df['low'].values; v = df['volume'].values; t = df['ts'].values

    n = len(c)
    if args.head > 0:
        n = min(args.head, len(c))
        c, h, l, v, t = c[:n], h[:n], l[:n], v[:n], t[:n]

    cfg = CTAConf(ADX_TH=22.0, VSPIKE=3.0, TAV=0.45,
                  ATR_M=4., TRAIL_A=7., TRAIL_M=3., TSB=288, DDR=0.15)
    print(f"Bars: {n}  |  Param: ADX={cfg.ADX_TH} VS={cfg.VSPIKE} TRAIL={cfg.TRAIL_A} TSB={cfg.TSB}")
    print(f"ExpDB: {not args.noexp}  |  DC+RuleDB: {not args.nodc}")

    # Track1: signal only (v5x_noexp equivalent)
    r0 = run(c, h, l, v, t, cfg, label='v6_noexp_base',
             use_dc=False, use_exp=False, compact=args.compact)

    # Track2: v5x + exp only
    r1 = run(c, h, l, v, t, cfg, label='v5x_with_exp',
             use_dc=False, use_exp=not args.noexp, compact=args.compact)

    # Track3: v6 full (DC+RuleDB+T5 + exp)
    r2 = run(c, h, l, v, t, cfg, label='v6_dc_rule_t5',
             use_dc=True, use_exp=not args.noexp, compact=args.compact)

    print()
    print("=" * 60)
    print(f"{'Label':25} {'Sharpe':>7} {'AnnRet':>7} {'MaxDD':>7} {'Trades':>6}")
    print("-" * 60)
    for r in [r0, r1, r2]:
        print(f"  {r['label']:<23} {r['sharpe']:+7.2f} {r['ann_ret']:+7.1%}"
              f"{r['max_dd']:>7.1%} {r['trades']:>6}")
    print(f"\nDelta DC - noDCExp : {r2['sharpe'] - r1['sharpe']:+.2f}")
    with open('/tmp/v6_result.json', 'w') as f_:
        json.dump(dict(noexp=r0, noexp_exp=r1, dc_rule_exp=r2), f_, indent=2)
    print("Saved /tmp/v6_result.json")
