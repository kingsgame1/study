
"""
Crypto CTA Trend v4 — Nicjiang7 经验学习版
Repository: /tmp/cta_v4_final.py
Date: 2026-05-18
"""

import pandas as pd, numpy as np, ccxt, json, os, hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

EXP_DB = Path("/home/ubuntu/CTA_expdb.json")


# ══════════════════════════════════════════════════════════════════════════
#  指标
# ══════════════════════════════════════════════════════════════════════════

def rsi(c, p=14):
    d = np.diff(c, prepend=c[0])
    g = pd.Series(np.where(d > 0, d, 0.)).ewm(alpha=1/p, adjust=False).mean().values
    l = pd.Series(np.where(d < 0, -d, 0.)).ewm(alpha=1/p, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        v = 100 - 100 / (1 + np.where(l == 0, 0.0, g / l))
    return np.nan_to_num(v)

def atr_(h, l, c, p=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    return pd.Series(tr).ewm(alpha=1/p, adjust=False).mean().values

def adx_(h, l, c, p=14):
    up = np.diff(h, prepend=h[0]); dn = np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    a = pd.Series(tr).ewm(alpha=1/p, adjust=False).mean().values
    pdi = 100 * pd.Series(pdm).ewm(alpha=1/p, adjust=False).mean().values / np.maximum(a, 1e-12)
    mdi = 100 * pd.Series(mdm).ewm(alpha=1/p, adjust=False).mean().values / np.maximum(a, 1e-12)
    dx = np.where((pdi + mdi) > 0, 200 * np.abs(pdi - mdi) / (pdi + mdi), 0.)
    return pd.Series(dx).ewm(alpha=1/p, adjust=False).mean().values

def bbpos(c, p=20, s=2):
    mid = pd.Series(c).ewm(span=p, adjust=False).mean().values
    sd  = pd.Series(c).ewm(span=p, adjust=False).std().fillna(0).values
    rng = mid + s * sd - (mid - s * sd)
    return np.where(rng > 0, (c - (mid - s * sd)) / rng - 0.5, 0.) * 2

def sma_(c, n):
    return pd.Series(c).ewm(span=n, adjust=False).mean().values

def momz(c, lb=20):
    r  = pd.Series(c).pct_change(lb).values
    mu = pd.Series(r).ewm(span=252, adjust=False).mean().fillna(0).values
    sd = pd.Series(r).ewm(span=252, adjust=False).std().fillna(1e-12).values
    return np.clip((r - mu) / sd, -3, 3)

def smart_money_flow(c, vol, lb=20):
    ret = pd.Series(c).pct_change().values
    vq  = pd.Series(vol).rolling(lb).quantile(0.5).values
    s   = np.where((ret > 0) & (vol > vq),  1,
          np.where((ret < 0) & (vol > vq), -1, 0))
    return pd.Series(s).rolling(lb).mean().fillna(0).values


# ══════════════════════════════════════════════════════════════════════════
#  ExpDB  — 经验数据库
# ══════════════════════════════════════════════════════════════════════════

class ExpDB:
    def __init__(self, path=EXP_DB):
        self.path = Path(path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f: return json.load(f)
        return []

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _hash(feats):
        """7维特征 → 8位哈希指纹：用于相似匹配"""
        bkt = '|'.join(str(round(float(feats.get(k, 0.0)), 1))
                       for k in ['adx', 'rsi', 'bb', 'mom', 'vspike', 'smf', 'trend_raw'])
        return hashlib.md5(bkt.encode()).hexdigest()[:8]

    @staticmethod
    def _iso(ts_ms):
        return datetime.utcfromtimestamp(ts_ms / 1000).isoformat()

    @staticmethod
    def ts_to_ms(iso_str):
        return int(datetime.fromisoformat(iso_str).timestamp() * 1000)

    def log_decision(self, ts_ms, price, dir_, feats):
        """记录开盘前决策快照，返回 (record_index, context_hash)"""
        rec = dict(
            ts=self._iso(ts_ms), hash=self._hash(feats),
            close=float(price), dir=int(dir_), features=feats,
            outcome={'ret': None, 'hit': None, 'mfe_pct': None, 'reviewed': False},
            weight=1.0, decay=1.0,
        )
        self.data.append(rec)
        self.save()
        return len(self.data) - 1, rec['hash']

    def review_one(self, idx, exit_ret, exit_mfe):
        """回测后对单条记录做 24h 复核"""
        rec = self.data[idx]
        rec['outcome']['ret'] = round(float(exit_ret), 4)
        rec['outcome']['mfe_pct'] = round(float(max(exit_mfe, 0)), 4)
        rec['outcome']['hit'] = bool(exit_ret > 0.001)
        rec['outcome']['reviewed'] = True
        rec['weight'] = round(1.5 if rec['outcome']['hit'] else 0.5, 2)
        rec['decay'] = 0.98
        self.save()

    def replay(self, feats, top_n=5):
        """时序衰减加权 + 哈希相似，返回 Top N 经验"""
        h_now = self._hash(feats)
        scored = []
        for rec in self.data:
            if not rec['outcome']['reviewed']:
                continue
            age_h = (datetime.now().timestamp() - datetime.fromisoformat(rec['ts']).timestamp()) / 3600
            decayed = rec['weight'] * (rec['decay'] ** max(age_h / 168, 0))
            sim = 1.0 if rec['hash'] == h_now else 0.3
            scored.append((sim * decayed, rec))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_n]]

    def conviction(self, entrants):
        """历史场景 hit rate → 修正 Conviction (10~90)"""
        if not entrants:
            return 50.0
        hr = sum(1 for e in entrants if e['outcome']['hit']) / len(entrants)
        return round(float(np.clip(50.0 + 20.0 * (hr - 0.5), 10.0, 90.0)), 1)

    def stats(self):
        rev = [r for r in self.data if r['outcome']['reviewed']]
        if not rev:
            return dict(total=len(self.data), reviewed=0, hits=0, hit_rate=0.0, avg_mfe=0.0)
        hts = [r['outcome']['hit'] for r in rev]
        mfes = [r['outcome']['mfe_pct'] or 0.0 for r in rev]
        return dict(total=len(self.data), reviewed=len(rev),
                    hits=sum(hts), hit_rate=sum(hts) / len(hts), avg_mfe=np.mean(mfes))


# ══════════════════════════════════════════════════════════════════════════
#  信号生成器
# ══════════════════════════════════════════════════════════════════════════

def build_feats(c, h, l, v, cfg):
    f = sma_(c, cfg.MF); s = sma_(c, cfg.MS)
    rv = rsi(c, cfg.RSI_P); aa = atr_(h, l, c)
    ax = adx_(h, l, c, cfg.ADX_P); bp = bbpos(c); m = momz(c, cfg.MLB)
    vm = pd.Series(v).ewm(span=cfg.VMA_P, adjust=False).mean().values
    vs = v / np.maximum(vm, 1e-12); smf = smart_money_flow(c, v)
    rn = (rv - 50.0) / 50.0
    ts_raw = ((f > s).astype(float) * 2 - 1) * 0.5 + rn * 0.3 + bp * 0.2
    tf = ax > cfg.ADX_TH; vf = vs > cfg.VSPIKE
    raw = np.where(tf & vf & (ts_raw > 0.35) & (m > 0.25),  1,
          np.where(tf & vf & (ts_raw < -0.35) & (m < -0.25), -1, 0))
    sig = pd.Series(np.where(raw == 0, np.nan, raw)).ffill().fillna(0).values
    tv = aa / np.maximum(c, 1e-12)
    med = np.concatenate([pd.Series(tv).rolling(252, min_periods=1).median().values[:252],
                          pd.Series(tv).rolling(252).median().values[252:]])
    k = cfg.TAV / (tv * np.sqrt(365.0 * 24.0))
    tgt = np.where(raw > 0, 1, np.where(raw < 0, -1, 0)) * np.clip(k, -cfg.MAXL, cfg.MAXL)
    return dict(adx=ax, rsi=rv, bb=bp, mom=m, vspike=vs, smf=smf,
                raw_sig=raw, sig=sig, tgt=tgt, atr=aa, tv=tv)


# ══════════════════════════════════════════════════════════════════════════
#  回测引擎
# ══════════════════════════════════════════════════════════════════════════

def run_backtest(c, h, l, v, ts_ms, cfg, use_exp=False):
    exp     = ExpDB() if use_exp else None
    f       = build_feats(c, h, l, v, cfg)
    n       = len(c)
    pos, ep, ea, trl, held = 0.0, 0.0, 0.0, 0.0, 0
    pke, eq = 1.0, 1.0
    rows = []
    bt_bar_idx = []          # bt_df 行 ↔ 原始 bar 索引（bar i → 行 i-1）
    entry_log = []           # [{bt_row, ts_ms, dir_, ep, db_idx, feats, conv}]
    n_bt_rows = 0            # 记录当前已经 append 了多少行到 rows

    for i in range(1, n):
        eq  *= 1 + pos * (c[i] / c[i - 1] - 1)
        pke  = max(pke, eq)

        # 退出
        exit_now = (pos > 0 and c[i] <= ep - cfg.ATR_M * ea) | \
                   (pos < 0 and c[i] >= ep + cfg.ATR_M * ea)
        if trl != 0:
            exit_now |= (pos > 0 and c[i] <= trl) or (pos < 0 and c[i] >= trl)
        if pos != 0:
            exit_now |= (eq / pke < 1 - cfg.DDR) | (held >= cfg.TSB) | (f['sig'][i] == 0)
        if exit_now:
            pos = 0.0; ep = 0.0; ea = 0.0; trl = 0.0; held = 0
            eq  *= 1 - 2 * cfg.FEE

        # 入场
        if pos == 0.0 and f['raw_sig'][i] != 0:
            sz = f['tgt'][i]
            coeff = 1.0
            if exp:
                feats_now = dict(
                    adx=round(float(f['adx'][i]), 2),
                    rsi=round(float(f['rsi'][i]), 1),
                    bb=round(float(f['bb'][i]), 2),
                    mom=round(float(f['mom'][i]), 2),
                    vspike=round(float(f['vspike'][i]), 2),
                    smf=round(float(f['smf'][i]), 4),
                    trend_raw=round(float(f['raw_sig'][i]), 3),
                )
                db_idx, _ = exp.log_decision(ts_ms[i], c[i], int(f['raw_sig'][i]), feats_now)
                # 检索相似经验 → 修正 Conviction → 仓位系数
                top_exp = exp.replay(feats_now)
                conv    = exp.conviction(top_exp)
                coeff   = conv / 50.0
                entry_log.append(dict(
                    bt_row=n_bt_rows,          # this row is still in rows[] now
                    ts_ms=ts_ms[i],
                    dir_=int(f['raw_sig'][i]),
                    ep_=c[i],
                    db_idx=db_idx,
                    feats=feats_now,
                    conv=conv,
                    coeff=coeff,
                ))
                print(f"  [IN] idx={i:5d} dir={int(f['raw_sig'][i]):+d} "
                      f"conv={conv:.0f} coeff={coeff:.2f}", flush=True)

            pos  = sz * coeff
            pos  = float(np.clip(pos, -cfg.MAXL, cfg.MAXL))
            ep   = c[i]; ea = f['atr'][i]; held = 0; trl = 0.0
            eq  *= 1 - cfg.FEE

        # 追踪止损
        if pos != 0.0 and ea > 0:
            if pos > 0:
                pnl_a = (c[i] - ep) / ea
                if pnl_a >= cfg.TRAIL_A: trl = max(trl, c[i] - cfg.TRAIL_M * ea)
            else:
                pnl_a = (ep - c[i]) / ea
                if pnl_a >= cfg.TRAIL_A:
                    tnew = c[i] + cfg.TRAIL_M * ea
                    trl = min(trl, tnew) if trl != 0.0 else tnew

        # 记录结果行
        bt_bar_idx.append(i - 1)    # 该行对应原始 bar i-1
        entry_px  = entry_log[-1]['ep_'] if entry_log and entry_log[-1]['ep_'] is not None else None
        rows.append((i - 1, c[i], f['adx'][i], f['sig'][i], pos,
                     f['raw_sig'][i], eq, pke, eq / pke - 1))
        n_bt_rows += 1

    bt_cols = ['bar_i', 'close', 'adx', 'sig', 'pos', 'raw_sig', 'eq', 'peak', 'dd']
    bt = pd.DataFrame(rows, columns=bt_cols)
    bt['ts_ms'] = ts_ms[bt['bar_i'].values]   # 精确对应原始时间戳

    return bt, exp, entry_log


# ══════════════════════════════════════════════════════════════════════════
#  回测后逐笔复盘
# ══════════════════════════════════════════════════════════════════════════

def review_all(bt, entry_log, exp_db, closes, highs, lows):
    reviewed=hits=0
    _TSB=int(144)  # TSB exit time window in bars
    bt_arr=bt['ts_ms'].values if 'ts_ms' in bt.columns else np.array([])
    bt_close=bt['close'].values
    abs_pos_arr=np.abs(bt['pos'].values)

    for lg in entry_log:
        db_rec = exp_db.data[lg['db_idx']]
        if db_rec['outcome']['reviewed']: continue
        ep_,d_=lg['ep_'],lg['dir_']

        # locate this entry's row in bt_df
        ei=np.where(bt_arr==lg['ts_ms'])[0]
        if len(ei)==0: continue
        ei_row=int(ei[0])

        # exit row: first row after entry where pos goes to 0, or _TSB bar
        exit_row=min(ei_row+_TSB,len(bt_arr)-1)
        for j in range(ei_row+1, min(ei_row+289, len(abs_pos_arr))):
            if abs_pos_arr[j]==0 and abs_pos_arr[j-1]!=0:
                exit_row=j; break

        c_entry = bt_close[ei_row]
        c24     = bt_close[min(ei_row+24, len(bt_close)-1)]
        c_exit  = bt_close[exit_row]
        ret24   = (c24-ep_)/ep_*d_
        ret_exit= (c_exit-ep_)/ep_*d_

        h_slice=highs[ei_row:exit_row] if exit_row>ei_row else np.array([ep_])
        l_slice=lows[ei_row:exit_row]  if exit_row>ei_row else np.array([ep_])
        mfe    = max(0.0,(h_slice.max()-ep_)/ep_*d_)
        mae    = max(0.0,(ep_-l_slice.min())/ep_*d_)

        hit    = ret_exit > 0.001
        db_rec['outcome']['ret']=round(float(ret_exit),4)
        db_rec['outcome']['mfe_pct']=round(float(mfe),4)
        db_rec['outcome']['mae_pct']=round(float(mae),4)
        db_rec['outcome']['hit']=bool(hit)
        db_rec['outcome']['reviewed']=True
        db_rec['weight']=round(1.5 if hit else 0.5,2)
        db_rec['decay']=0.98
        reviewed+=1; hits+=int(hit)
    exp_db.save()
    return reviewed,hits


# ══════════════════════════════════════════════════════════════════════════
#  绩效评估
# ══════════════════════════════════════════════════════════════════════════

def evaluate(bt, label=''):
    eq    = bt['eq'].values
    pk    = np.maximum.accumulate(eq)
    dd    = eq / pk - 1
    ret   = np.diff(eq) / eq[:-1]
    ar    = eq[-1] ** (365.0 * 24.0 / max(len(eq), 1)) - 1.0
    av    = ret.std() * np.sqrt(365.0 * 24.0)
    sh    = ar / av if av > 0 else 0.0
    mdd   = dd.min()
    cal   = ar / abs(mdd) if mdd < 0 else 0.0
    n_tr  = int(bt['pos'].ne(bt['pos'].shift(1)).sum())
    print(f"\n═══ {label} ═══\n"
          f"  Total Return: {eq[-1] - 1:.1%}\n"
          f"  Ann Return:   {ar:.1%}\n"
          f"  Ann Vol:      {av:.1%}\n"
          f"  Sharpe:       {sh:.2f}\n"
          f"  Max DD:       {mdd:.1%}\n"
          f"  Calmar:       {cal:.2f}\n"
          f"  Trades:       {n_tr}\n"
          f"  Bars:         {len(bt)}")
    return dict(sharpe=sh, ann_ret=ar, ann_vol=av, max_dd=mdd, calmar=cal, trades=n_tr)


# ══════════════════════════════════════════════════════════════════════════
#  CTAConf
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CTAConf:
    MF:int=20; MS:int=60; MLB:int=20
    ADX_P:int=14; ADX_TH:float=15.0; RSI_P:int=14
    VSPIKE:float=1.5; VMA_P:int=20
    TAV:float=0.60; MAXL:float=1.0
    ATR_M:float=2.0; TRAIL_A:float=3.0; TRAIL_M:float=2.0
    TSB:int=144; DDR:float=0.25; FEE:float=0.0006


# ══════════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import ccxt
    np.random.seed(42)
    print("── 拉取 BTC/USDT 1h ──")
    ex = ccxt.binance({'enableRateLimit': True})
    since = ex.parse8601('2022-01-01T00:00:00Z')
    bars = []
    while len(bars) < 2800:
        try: b = ex.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
        except Exception as e: print(f"err:{e}"); break
        if not b: break
        bars.extend(b); since = b[-1][0] + 3600_000
        print(f"  {len(bars)} bars", flush=True)
        if len(b) < 1000: break
    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df = df.dropna().reset_index(drop=True)
    c  = df['close'].values
    h  = df['high'].values
    l  = df['low'].values
    v  = df['volume'].values
    t  = df['ts'].values
    print(f"  [{len(df)} bars] {df.ts.iloc[0]}→{df.ts.iloc[-1]}  P:[{c.min():.0f},{c.max():.0f}]")
    cfg = CTAConf(ADX_TH=15., VSPIKE=1.5, TAV=0.45)

    # ① 基准
    bt0, *_ = run_backtest(c, h, l, v, t, cfg, use_exp=False)
    p0 = evaluate(bt0, '① 基准（无经验库）')

    # ② 经验增强
    bt1, exp_db, entry_log = run_backtest(c, h, l, v, t, cfg, use_exp=True)
    reviewed, hits = synchronised_review(bt1, entry_log, exp_db, c, h, l)
    print(f"\n── 逐笔复盘 ──\n"
          f"  已打分: {reviewed} | 判断正确: {hits}")
    p1 = evaluate(bt1, '② 经验增强（Conviction修正）')

    # 经验库统计
    st = exp_db.stats()
    print(f"\n── 经验库 ──\n"
          f"  总记录: {st['total']}  |  已复盘: {st['reviewed']}  |  "
          f"正确: {st['hits']}  |  HitRate: {st['hit_rate']:.0%}  |  "
          f"AvgMFE: {st['avg_mfe']:.2%}")

    delta = p1['sharpe'] - p0['sharpe']
    print(f"\n── 升级结论 ──\n"
          f"  Sharpe  {p0['sharpe']:.2f}  →  {p1['sharpe']:.2f} (Δ{delta:+.2f})\n"
          f"  MaxDD   {p0['max_dd']:.1%}  →  {p1['max_dd']:.1%}")
    print(f"\n── ExpDB 前5条记录 ──")
    for r in exp_db.data[-5:]:
        o = r['outcome']
        print(f"  {r['ts'][:19]} dir={r['dir']:+d}  ret={o.get('ret')}  hit={o.get('hit')}  "
              f"mfe={o.get('mfe_pct')}  w={r['weight']:.1f}")
    import json as _json
    with open('/tmp/cta_v4_result.json', 'w') as f:
        _json.dump(p1, f, indent=2)
