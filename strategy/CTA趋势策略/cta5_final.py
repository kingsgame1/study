
"""
V5 Final Results 31k bar Full
运行: /usr/bin/python3 cta5_final.py
"""
import pandas as pd, numpy as np, ccxt, json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

EXP_DB = Path("/home/ubuntu/CTA_expdb.json")

# ─── 指标 ───────────────────────────────────────────────────────────────────
def  rsi_(c,p=14):
    d=np.diff(c,prepend=c[0]); g=np.where(d>0,d,0.); l=np.where(d<0,-d,0.)
    mg=pd.Series(g).ewm(alpha=1/p,adjust=False).mean().values
    ml=pd.Series(l).ewm(alpha=1/p,adjust=False).mean().values
    with np.errstate(divide='ignore',invalid='ignore'): v=100-100/(1+np.where(ml==0,0.,mg/ml))
    return np.nan_to_num(v)

def  atr_(h,l,c,p=14):
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    return pd.Series(tr).ewm(alpha=1/p,adjust=False).mean().values

def  adx_(h,l,c,p=14):
    up=np.diff(h,prepend=h[0]); dn=np.diff(l,prepend=l[0])
    pd_,md_=np.where((up>dn)&(up>0),up,0.),np.where((dn>up)&(dn>0),dn,0.)
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    a=pd.Series(tr).ewm(alpha=1/p,adjust=False).mean().values
    pdi=100*pd.Series(pd_).ewm(alpha=1/p,adjust=False).mean().values/np.maximum(a,1e-12)
    mdi=100*pd.Series(md_).ewm(alpha=1/p,adjust=False).mean().values/np.maximum(a,1e-12)
    dx=np.where((pdi+mdi)>0,200*np.abs(pdi-mdi)/(pdi+mdi),0.)
    return pd.Series(dx).ewm(alpha=1/p,adjust=False).mean().values

def  bbpos(c,p=20,s=2):
    mid=pd.Series(c).ewm(span=p,adjust=False).mean().values
    sd=pd.Series(c).ewm(span=p,adjust=False).std().fillna(0).values
    ub=mid+s*sd; lb=mid-s*sd; rng=ub-lb
    return np.where(rng>0,(c-lb)/rng-0.5,0.)*2

def  sma_(c,n): return pd.Series(c).ewm(span=n,adjust=False).mean().values

def  momz(c,lb=20):
    r=pd.Series(c).pct_change(lb).values
    mu=pd.Series(r).ewm(span=252,adjust=False).mean().fillna(0.).values
    sd=pd.Series(r).ewm(span=252,adjust=False).std().fillna(1e-12).values
    return np.clip((r-mu)/sd,-3,3)

def  smf(c,v,lb=20):
    ret=pd.Series(c).pct_change().values; vq=pd.Series(v).rolling(lb).quantile(0.5).values
    s=np.where((ret>0)&(v>vq),1,np.where((ret<0)&(v>vq),-1,0))
    return pd.Series(s).rolling(lb).mean().fillna(0).values

def  build_feats(c,h,l,v,cfg):
    f=sma_(c,cfg.MF); s=sma_(c,cfg.MS)
    rs=rsi_(c,cfg.RSI_P); aa=atr_(h,l,c)
    ax=adx_(h,l,c,cfg.ADX_P); bp=bbpos(c); m=momz(c,cfg.MLB)
    vm=pd.Series(v).ewm(span=cfg.VMA_P,adjust=False).mean().values
    vs=v/np.maximum(vm,1e-12); smf_=smf(c,v)
    rn=(rs-50.)/50.
    ts=((f>s).astype(float)*2-1)*0.5+rn*0.3+bp*0.2
    tf=ax>cfg.ADX_TH; vf=vs>cfg.VSPIKE
    raw=np.where(tf&vf&(ts>0.35)&(m>0.25),1,
         np.where(tf&vf&(ts<-0.35)&(m<-0.25),-1,0))
    sig=pd.Series(np.where(raw==0,np.nan,raw)).ffill().fillna(0).values
    tv=aa/np.maximum(c,1e-12)
    tgt=np.where(raw>0,1,np.where(raw<0,-1,0))*np.clip(cfg.TAV/(tv*np.sqrt(365.*24.)),-cfg.MAXL,cfg.MAXL)
    return dict(adx=ax,rsi=rs,bb=bp,mom=m,vspike=vs,smf=smf_,
                raw=raw,sig=sig,tgt=tgt,atr=aa,tv=tv)


@dataclass
class CTAConf:
    MF:int=20; MS:int=60; MLB:int=20
    ADX_P:int=14; ADX_TH:float=22.0; RSI_P:int=14
    VSPIKE:float=3.0; VMA_P:int=20
    TAV:float=0.45; MAXL:float=1.0
    ATR_M:float=4.0; TRAIL_A:float=7.0; TRAIL_M:float=3.0
    TSB:int=288; DDR:float=0.15; FEE:float=0.0006


def evaluate(bt,label=''):
    eq=bt['eq'].values; pk=np.maximum.accumulate(eq)
    dd=eq/pk-1; ret_s=np.diff(eq)/eq[:-1]
    ar=eq[-1]**(365.*24./max(len(eq),1))-1.0
    av=ret_s.std()*np.sqrt(365.*24.)
    sh=ar/av if av>0 else 0.; mdd=dd.min()
    n_tr=int(bt['pos'].ne(bt['pos'].shift(1)).sum())
    print(f"\n═══ {label} ═══\n"
          f"  Total: {eq[-1]-1.0:+.2%}\n  AnnRet:{ar:+.1%}\n"
          f"  Sharpe:{sh:.2f}\n  MaxDD :{mdd:.1%}\n  Trades:{n_tr}")
    return dict(sharpe=sh,ann_ret=ar,max_dd=mdd,trades=n_tr)


class ExpDB:
    def __init__(self): self.data=self._load()
    def _load(self):
        if EXP_DB.exists():
            try: return json.load(open(EXP_DB))
            except: return []
        return []
    def save(self):
        with open(EXP_DB,'w') as f: json.dump(self.data,f,indent=2)
    @staticmethod
    def _hash(f): return __import__('hashlib').md5('|'.join(str(round(float(f.get(k,0.)),1)) for k in ['adx','rsi','bb','mom','vspike','smf','trend_raw']).encode()).hexdigest()[:8]
    @staticmethod
    def _iso(ts_ms): return datetime.utcfromtimestamp(int(ts_ms)/1000).isoformat()
    def log_dec(self,ts_ms,price,dir_,feats):
        rec=dict(ts=self._iso(ts_ms),hash=self._hash(feats),close=float(price),
                 dir=int(dir_),features=feats,
                 outcome={'ret':None,'hit':None,'mfe':None,'reviewed':False},weight=1.,decay=.98)
        self.data.append(rec); self.save(); return len(self.data)-1
    def replay(self,feats,top_n=5):
        h_now=self._hash(feats); scored=[]
        for r in self.data:
            if not r['outcome']['reviewed']: continue
            age=(datetime.now().timestamp()-datetime.fromisoformat(r['ts']).timestamp())/3600
            sim=1.0 if r['hash']==h_now else 0.3
            scored.append((sim*r['weight']*(r['decay']**max(age/168,0)), r))
        scored.sort(key=lambda x:-x[0])
        return [r for _,r in scored[:top_n]]
    def conv_(self,ents):
        if not ents: return 50.0
        hr=sum(1 for e in ents if e['outcome']['hit'])/len(ents)
        return float(np.clip(50.+20.*(hr-.5),10.,90.))
    def review_(self, bt, entries, closes, highs, lows):
        bt_c=bt['close'].values; bt_ts=bt['ts_ms'].values
        ap=np.abs(bt['pos'].values); rev=hits=0
        for lg in entries:
            r=self.data[lg['di']]           # ← renamed entry key
            if r['outcome']['reviewed']: continue
            ei_arr=np.where(bt_ts==int(lg['ts']))[0]
            if len(ei_arr)==0: continue
            er=int(ei_arr[0]); d_=lg['dir']; ep_=lg['ep']
            xr=min(er+250,len(bt_c)-1)
            for j in range(er+1,min(er+289,len(ap))):
                if ap[j]==0 and ap[j-1]!=0: xr=j; break
            ret=(bt_c[xr]-ep_)/ep_*d_; hit=ret>0.001
            mfe=max(0.,(highs[er:xr].max()-ep_)/ep_*d_ if xr>er else 0.)
            r['outcome']['ret']=round(float(ret),4)
            r['outcome']['mfe']=round(float(mfe),4)
            r['outcome']['hit']=bool(hit); r['outcome']['reviewed']=True
            r['weight']=round(1.5 if hit else .5,2)
            rev+=1; hits+=int(hit)
        self.save(); return rev,hits

    def stats(self):
        rev=[r for r in self.data if r['outcome']['reviewed']]
        if not rev: return dict(total=len(self.data),reviewed=0,hits=0,hr=0,avg_mfe=0,avg_win=0,avg_loss=0,win_count=0,loss_count=0)
        hts=[r['outcome']['hit'] for r in rev]
        mfes=[(r['outcome']['mfe'] or 0.) for r in rev]
        rts=[(r['outcome']['ret'] or 0.) for r in rev]
        wins=[r for r in rts if r>0]; losses=[r for r in rts if r<0]
        return dict(total=len(self.data),reviewed=len(rev),
                    hits=sum(hts),hr=sum(hts)/len(hts),
                    avg_mfe=np.mean(mfes),avg_win=np.mean(wins) if wins else 0,
                    avg_loss=np.mean(losses) if losses else 0,
                    win_count=len(wins),loss_count=len(losses))


def run_main(c,h,l,v,ts_ms,cfg,use_exp=False,label=''):
    EXP_DB.write_text("[]")
    exp=ExpDB()
    f=build_feats(c,h,l,v,cfg); n=len(c)
    pos,ep,ea,trl,held=0.,0.,0.,0.,0; eq=1.; pk=1.
    rows,entries=[],[]

    for i in range(1,n):
        eq*=1+pos*(c[i]/c[i-1]-1); pk=max(pk,eq)
        done=(pos>0 and c[i]<=ep-cfg.ATR_M*ea)|(pos<0 and c[i]>=ep+cfg.ATR_M*ea)
        if trl!=0: done|=((pos>0 and c[i]<=trl) or (pos<0 and c[i]>=trl))
        if pos!=0: done|=(eq/pk<1-cfg.DDR)|(held>=cfg.TSB)|(f['sig'][i]==0)
        if done: pos=0;ep=0;ea=0;trl=0;held=0; eq*=1-2*cfg.FEE
        if pos==0. and f['raw'][i]!=0:
            sz=f['tgt'][i]*1.; coeff=1.
            feats_now=dict(adx=round(float(f['adx'][i]),1),rsi=round(float(f['rsi'][i]),1),
                           bb=round(float(f['bb'][i]),2),mom=round(float(f['mom'][i]),2),
                           vspike=round(float(f['vspike'][i]),2),smf=round(float(f['smf'][i]),3),
                           trend_raw=round(float(f['raw'][i]),3))
            # entry log only if used_exp
            redis__log = 'log_attempt'
            # always log decision (even noexp, to build consistent schema)
            db_idx=exp.log_dec(ts_ms[i],c[i],int(f['raw'][i]),feats_now)
            if use_exp:
                top=exp.replay(feats_now); conv_raw=exp.conv_(top); coeff=conv_raw/50.
            entries.append(dict(ts=ts_ms[i],dir=int(f['raw'][i]),
                                ep=c[i],di=db_idx,feats=feats_now,cv=coeff))
            pos=sz*coeff; pos=float(np.clip(pos,-cfg.MAXL,cfg.MAXL))
            ep=c[i];ea=f['atr'][i];held=0;trl=0.;eq*=1-cfg.FEE
        if pos!=0. and ea>0:
            if pos>0:
                p=(c[i]-ep)/ea
                if p>=cfg.TRAIL_A: trl=max(trl,c[i]-cfg.TRAIL_M*ea)
            else:
                p=(ep-c[i])/ea
                if p>=cfg.TRAIL_A: tnew=c[i]+cfg.TRAIL_M*ea; trl=min(trl,tnew) if trl!=0. else tnew
        rows.append((c[i],f['adx'][i],f['sig'][i],pos,f['raw'][i],eq,ts_ms[i]))

    bt=pd.DataFrame(rows,columns=['close','adx','sig','pos','raw_sig','eq','ts_ms'])
    pk=bt['eq'].cummax(); bt['dd']=bt['eq']/pk-1

    if use_exp:
        rev,hits=exp.review_(bt,entries,c,h,l); db=exp.stats()
    else:
        rev=hits=0; db=dict(total=0,reviewed=0,hits=0,hr=0,avg_mfe=0,
                            avg_win=0,avg_loss=0,win_count=0,loss_count=0)

    eq_arr=bt['eq'].values; pk_arr=np.maximum.accumulate(eq_arr)
    dd2=eq_arr/pk_arr-1; rs=np.diff(eq_arr)/eq_arr[:-1]
    a_r=float(eq_arr[-1]**(365.*24./max(len(eq_arr),1))-1.)
    a_v=rs.std()*np.sqrt(365.*24.)
    sh_=a_r/a_v if a_v else 0.
    n_tr=int(bt['pos'].ne(bt['pos'].shift(1)).sum())
    net_per_t=(a_r/n_tr if n_tr else 0)
    res=dict(label=label,sharpe=sh_,ann_ret=a_r,max_dd=dd2.min(),trades=n_tr,
             reviewed=db.get('reviewed',0),hit_rate=db.get('hr',0),
             db_total=db.get('total',0),net_per_trade=net_per_t,
             avg_win=db.get('avg_win',0),avg_loss=db.get('avg_loss',0),
             win_count=db.get('win_count',0),loss_count=db.get('loss_count',0))
    # print result
    print(f"\n═══ {label} ═══\n"
          f"  Total: {eq_arr[-1]-1.0:+.2%} | Sharpe: {sh_:.2f}"
          f" | MaxDD: {dd2.min():.1%} | Trades: {n_tr}")
    if use_exp:
        print(f"  [ExpDB] reviewed={db.get('reviewed',0)} | hits={db.get('hits',0)} "
              f"| avg_win={db.get('avg_win',0):.4f} | avg_loss={db.get('avg_loss',0):.4f}")
    return res


if __name__=='__main__':
    import ccxt
    df=pd.read_parquet("/tmp/btc_30k.parquet")
    c=df['close'].values; h=df['high'].values; l=df['low'].values; v=df['volume'].values; t=df['ts'].values
    print(f"── V5 Final 30kbar: {len(df)} bars ──")
    cfg = CTAConf(ADX_TH=22., VSPIKE=3.0, TAV=0.45,
                  ATR_M=4., TRAIL_A=7., TRAIL_M=3., TSB=288, DDR=0.15)
    print(f"cfg: {cfg}")

    # Track1: noexp (signal only)
    print("\nTRACK1  signal only")
    r0=run_main(c,h,l,v,t,cfg,use_exp=False,label='v5x_noexp_ADX22_VS30')

    # Track2: with exp
    print("\nTRACK2  signal + experience")
    r1=run_main(c,h,l,v,t,cfg,use_exp=True,label='v5x_exp_ADX22_VS30')

    print(f"\n“= = = = = = = = = = = = = = = = = = = = = = = = =")
    print(f"  {'':24} | {'Sharpe':>6}  {'AnnRet':>6}  {'MaxDD':>7}  {'Trades':>6}  {'AvgWin':>7}  {'AvgLoss':>7}")
    for r in [r0,r1]:
        print(f"  {r['label']:<24} | {r['sharpe']:+6.2f}  {r['ann_ret']:+.1%}   {r['max_dd']:>7.1%}  "
              f"{r['trades']:>6}  {r.get('avg_win',0):>7.4f}  {r.get('avg_loss',0):>7.4f}")
    print(f"  {'Δ (exp - noexp)':<24} | {r1['sharpe']-r0['sharpe']:+.2f}  ..ΔTrades={r1['trades']-r0['trades']}")
    with open('/tmp/v5_final_result.json','w') as f:
        json.dump({'noexp':r0,'exp':r1},f,indent=2)
