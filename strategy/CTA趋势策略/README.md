# CTA 趋势策略 — Crypto CTA Trend Alpha

基于 Investment Researcher 框架和 Nicjiang7 经验驱动交易方法论构建的加密货币 CTA 趋势跟随策略。

## 策略版本

| 版本 | 核心特性 | 状态 |
|---|---|---|
| v4 | ExpDB 经验库 + 24h 回顾 + Conviction 仓位联动 | ✅ 稳定 |
| v5 | VSPIKE/ADX 参数扫描 + 30k 全量验证 | ✅ 稳定 |
| v6 | 8-DC 决策上下文 + RuleDB 规则引擎 + 元认知 | 🔧 Beta |

## 快速开始

```bash
git clone git@github.com:kingsgame1/study.git
cd study/strategy/CTA趋势策略
python3 cta5_final.py   # 推荐先跑 v5
python3 cta_v6.py        # v6（实验性）
```

## 回测结果

### v5x（最优参数）

| 指标 | 值 |
|---|---|
| 交易数 | 414 |
| Sharpe | -0.77 |
| MaxDD | -39.3% |
| 胜率 | 36% |
| 平均盈利 | +1.56% |
| 平均亏损 | -1.26% |
| 每笔净收益 | -0.11% |

### v6（30k bar 全量）

| 配置 | Sharpe | MaxDD | ExpDB 记录 |
|---|---|---|---|
| noexp_base | -0.97 | -38.4% | 0 |
| with_exp | -0.93 | -38.9% | 442 |
| dc+rule+t5 | -0.94 | -39.0% | 658 |

## 结构性瓶颈

手续费（0.12% round trip）是主要瓶颈。avg_win ~1.56% 中手续费占 77%。突破路径：avg_win 需提升至 3%+，或 avg_loss 缩小至 -0.8% 以内，或 hit_rate 提升至 50%+。

## 参考

- Investment Researcher: `msitarzewski/agency-agents` → `finance/Investment Researcher.md`
- 经验驱动交易框架: https://x.com/i/status/2056264228860747974 (Nicjiang7)
- 策略原文参考: https://x.com/i/status/2049128952070066341 (Nicjiang7)
