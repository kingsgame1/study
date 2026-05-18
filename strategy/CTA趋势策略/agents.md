# CTA趋势策略 — agents.md

> 项目: Crypto CTA Trend Alpha Strategy
> 仓库: kingsgame1/study → strategy/CTA趋势策略/

## 文件说明

| 文件 | 用途 |
|---|---|
| `cta_v4_final.py` | v4 稳定版：ExpDB + Conviction 机制 |
| `cta5_final.py` | v5 稳定版：参数优化后全量验证 |
| `cta_v6.py` | v6 主程序：DC + RuleDB + T1~T5 全部升级 |
| `cta_v6_core.py` | v6 核心模块：DecisionContext (8-DC) + RuleDB |
| `v6_plan.md` | v6 升级路线图（T1~T5 任务拆解） |
| `v5_final_summary.txt` | v5 全量扫描结果摘要 |
| `v4_article_analysis.json` | v4 ExpDB 分析结果 |
| `CTA_expdb.json` | ⚠️ 注：此文件在 /home/ubuntu/ 根目录，不入库（持久化跨 run 使用） |

## 数据

- 缓存：`/tmp/btc_30k.parquet`（30k bars，2022-01 → 2025-06，不入库）
- ExpDB：`/home/ubuntu/CTA_expdb.json`（不入库，防止数据泄漏过大）

## 运行

```bash
cd /home/ubuntu/study/strategy/CTA趋势策略
python3 cta_v4_final.py      # v4 回测
python3 cta5_final.py         # v5 回测
python3 cta_v6.py             # v6 回测（需 cta_v6_core.py 同目录）
```

## 关键参数（v5x 最优）

```
VSPIKE = 3.0
ADX_TH = 22
ATR_M  = 4
TRAIL_A = 7
TSB    = 288
Fee    = 0.06% per trade (0.12% round trip)
```

## 已知 Bug / 待修复

1. `RuleDB._hit()` dd_30d 逻辑方向反向
2. `dc_match_count` 累加逻辑异常
3. v6 Sharpe 与 v5x 持平（Beta 阶段，待修复 T1~T5 实现 bug）

## 版本历史

- v4: ExpDB + 24h review loop + Conviction（首次引入经验驱动）
- v5: VSPIKE/ADX 参数扫描 + 30k 全量验证（信号改进路径确认）
- v6: 8-DC + RuleDB + 闸门自主化 + 元认知（来自 Nicjiang7 第二篇文章）

*Last updated: 2026-05-19*
