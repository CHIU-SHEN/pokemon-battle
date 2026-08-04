# Top10 Adapter 在线代理 smoke 报告

> 日期：2026-07-29  
> 机器可读结果：`reports/top10_adapter_online_smoke.json`

## 结论

10 套冻结 Adapter 均已通过在线代理接入门槛，可以进入正式内部循环赛和外部
矩阵。统一脚本共运行 20 个 matchup、200 局：每个候选对 Random 10 局、镜像
10 局。

- 通过候选：10/10；
- 异常：0；
- 非法动作：0；
- Adapter 模型动作：8,921 次；
- 各候选两组对局的最大 p95 单决策耗时：2.82ms；
- 10/10 牌表哈希与 candidate ID 绑定一致，且同一 matchup 内牌表保持一致。

在线代理仅在非平凡强制单选决策上调用 Adapter；强制唯一选项直接返回，其他
选择形状使用规则回退，规则异常时使用安全回退。动作来源均写入对局汇总。

## 运行方式

```bash
python scripts/run_top10_adapter_smoke.py --games 10 --seed 20260729
```

底层比赛引擎仍不暴露 RNG seed，脚本 seed 只能控制 Python 侧代理。因此本轮
只用于工程门槛验收；每个候选仅 10 局的 Random/镜像胜率不能用于 Top2 排名。

## 阶段决定

在线接入阻塞已解除。下一步可以实现并启动 45 个内部组合的交换先后手循环赛，
同时运行固定外部对手矩阵；筛选期间仍不得覆盖 `submission/deck.csv`。

