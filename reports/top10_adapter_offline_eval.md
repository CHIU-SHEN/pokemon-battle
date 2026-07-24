# Top10 Adapter 训练整理与离线复评

> 评估日期：2026-07-24  
> 机器可读结果：`reports/top10_adapter_offline_eval.json`  
> 统一评估入口：`src/train/eval_adapters.py`

## 结论

训练回传包 SHA-256 为
`66702A8C85B1FFDDBB29293A33E750F023F671CC46B0B62800B7A9A4AF70FCB7`，
与随包校验文件一致。10/10 个目录都包含可加载的 `best.pt`、`last.pt`
和四轮训练指标；所有 checkpoint 都绑定同一冻结数据哈希
`E8DC4DC2784A3505EAA159255A735A2C50B907DB66A5F9AB7759BEC326062370`，
每个 Adapter 有 24,833 个参数。

统一 test 复评覆盖基础训练集的 87,992 条记录。按 sampling view 排除
1,730 条未知玩家牌表后，对每个候选分别统计 exact、similar 和 general。
10/10 都是 0 个非法 top-1。首轮只在基础训练集 test 上评估时，
`alakazam_battle_cage_split` 的 exact 子集只有 117 条并显示负迁移；
完成补充数据正式转换并纳入按整局隔离的 335 条补充 test 后，exact 共
452 条，Top-1 相对主干提升 17.37pp。最终 10/10 均通过离线门槛，可以
进入 Arena。

## 训练结果

| 候选 | best epoch | best valid loss | valid top-1 | 训练判断 |
| --- | ---: | ---: | ---: | --- |
| alakazam_battle_cage_split | 4 | 2.1592 | 60.10% | 完整 exact test +17.37pp，通过 |
| alakazam_neutralization_zone | 4 | 2.1026 | 60.59% | 正常 |
| alakazam_nighttime_mine | 3 | 2.1218 | 60.49% | epoch 3 最佳 |
| crustle_kangaskhan_cage | 2 | 2.1514 | 60.45% | epoch 2 最佳 |
| crustle_kangaskhan_petrel | 1 | 2.1549 | 60.41% | 后续 epoch 过拟合，保留 best epoch 1 |
| cynthia_garchomp_roserade | 3 | 2.1430 | 60.55% | epoch 3 最佳 |
| marnie_grimmsnarl_dudunsparce | 2 | 2.1328 | 60.46% | epoch 2 最佳 |
| marnie_grimmsnarl_froslass | 3 | 2.1255 | 60.49% | epoch 3 最佳 |
| marnie_grimmsnarl_tatsugiri | 4 | 2.1343 | 60.54% | 正常 |
| mega_starmie_dusknoir | 4 | 2.1282 | 60.58% | 正常 |

不同 sampling view 的 valid 构成和权重不同，因此 valid loss 只用于同一
候选内选 epoch，不能用来横向排 Top10。

## 冻结 test：相对共享主干的策略变化

表中为 Adapter 相对 `SL-0-shared` 的 policy top-1 百分点变化；括号内为
该层记录数。

| 候选 | exact | similar | general | 非法 | batch=1 p95 | 决定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| alakazam_battle_cage_split | **+17.37pp** (452) | -0.30pp (43,324) | -0.01pp (42,821) | 0 | 2.79ms | 晋级 Arena |
| alakazam_neutralization_zone | +0.98pp (5,136) | +0.61pp (38,305) | +0.83pp (42,821) | 0 | 1.78ms | 晋级 Arena |
| alakazam_nighttime_mine | +0.72pp (21,345) | +0.58pp (22,096) | +0.68pp (42,821) | 0 | 2.08ms | 晋级 Arena |
| crustle_kangaskhan_cage | +0.56pp (12,424) | +1.58pp (4,508) | +0.57pp (69,330) | 0 | 1.81ms | 晋级 Arena |
| crustle_kangaskhan_petrel | +4.90pp (245) | +0.53pp (16,687) | +0.50pp (69,330) | 0 | 1.77ms | 晋级 Arena，关注小 exact 样本 |
| cynthia_garchomp_roserade | +0.32pp (960) | +1.95pp (670) | +0.65pp (84,632) | 0 | 1.73ms | 晋级 Arena |
| marnie_grimmsnarl_dudunsparce | +0.00pp (113) | +0.60pp (10,163) | +0.58pp (75,986) | 0 | 1.76ms | 晋级 Arena，关注小 exact 样本 |
| marnie_grimmsnarl_froslass | +0.00pp (2,580) | +0.90pp (7,696) | +0.61pp (75,986) | 0 | 1.72ms | 晋级 Arena |
| marnie_grimmsnarl_tatsugiri | +3.61pp (195) | +0.85pp (10,081) | +0.68pp (75,986) | 0 | 1.78ms | 晋级 Arena，关注小 exact 样本 |
| mega_starmie_dusknoir | +3.88pp (965) | 无 test 样本 | +0.69pp (85,297) | 0 | 1.66ms | 晋级 Arena |

batch=1 延迟是在本地 CUDA、AMP 开启、预热 20 次后重复 100 次的模型前向
p95，不包含 observation 解析和对战引擎。共享主干 p95 为 2.13ms；所有
Adapter 都远低于项目 50ms 高延迟门槛。GPU 短测存在调度噪声，不能把
Adapter p95 比主干略低解释为 Adapter 加速。

## 关键数据问题

原始 `exact_supplement_v1.jsonl` 是 `observed_decision_v1`，不能被当前
Adapter DataLoader 直接消费。现已通过
`scripts/convert_adapter_supplement.py` 转换为
`exact_supplement_training_v1.jsonl`：7,494/7,494 条有效、100 局、
0 重复、0 非法监督动作、0 跨 split，SHA-256 为
`EFDA8273BD8AB3B349B1738AFC566633379D253BCF949FFC0360D56D562B6A5C`。

RTX 5060 上使用正式增量重训 4 epoch，用时 24 分 23 秒。新旧 best
checkpoint 的参数最大绝对差仅 `3.5e-6`，训练曲线逐轮等价，证明服务器
回传模型与正式转换后的训练结果实质一致。完整复评见
`reports/alakazam_battle_cage_split_retrain_eval.json`。

## 下一步动作

1. 10 套 Adapter 均可进入完整单循环。
2. Arena 仍需共同对局设置、交换先后手、固定外部对手和完整异常/非法动作
   统计。离线 top-1 提升不等价于整局胜率提升。
3. `alakazam_battle_cage_split` 在基础来源 117 条 exact 子集和完整 452 条
   exact test 上方向相反，后续 Arena 应重点检查其跨来源泛化，不能只依据
   +17.37pp 离线结果直接判为冠军。
