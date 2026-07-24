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
10/10 都是 0 个非法 top-1；9 个通过离线门槛，可以进入 Arena；
`alakazam_battle_cage_split` 必须先修复补充数据并重训，不能以当前
checkpoint 进入正式循环赛。

## 训练结果

| 候选 | best epoch | best valid loss | valid top-1 | 训练判断 |
| --- | ---: | ---: | ---: | --- |
| alakazam_battle_cage_split | 4 | 2.1592 | 60.10% | 曲线仍改善，但冻结 test exact 明显负迁移 |
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
| alakazam_battle_cage_split | **-5.98pp** (117) | -0.30pp (43,324) | -0.01pp (42,821) | 0 | 1.69ms | **重训** |
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

`data/adapter_views/alakazam_battle_cage_split/exact_supplement_v1.jsonl`
实际是 `observed_decision_v1`，没有 `supervision.soft_policy` 和
`supervision.head_weights`；而 `AdapterJsonlDataset` 及
`collate_training_rows` 要求输入为 `training_decision_v1`。原训练命令直接
把这份补充文件传给 `train_adapter.py`，按当前仓库代码会在读取 exact
记录时触发 `KeyError: supervision`。

因此当前 `alakazam_battle_cage_split` checkpoint 不能被视为已经正确使用
6,386 条补充 train 记录。其基础 test exact 仅有 117 条，而 sampling view
中的 452 条包含 335 条尚未转换为正式监督 schema 的补充 test 记录。完整
离线复评严格只使用冻结的 `training_decision_v1`，没有伪造补充标签。

## 下一步动作

1. 将 7,494 条补充轨迹转换并审计为独立的
   `training_decision_v1` 增量文件，保留原有 game-level split。
2. 只重训 `alakazam_battle_cage_split`，至少保留一个“不使用补充数据”的
   对照；按 exact test 表现选 checkpoint，不能只按混合 valid loss。
3. 重训通过后再启动完整 10 套单循环。当前可先用已晋级的 9 套跑预筛，
   但最终 45 组正式矩阵必须等待第 10 套复评合格。
4. Arena 仍需共同对局设置、交换先后手、固定外部对手和完整异常/非法动作
   统计。离线 top-1 提升不等价于整局胜率提升。

