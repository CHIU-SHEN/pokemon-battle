# 仓库安全整理报告（2026-07-29）

## 范围与安全模式

本次使用安全整理模式，只处理可再生成缓存、临时 Arena 输出、在线 Adapter
接入代码、相关测试与当前交接文档。未删除或移动训练数据、冻结 checkpoint、
历史实验、现有提交包和服务器回传原包。

## 已清理

- 删除开发 smoke：`experiments/top2_adapter_smoke_dev/`（2 个文件，10,129 字节）。
- 删除可再生成的正式 smoke 原始输出：`experiments/top2_adapter_smoke/`
  （40 个文件，1,271,609 字节）；保留汇总报告
  `reports/top10_adapter_online_smoke.json` 和 `.md`。
- 删除 `.pytest_cache/` 和全部 `__pycache__/`。
- 清理由历史实验和 `submission/cg/` 误纳入 Git 的 25 个 `.pyc` 文件。
- `.gitignore` 新增 Top2 smoke 原始输出和服务器回传压缩包规则。

## 已整理与新增

- 在线代理入口：`src/arena/adapter_agent.py`。
- Arena 加载方式：`adapter:<candidate_id>`。
- 统一 smoke 脚本：`scripts/run_top10_adapter_smoke.py`。
- v2 交接包构建器：`scripts/build_top2_arena_handoff.py`。
- Top2 校验脚本增加在线入口、10/10 smoke 和候选集合检查。
- `项目进度.md`、`TOP2_ARENA_SERVER_HANDOFF.md` 与 `reports/README.md`
  已同步当前阶段。
- `tests/README.md` 说明测试脚本的实际执行方式和提交包重建副作用。

## 验证结果

- 在线 Adapter fixture：通过；37 次模型选择、5 次规则回退、0 异常。
- Top2 本地交接校验：通过；10 个 Adapter、10 副牌表、主备角色、在线
  smoke 10/10 和基础数据哈希一致。
- 项目验收脚本：18/18 通过。
- 3 个会重建已跟踪提交 ZIP 的测试未在普通回归中运行：
  `test_flat_submission.py`、`test_sl0_submission.py`、`test_gru_submission.py`。
- `git diff --check`：通过，仅有 Windows CRLF 提示。

## 服务器交接包

- 文件：`server_uploads/pokemon-tcg-top2-arena-handoff-v2.tar.gz`
- 外层 SHA-256：以同目录 `.tar.gz.sha256` 最终旁车文件为准。
- 独立解压验证：全部包内文件哈希通过；在线代理测试和 Top2 校验通过。

该包用于 Top10 正式 Arena、外部矩阵与 Top2 冻结，不用于重新训练 Adapter；
明确排除了 5.4GB 监督训练 JSONL、原始 Arena 对局、`last.pt` 和服务器结果压缩包。

## 剩余风险与后续

- 底层比赛引擎不暴露 RNG seed，正式矩阵需交换先后手并保存置信区间。
- 45 个内部组合和固定外部矩阵编排尚未实现/运行。
- 当前工作区改动尚未提交 Git。

建议提交说明：

```text
整理 Top2 Arena 在线代理并生成 v2 服务器交接包
```
