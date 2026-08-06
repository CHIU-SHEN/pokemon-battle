# 统一缓存目录设计

日期：2026-08-06

## 目标

清除仓库根目录中现有的可再生测试与检查缓存，并让后续工具统一把缓存写入
`.cache/`，避免根目录再次出现大量 `.pytest-*`、`.test-tmp-*` 等目录。

## 范围

- 删除现有 `.pytest-*`、`.pytest_cache`、`.test-tmp-*` 和
  `.kaggle-notebook-inspect`。这些目录只包含可再生的测试或检查临时数据。
- 在 `pytest.ini` 中把 pytest 缓存和临时目录固定到 `.cache/pytest/` 下。
- 在 `.gitignore` 中整体忽略 `.cache/`，同时保留旧目录模式，防止旧命令再次污染
  Git 状态。
- 审计当前未跟踪文件及常见生成目录，补充明确安全的忽略规则。

## Git 忽略策略

采用保守策略：

- 忽略缓存、测试临时目录、运行中的 progress 文件、原始日志和可重复生成的大型归档。
- 不统一忽略 `reports/*.json` 或 Markdown 总结；这些文件可能是需要审阅、提交的正式结论。
- 不删除、不移动 `artifacts/` 和 `reports/` 中现有训练结果。
- 已被 Git 跟踪的文件不因新增忽略规则而移除。

## 目录结构

```text
.cache/
└── pytest/
    ├── cache/
    └── tmp/
```

其他工具需要本地缓存时，应在 `.cache/<tool-name>/` 下建立独立子目录。

## 验证

1. 清理后确认根目录不再存在目标缓存目录。
2. 运行关键 pytest 测试，不额外传入 `--basetemp`。
3. 确认测试通过，且新缓存只出现在 `.cache/pytest/`。
4. 检查 `git status --ignored`，确认缓存与新增的可再生文件均被正确忽略，正式报告仍可见。
