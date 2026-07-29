# 测试说明

本目录中的 `test_*.py` 是可直接执行的验收脚本，不是 pytest 函数集合。

在项目根目录设置 `PYTHONPATH` 后逐个运行，例如：

```bash
python tests/test_adapter_arena_agent.py
python scripts/verify_top2_handoff.py
```

部分提交包测试会重新生成 `final_submissions/*.zip`。在只读或需要保持工作区
不变的环境中，应优先运行与当前改动相关的测试，并在完整发布验收时单独运行
提交包重建测试。

