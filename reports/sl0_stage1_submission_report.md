# SL-0 Stage 1 Kaggle 阶段提交报告

## 提交包

- 文件：`final_submissions/sl0_shared_stage1.zip`
- 大小：3,708,996 bytes
- SHA-256：`FA48BA137FABD245CCDAE35F7B523EE61DFB1D952DDF1B9ED5E4468FFC12C1EB`
- 入口：根目录 `main.py`
- 模型：`artifacts/sl0_shared_full/best.pt` 导出的纯 NumPy `sl0_shared_best.npz`
- 依赖：包内不含 PyTorch，不依赖运行时联网安装

## 决策门控

`SL-0` 只处理“`minCount == maxCount == 1` 且 option 多于 1 个”的强制单选决策。多选、可空选、模型加载失败、推理异常或非法输出均回退至 V0 规则和安全动作。

## 验证

- NumPy 导出与 PyTorch checkpoint 的 logits 在容差内一致，argmax 一致。
- raw-exec 方式加载成功。
- 50 个真实 observation 合法性冒烟通过。
- zip 内无 `__pycache__` 和 `.pyc`。

## 本地与 Safe V0 对比

| 位置 | 对局 | SL-0 胜 | Safe V0 胜 | 平局 | 异常/非法 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SL-0 为 agent0 | 100 | 68 | 32 | 0 | 0 |
| SL-0 为 agent1 | 100 | 30 | 70 | 0 | 0 |
| 合计 | 200 | 98 | 102 | 0 | 0 |

SL-0 合计胜率为 49%，与 Safe V0 基本同档，尚不构成本地晋级证据。该包定位为阶段性实验提交；默认正式安全基线仍是 `submission_flat_safe_v0.zip`。
