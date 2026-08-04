# 服务器 GRU 冻结 Test：当前唯一操作路线

> 当前任务只有一个：使用已经训练好的首轮 GRU，生成两个详细 test JSON。  
> 本轮不训练模型，不跑 history，不跑第二 seed，不做 Arena。

## 先记住这两个词

现在要运行：

```text
eval_sequence
```

现在绝对不要运行：

```text
train_sequence
train_history
--seed 20260721
```

“冻结 test”没有额外的冻结按钮。V3 包里的 test 数据和首轮模型已经固定。我们只是读取它们并生成报告。

## 第 1 步：如果错误训练还在运行，先停止

如果终端仍被下面这种命令占用：

```text
python -m src.train.train_sequence
```

另开一个 SSH 窗口，执行：

```bash
pgrep -af "src.train.train_sequence"
```

找到包含 `sl1_gru_seed20260721` 的进程 PID，例如 `12345`，然后执行：

```bash
kill -INT 12345
```

等待 10 秒，再检查：

```bash
pgrep -af "src.train.train_sequence"
```

如果该进程仍存在：

```bash
kill -TERM 12345
```

确认已经停止：

```bash
nvidia-smi
```

不要杀其他用户或其他项目的进程。

## 第 2 步：进入正确目录

在服务器执行。将路径换成你的实际解压位置：

```bash
cd /home/qiumz/pokemon_train/pokemon-tcg-sl0-sl1-handoff-v3
```

检查当前位置：

```bash
pwd
```

应以这个目录结尾：

```text
pokemon-tcg-sl0-sl1-handoff-v3
```

## 第 3 步：确认 Python 环境和 GPU

如果提示符开头已经显示 `(pokemon)`，说明环境通常已经激活。继续执行：

```bash
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

必须看到：

```text
cuda= True
```

如果是 `False`，停止并处理 GPU 环境，不要继续。

## 第 4 步：确认首轮模型存在

```bash
ls -lh artifacts/sl1_gru_full/best.pt artifacts/sl1_gru_full/last.pt
```

必须同时显示：

```text
artifacts/sl1_gru_full/best.pt
artifacts/sl1_gru_full/last.pt
```

如果缺少任意一个文件，停止，不要训练替代它。

## 第 5 步：创建报告目录

```bash
mkdir -p reports
```

## 第 6 步：评估 best.pt

完整复制下面这一行：

```bash
python -m src.train.eval_sequence --checkpoint artifacts/sl1_gru_full/best.pt --split test --device cuda --batch-size 64 --window-length 16 --output reports/sl1_gru_best_detailed_test.json
```

等待命令自己结束。它会在终端打印一大段 JSON，然后重新出现命令提示符。

这一步不会训练，不会修改 `best.pt`。

## 第 7 步：评估 last.pt

第 6 步结束后，再完整复制下面这一行：

```bash
python -m src.train.eval_sequence --checkpoint artifacts/sl1_gru_full/last.pt --split test --device cuda --batch-size 64 --window-length 16 --output reports/sl1_gru_last_detailed_test.json
```

同样等待命令结束并重新出现提示符。

这一步不会训练，不会修改 `last.pt`。

## 第 8 步：确认两个报告存在

```bash
ls -lh reports/sl1_gru_best_detailed_test.json reports/sl1_gru_last_detailed_test.json
```

必须同时显示两个文件。然后检查 JSON：

```bash
python -c "import json; json.load(open('reports/sl1_gru_best_detailed_test.json')); json.load(open('reports/sl1_gru_last_detailed_test.json')); print('OK: two reports are valid')"
```

必须看到：

```text
OK: two reports are valid
```

## 第 9 步：打包两个报告

```bash
tar -czf gru-detailed-test-results.tar.gz reports/sl1_gru_best_detailed_test.json reports/sl1_gru_last_detailed_test.json
```

生成校验文件：

```bash
sha256sum gru-detailed-test-results.tar.gz > gru-detailed-test-results.tar.gz.sha256
```

检查：

```bash
ls -lh gru-detailed-test-results.tar.gz gru-detailed-test-results.tar.gz.sha256
```

## 第 10 步：停止

到这里当前任务已经完成。

不要继续运行任何训练命令。尤其不要运行：

```text
python -m src.train.train_sequence
python -m src.train.train_history
--seed 20260721
```

从服务器下载：

```text
gru-detailed-test-results.tar.gz
gru-detailed-test-results.tar.gz.sha256
```

把这两个文件交回本项目分析。我们看完结果后，再单独生成下一阶段的唯一操作指南。

## 出错时怎么做

任何一步出现 traceback、`ERROR`、CUDA 不可用或文件不存在，都立即停止。

请保存并发回：

```text
报错完整截图
实际执行的命令
pwd 输出
nvidia-smi 输出
```

不要自己跳到后面的训练步骤。

## 暂勿阅读的后续参考

未来的 history、第二 seed 和 Arena 方案已移动到：

```text
docs/operations/服务器SL0-history与GRU后续实验参考（暂勿执行）.md
```

该文件不是当前操作指南。只有本轮两个详细 JSON 分析完成后，才会决定是否使用。
