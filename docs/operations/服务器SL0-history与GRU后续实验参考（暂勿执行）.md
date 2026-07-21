# 服务器 SL-0-history 与 GRU 训练指南

本阶段比较 24 维显式历史特征、冻结的 `SL-0-shared` 基线与 `SL-1-gru`。GRU 不读取对手隐藏选择，而是编码“当前公开局面 + 上一步己方动作 + 相邻两次己方观察之间的公开局面差分”。截至 2026-07-20，GRU 已完成服务器 6 epoch 全量训练和首次冻结 test；`SL-0-history` 全量 A/B、GRU 详细复评、第二 seed 与固定 Arena 仍待完成。

## 你现在只做这一件事

现在**不要训练任何模型**，也不要先看后面的 history、第二 seed、Arena。

“冻结 test 评估”在你这里没有额外的“冻结按钮”或“冻结命令”。交接包里的 test 数据和首轮 GRU 模型已经固定好了。你只需要：

```text
上传并解压 V3
→ 准备 Python/GPU
→ 运行 best.pt 评估
→ 运行 last.pt 评估
→ 得到两个 JSON
→ 停止，把两个 JSON 下载回来
```

服务器项目目录准备好后，只执行下面这组命令：

```bash
cd /path/to/work/pokemon-tcg-sl0-sl1-handoff-v3
mkdir -p reports

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_best_detailed_test.json

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/last.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_last_detailed_test.json

ls -lh \
  reports/sl1_gru_best_detailed_test.json \
  reports/sl1_gru_last_detailed_test.json
```

最后 `ls` 能显示这两个文件，就完成了当前任务：

```text
reports/sl1_gru_best_detailed_test.json
reports/sl1_gru_last_detailed_test.json
```

此时立即停止，不要继续训练。把这两个 JSON 下载回来交给本项目分析，然后我们再决定是否跑 history 和第二 seed。

如果你还没上传和配置服务器，继续看 A、B、C 三节；如果环境已经准备好，直接执行上面的命令即可。后面的 D～10 节是解释和后续备用步骤，当前不要求一次全部完成。

## 阅读方式：第一次使用服务器的人先看这里

本文中的命令分为两类：

- 标注“在本地 Windows PowerShell 执行”的命令，在你自己的电脑上运行；
- 标注“在服务器执行”的命令，先 SSH 登录服务器，再在服务器终端运行。

代码块开头的 `$`、用户名、服务器地址等提示符不要照抄。`YOUR_USER`、`SERVER_HOST` 和 `/path/to/work` 是占位符，必须替换成管理员提供的真实值。

几个常用词：

- **交接包**：需要上传的 `.tar.gz` 大文件，里面已有代码、数据和首轮模型；
- **checkpoint**：模型权重文件，扩展名为 `.pt`；
- **seed**：控制随机性的数字，用另一个 seed 重跑可检查结果是否稳定；
- **test**：冻结测试集，只用于最终比较，不能参与训练；
- **CUDA**：让 PyTorch 使用 NVIDIA GPU；
- **退出码 0**：命令成功完成；看到 traceback、`ERROR` 或非 0 状态就不要继续下一步。

本轮需要上传的只有两个文件：

```text
pokemon-tcg-sl0-sl1-handoff-v3.tar.gz
pokemon-tcg-sl0-sl1-handoff-v3.tar.gz.sha256
```

不要在本地解压交接包，也不要上传旧 V1/V2 包。`.tar.gz` 是实际内容，`.sha256` 是用于检查上传是否损坏的验货单。

## A. 从本地上传并登录服务器

### A.1 确认本地文件

在本地 Windows PowerShell 中进入项目目录：

```powershell
cd "E:\学校文件\kaggle\pokemon-battle"
Get-Item `
  ".\release_assets\pokemon-tcg-sl0-sl1-handoff-v3.tar.gz", `
  ".\release_assets\pokemon-tcg-sl0-sl1-handoff-v3.tar.gz.sha256"
```

应看到大包约 `400555601` 字节，校验文件约 `104` 字节。如果文件不存在，不要继续上传。

### A.2 上传

如果服务器支持 `scp`，在本地 PowerShell 执行下面命令。先把三个占位符替换掉：

```powershell
scp `
  ".\release_assets\pokemon-tcg-sl0-sl1-handoff-v3.tar.gz" `
  ".\release_assets\pokemon-tcg-sl0-sl1-handoff-v3.tar.gz.sha256" `
  YOUR_USER@SERVER_HOST:/path/to/work/
```

例子仅用于理解格式：如果用户名是 `alice`、服务器是 `gpu.example.com`、目录是 `/home/alice/work`，最后一段应写成：

```text
alice@gpu.example.com:/home/alice/work/
```

如果服务器只能通过网页、堡垒机、WinSCP 或学校文件平台上传，就在对应界面把这两个文件上传到同一个目录。不要上传整个项目文件夹。

### A.3 SSH 登录

仍在本地 PowerShell 执行：

```powershell
ssh YOUR_USER@SERVER_HOST
```

成功后命令提示符会变成服务器的 Linux 提示符。后续命令都在服务器执行，直到明确写“回到本地”。

## B. 服务器解压前检查

进入刚才上传文件的目录：

```bash
cd /path/to/work
pwd
ls -lh pokemon-tcg-sl0-sl1-handoff-v3.tar.gz*
```

`pwd` 应显示预期目录；`ls` 应同时显示 `.tar.gz` 和 `.sha256`。

校验上传完整性：

```bash
sha256sum -c pokemon-tcg-sl0-sl1-handoff-v3.tar.gz.sha256
```

唯一可接受的核心结果是：

```text
pokemon-tcg-sl0-sl1-handoff-v3.tar.gz: OK
```

如果显示 `FAILED`，删除服务器上损坏的两个文件并重新上传，不要尝试解压。

检查磁盘空间：

```bash
df -h .
```

建议当前分区至少保留 15 GiB 可用空间，用于解压约 11 GiB 数据、训练 checkpoint 和报告。

解压并进入项目：

```bash
tar -xzf pokemon-tcg-sl0-sl1-handoff-v3.tar.gz
cd pokemon-tcg-sl0-sl1-handoff-v3
pwd
ls
```

`ls` 至少应看到：

```text
START_HERE.md
SERVER_TRAINING_GUIDE.md
SHA256SUMS
data
artifacts
src
tests
```

后续无论断线重连多少次，都要先执行：

```bash
cd /path/to/work/pokemon-tcg-sl0-sl1-handoff-v3
```

## C. 准备 Python 和 GPU 环境

### C.1 优先使用服务器已有环境

先检查：

```bash
python --version
nvidia-smi
```

需要 Python 3.11 或兼容版本，并且 `nvidia-smi` 能看到 NVIDIA GPU。如果学校使用 Slurm，登录节点可能看不到 GPU；此时应先按管理员要求申请 GPU 计算节点，例如：

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=12:00:00 --pty bash
```

这只是常见示例，分区名、账户和资源参数必须服从服务器规定。如果管理员提供了专用提交命令，以管理员说明为准。

### C.2 使用 Conda（推荐）

如果服务器有 Conda：

```bash
conda create -n pokemon-tcg python=3.11 -y
conda activate pokemon-tcg
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

断线重连后要重新执行：

```bash
conda activate pokemon-tcg
cd /path/to/work/pokemon-tcg-sl0-sl1-handoff-v3
```

如果服务器没有 Conda，可使用普通虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

不要同时使用 Conda 和 `.venv`。

### C.3 确认 PyTorch 真正识别 GPU

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

预期 `cuda=True`，并显示 GPU 名称。若为 `False`，不要开始全量训练；先检查是否仍在登录节点、是否申请了 GPU、PyTorch 是否为 CUDA 版本。

## D. “冻结 test 评估”具体怎么操作

“冻结”不是再训练一种模型，也不是把文件压缩起来。它表示：

1. test 对局和序列索引固定，不能加入训练；
2. 先用 validation 选择 checkpoint，选定后不再根据 test 结果更换训练参数；
3. 对固定 checkpoint 做只读推理，生成报告；
4. 保存数据、checkpoint 和报告哈希，保证以后能复现同一次评估。

V3 交接包已经完成 train/valid/test 按整局切分。你不需要手工拆分 test，也不需要复制 5 GiB JSONL。下面是实际操作。

### D.1 校验被冻结的数据

在服务器项目根目录执行：

```bash
sha256sum -c SHA256SUMS
```

该命令会读取大数据，可能需要几分钟。每一行都必须显示 `OK`。其中：

- `training_decisions_v1.jsonl` 是 GRU 使用的事实层；
- `sequence_trajectories_v1.jsonl` 固定了同视角轨迹和 test endpoint；
- `sequence_manifest_v1.json` 保存数据来源和索引哈希；
- `training_decisions_history_v1.jsonl` 是 history 模型使用的对应数据视图。

如果任意一行 `FAILED`，停止操作并重新解压 V3。不要重新生成数据来“修复”哈希。

### D.2 锁定首轮 checkpoint 并保存哈希

首轮 GRU 的 checkpoint 已由 validation 选好：

- `best.pt`：validation 总 loss 最低；
- `last.pt`：最后一个 epoch，用于检查 policy/value 分叉。

先保存它们的哈希：

```bash
mkdir -p reports
sha256sum \
  artifacts/sl1_gru_full/best.pt \
  artifacts/sl1_gru_full/last.pt \
  > reports/sl1_gru_seed20260717_checkpoints.sha256

cat reports/sl1_gru_seed20260717_checkpoints.sha256
```

再设为只读，防止误续训覆盖：

```bash
chmod a-w \
  artifacts/sl1_gru_full/best.pt \
  artifacts/sl1_gru_full/last.pt
```

这不会妨碍评估，因为评估只读取 checkpoint。如果以后确实需要恢复写权限，必须先说明原因，再使用 `chmod u+w 文件名`；本轮不需要这样做。

### D.3 执行冻结 test

冻结 test 不是一个单独的 `freeze` 命令，实际就是对固定 checkpoint 运行详细评估器：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_best_detailed_test.json

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/last.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_last_detailed_test.json
```

各参数的含义：

- `--checkpoint`：被冻结、只读的模型；
- `--split test`：明确只评估冻结 test，不使用 train/valid；
- `--window-length 16`：必须与训练和对比口径一致；
- `--output`：把结果写到新 JSON，不修改模型；
- `--device cuda`：使用 GPU 推理。

运行过程中不会出现 epoch，也不会更新权重。它只会读取 test 窗口、计算预测并写报告。

### D.4 检查报告并把报告也冻结

```bash
python -c "import json; [json.load(open(p)) for p in ['reports/sl1_gru_best_detailed_test.json','reports/sl1_gru_last_detailed_test.json']]; print('OK: frozen test reports are valid JSON')"

sha256sum \
  reports/sl1_gru_best_detailed_test.json \
  reports/sl1_gru_last_detailed_test.json \
  > reports/sl1_gru_seed20260717_frozen_test.sha256

chmod a-w \
  reports/sl1_gru_best_detailed_test.json \
  reports/sl1_gru_last_detailed_test.json \
  reports/sl1_gru_seed20260717_frozen_test.sha256
```

看到 `OK: frozen test reports are valid JSON`，并生成 `.sha256`，才算这次冻结 test 操作完成。

### D.5 冻结之后允许和禁止做什么

允许：

- 读取和比较报告；
- 用预先固定的第二 seed 重新训练到另一个目录；
- 对第二 seed 重复同样的冻结评估；
- 开发在线运行时和 Arena。

禁止：

- 把 test 样本加入训练；
- 因为 test 不好而反复修改模型，再对同一个 test 无限试验；
- 覆盖首轮 `best.pt`、`last.pt` 或详细 test JSON；
- 改 `window-length` 后仍声称与 SL-0 同口径；
- 手工修改报告数字。

如果根据这次 test 结果决定修改 loss 或网络结构，新版本必须有新的模型版本名，并把当前 test 视为已经参与过决策；正式发布前应准备新的最终保留集，不能继续把同一 test 当完全未见数据。

## 0. 当前起点与最短执行路线

本指南保留了从零训练所需的完整命令，但当前服务器不应重跑已经完成的首轮 GRU。假定服务器已有：

- `artifacts/sl1_gru_full/best.pt`；
- `artifacts/sl1_gru_full/last.pt`；
- 第 2 节列出的训练数据和 manifest；
- CUDA 可用的 Python/PyTorch 环境。

从当前状态严格按以下编号执行：

1. 完成 A～C 节的上传、解压和环境检查；
2. 执行 D 节，锁定输入并详细复评首轮 GRU best/last；
3. 按第 6.3 节判断是否继续；未通过就停止，不跑第二 seed；
4. 通过后执行第 3～4 节，补齐 `SL-0-history` 全量对照；
5. 执行第 6.4 节，在独立目录训练第二 seed 并详细评估；
6. 下载第 8 节列出的结果，由本地完成三模型离线汇总；
7. 离线门槛通过后再开发在线运行时；当前指南不会直接产出 Arena 胜率。

所有命令都从交接包或仓库根目录执行。运行前先创建报告目录：

```bash
mkdir -p reports
```

建议为本次终端保留完整日志。执行长命令时可以在命令末尾加：

```text
2>&1 | tee 对应日志名.log
```

例如：

```bash
python -m src.train.eval_sequence --help 2>&1 | tee logs_eval_help.log
```

`tee` 会同时在屏幕显示并保存日志。训练命令不要放到后台后就关闭终端；如果服务器没有作业系统，建议使用 `tmux`：

```bash
tmux new -s pokemon
```

临时离开按 `Ctrl+B`，再按 `D`；回来后运行 `tmux attach -t pokemon`。

## 1. 已完成的本机验证

24 维特征只读取当前决策之前、同一玩家视角可见的动作，包括最近 8 步动作计数、当前回合资源使用计数和距关键动作的步数，不读取未来动作。

```bash
python -m src.train.train_history \
  --device cuda --epochs 3 --batch-size 64 --num-workers 0 \
  --max-train-samples 10000 --max-valid-samples 2000 \
  --shuffle-buffer 2048 --output artifacts/dev_smoke/sl0_history_10k
```

3 个 epoch 均正常完成，每轮约 6–7 秒；最佳 valid loss 为 `2.1112`，并生成全部 checkpoint 和运行记录。该结果只证明数据、反向传播、CUDA 和 checkpoint 流程可用，不能证明稳定增益。

## 2. 服务器文件与校验

优先上传新版自包含交接包：

```text
release_assets/pokemon-tcg-sl0-sl1-handoff-v3.tar.gz
SHA-256 见同目录 pokemon-tcg-sl0-sl1-handoff-v3.tar.gz.sha256
```

在服务器解压后，先阅读 `START_HERE.md` 并运行 `sha256sum -c SHA256SUMS`。V3 包已包含下面列出的两份数据视图、序列索引、Combo 标签、SL-0 最优 checkpoint、首轮 GRU best/last checkpoint、首次 test 报告、训练与详细评估代码、测试和本指南，不需要另外克隆仓库，也不需要另外上传首轮 GRU 产物。

如果沿用第一次 GRU 训练目录而不是重新解压交接包，必须至少同步以下新版文件：

```text
src/train/eval_sequence.py
src/train/shared_data.py
```

其中 `shared_data.py` 包含序列嵌套 batch 的 CUDA 搬运修复，不能只上传评估脚本。

需要以下文件：

- `data/training/training_decisions_history_v1.jsonl`（约 5.20 GiB）；
- `data/training/training_history_manifest_v1.json`；
- `data/training/training_decisions_v1.jsonl`；
- `data/training/sequence_trajectories_v1.jsonl`；
- `data/training/sequence_manifest_v1.json`；
- `artifacts/sl0_shared_full/best.pt`；
- 当前仓库代码。

历史数据 SHA-256：

```text
35AF23BEC88280A879DD2A641A4EB315C3A5445EBA8FF5F3509A63EE13C80CE4
```

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
test -f data/training/training_decisions_v1.jsonl
test -f data/training/sequence_trajectories_v1.jsonl
test -f data/training/sequence_manifest_v1.json
test -f artifacts/sl0_shared_full/best.pt
test -f artifacts/sl1_gru_full/best.pt
test -f artifacts/sl1_gru_full/last.pt
sha256sum data/training/training_decisions_history_v1.jsonl
PYTHONPATH=. python tests/test_shared_training.py
PYTHONPATH=. python tests/test_history_training.py
PYTHONPATH=. python tests/test_sequence_index.py
PYTHONPATH=. python tests/test_sequence_model.py
```

预期历史数据 SHA-256 必须严格等于上面的值，所有 `test -f` 和测试命令退出码必须为 `0`。任一失败都先修复文件或环境，不继续训练/评估。

## 3. 正式训练 SL-0-history

先做服务器冒烟：

```bash
python -m src.train.train_history \
  --device cuda --epochs 1 --batch-size 64 --num-workers 0 \
  --max-train-samples 10000 --max-valid-samples 2000 \
  --output artifacts/sl0_history_server_smoke
```

再做全量单卡训练：

```bash
python -m src.train.train_history \
  --device cuda --epochs 6 --batch-size 256 --num-workers 4 \
  --shuffle-buffer 8192 --learning-rate 3e-4 \
  --output artifacts/sl0_history_full
```

训练默认从 `artifacts/sl0_shared_full/best.pt` 热启动。新增 24 维输入对应的权重初始化为零，因此开始时输出与原 SL-0 一致。若显存不足，先减小 `--batch-size`，再用 `--grad-accum` 保持有效 batch。

训练过程中每个 epoch 会打印一行 JSON。只要仍在持续输出且没有 traceback，就让它完成。结束后执行：

```bash
ls -lh artifacts/sl0_history_full/
```

至少应看到 `best.pt`、`last.pt`、`metrics.jsonl` 和 `run_config.json`。缺少 `best.pt` 或 `last.pt` 表示训练没有正常完成。

断点续训不需要再次传初始化 checkpoint：

```bash
python -m src.train.train_history \
  --device cuda --epochs 8 \
  --resume artifacts/sl0_history_full/last.pt \
  --output artifacts/sl0_history_full
```

## 4. 冻结测试集评估与 A/B 门槛

```bash
python -m src.train.eval_shared \
  --checkpoint artifacts/sl0_history_full/best.pt \
  --data data/training/training_decisions_history_v1.jsonl \
  --manifest data/training/training_history_manifest_v1.json \
  --split test --device cuda --batch-size 256 \
  --output reports/sl0_history_test.json
```

检查报告是否成功生成：

```bash
python -c "import json; r=json.load(open('reports/sl0_history_test.json')); print(json.dumps({'overall':r['overall'],'non_forced':r['non_forced'],'legality':r['legality']}, indent=2))"
```

看到格式化 JSON 且 `illegal_top1_predictions` 为数字，说明报告可用。不要手工修改 JSON。

基线为 `reports/sl0_shared_test.json`：test loss `2.1134`、policy top-1 `60.27%`、非强制单选 top-1 `57.72%`、value MSE `0.8916`、非法 top-1 为 `0`。

进入 GRU 前至少满足：

1. 非强制单选 top-1 提升至少 `0.3` 个百分点，且非法 top-1 仍为 `0`；
2. test loss 不高于基线，value MSE 不出现明显退化；
3. 换一个 seed 复跑时，核心指标提升方向一致；
4. 固定 Arena 的胜率或 Combo 完成率提升，且推理延迟可接受。

若只有单次离线评估的小幅波动，不算稳定增益，应保留 SL-0 基线并停止扩大模型。

## 5. SL-1-gru 输入契约

每条轨迹仍按 `game_id + current_player` 构建，保证训练和比赛推理使用相同视角。每个时间步包含：

- 当前 `SL-0` 状态与动态 options；
- 上一步己方实际动作对应的 option 特征均值；
- 当前与上一次己方观察之间的 24 维变化：前 22 个公开动态状态的差值、回合切换标志、可见日志存在标志。

该差分会显式呈现对手造成的公开结果，例如对手手牌/牌库/弃牌、场上宝可梦、主动位 HP/能量、双方奖赏卡和我方受伤的变化。`public_history` 可能为空，因此只作为附加标志，不能作为唯一对手信息源。禁止把录像中对手不可见的 option 或隐藏手牌写入输入。

实现位置：

- `src/train/transition_features.py`：差分和上一动作编码；
- `src/train/sequence_data.py`：窗口、前置样本、padding 和终点监督；
- `src/train/sequence_model.py`：SL-0 编码器、单层 GRU 与时间残差；
- `src/train/train_sequence.py`：训练、热启动、验证和 checkpoint。

## 6. GRU 服务器训练

先做冒烟：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 1 --batch-size 8 --window-length 8 \
  --num-workers 0 --max-train-windows 1000 --max-valid-windows 200 \
  --output artifacts/sl1_gru_server_smoke
```

再做长度 16 的正式训练：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 6 --batch-size 32 --window-length 16 \
  --num-workers 4 --learning-rate 3e-4 \
  --output artifacts/sl1_gru_full
```

默认从 `artifacts/sl0_shared_full/best.pt` 热启动。时间残差投影初始化为零，因此初始 policy/value 与 SL-0 一致；随后梯度才逐步启用差分、上一动作和循环状态。每个滑窗只在终点计算监督，避免同一历史样本在一个窗口内重复计权。

断点续训只用于同一次训练被中断的情况。首轮训练已正常完成，不要执行下面命令；否则会继续写入并覆盖 `artifacts/sl1_gru_full/last.pt`：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 8 --resume artifacts/sl1_gru_full/last.pt \
  --output artifacts/sl1_gru_full
```

冻结 test 评估：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_test.json
```

checkpoint 会同时校验原始训练集和序列索引哈希。对比时必须固定相同的 test endpoint 与窗口长度。

### 6.1 首次服务器结果

正式训练参数为长度 16、batch 32、学习率 `3e-4`、seed `20260717`、6 epoch。产物已归档到 `artifacts/sl1_gru_full/`：

- `best.pt`：epoch 1，valid loss `2.0841`、policy top-1 `60.61%`、value MSE `0.8645`；
- `last.pt`：epoch 5，valid loss `2.0905`、policy top-1 `63.09%`、value MSE `0.9419`；
- `reports/sl1_gru_test.json`：使用 best checkpoint 的首次 test，loss `2.0433`、policy top-1 `60.95%`、value MSE `0.8275`。

相对 `SL-0-shared`，首次 test 的总 loss 降低 3.32%，policy top-1 提升 0.68 个百分点，value MSE 降低 7.19%。这构成继续验证的正向证据，但旧报告没有非强制单选、来源分组、非法 top-1 和性能明细，不能单独作为正式晋级结论。

### 6.2 立即执行的详细复评

新版 `src/train/eval_sequence.py` 已补齐与 SL-0 同口径的总体、非强制单选、policy source、非法 top-1 和吞吐指标。分别评估两个 checkpoint，避免“总 loss 最优”和“policy 最优”选择目标不一致。以下命令不会修改 checkpoint：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_best_detailed_test.json

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/last.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_last_detailed_test.json
```

两个命令都应正常结束，并分别生成 JSON。先检查文件存在且 JSON 可解析：

```bash
python -c "import json; [json.load(open(p)) for p in ['reports/sl1_gru_best_detailed_test.json','reports/sl1_gru_last_detailed_test.json']]; print('OK: detailed GRU reports')"
```

再打印最重要的字段：

```bash
python - <<'PY'
import json
for path in [
    "reports/sl1_gru_best_detailed_test.json",
    "reports/sl1_gru_last_detailed_test.json",
]:
    report = json.load(open(path))
    print("\n", path)
    print("overall =", report["overall"])
    print("non_forced =", report["non_forced"])
    print("legality =", report["legality"])
    print("performance =", report["performance"])
PY
```

这段命令只读取报告，不会修改 checkpoint。保留终端输出和两个 JSON。

### 6.3 首轮详细复评停止条件

以 `reports/sl0_shared_test.json` 的非强制 top-1 `57.72%` 为主基线。best/last 至少有一个同时满足：

1. `non_forced.policy_top1 >= 0.5802`，即相对 SL-0 至少提高约 0.3 个百分点；
2. `overall.loss <= 2.1134`；
3. `overall.value_mse` 不明显差于 `0.8916`；
4. `legality.illegal_top1_predictions == 0`；
5. 各主要 `by_policy_source` 没有无法解释的大幅退化。

若两个 checkpoint 都未通过，立即停止第二 seed 和 `SL-0-history` 之后的模型扩张，先分析 loss 权重、checkpoint 选择与历史输入。若至少一个通过，记录首轮候选 checkpoint，并继续下面步骤。离线通过只代表取得继续验证资格，不代表已经晋级。

### 6.4 第二 seed 独立复跑

第二 seed 固定为 `20260721`，输出到全新目录，禁止覆盖首轮 `artifacts/sl1_gru_full/`：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 6 --batch-size 32 --window-length 16 \
  --num-workers 4 --learning-rate 3e-4 --seed 20260721 \
  --output artifacts/sl1_gru_seed20260721
```

训练结束后检查：

```bash
ls -lh artifacts/sl1_gru_seed20260721/
```

至少应看到 `best.pt` 和 `last.pt`。目录名必须是 `sl1_gru_seed20260721`，不能写成首轮目录 `sl1_gru_full`。

训练结束后同样详细评估 best/last：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_seed20260721/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_seed20260721_best_detailed_test.json

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_seed20260721/last.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_seed20260721_last_detailed_test.json
```

不要根据结果再更换 seed。第二 seed 至少一个 checkpoint 也应满足第 6.3 节硬条件，且核心指标相对 SL-0 的方向与首轮一致；否则 GRU 暂不冻结。

## 7. 执行顺序

```text
SL-0-shared 基线
  -> SL-1-gru 服务器全量训练与首次 test ✅
  -> best/last 详细同口径复评
  -> SL-0-history 全量对照
  -> 第二 seed 独立复跑与详细复评
  -> 下载并汇总三模型离线结果
  -> 三模型固定 Arena + Combo + 时延验收
  -> 通过后实现 NumPy 在线运行时
```

不要把双方训练记录简单交错后直接送入 GRU：比赛时看不到对手的隐藏选择。正式晋级仍比较 `SL-0-shared`、`SL-0-history` 和 `SL-1-gru`；GRU checkpoint 通过离线与 Arena 门槛后，才实现并验证 NumPy 在线运行时和提交包。

## 8. 本轮必须下载的结果

完成上述服务器步骤后，下载以下文件，不需要下载 5 GiB JSONL：

```text
reports/sl1_gru_best_detailed_test.json
reports/sl1_gru_last_detailed_test.json
artifacts/sl0_history_full/best.pt
artifacts/sl0_history_full/last.pt
artifacts/sl0_history_full/metrics.jsonl
artifacts/sl0_history_full/run_config.json
reports/sl0_history_test.json
artifacts/sl1_gru_seed20260721/best.pt
artifacts/sl1_gru_seed20260721/last.pt
reports/sl1_gru_seed20260721_best_detailed_test.json
reports/sl1_gru_seed20260721_last_detailed_test.json
```

若训练脚本实际没有生成某个 history 元数据文件，以目录中的真实产物为准，但 checkpoint 与 test JSON 必须保留。下载完成后计算 SHA-256，并与服务器端 `sha256sum` 输出一起保存。

### 8.1 在服务器打包本轮结果

为方便下载，在服务器项目根目录执行：

```bash
tar -czf pokemon-gru-validation-results-v1.tar.gz \
  reports/sl1_gru_best_detailed_test.json \
  reports/sl1_gru_last_detailed_test.json \
  artifacts/sl0_history_full \
  reports/sl0_history_test.json \
  artifacts/sl1_gru_seed20260721/best.pt \
  artifacts/sl1_gru_seed20260721/last.pt \
  reports/sl1_gru_seed20260721_best_detailed_test.json \
  reports/sl1_gru_seed20260721_last_detailed_test.json

sha256sum pokemon-gru-validation-results-v1.tar.gz \
  > pokemon-gru-validation-results-v1.tar.gz.sha256
ls -lh pokemon-gru-validation-results-v1.tar.gz*
```

如果按停止条件提前结束，删除命令中尚不存在的路径后再打包。不要伪造空文件。

### 8.2 回到本地下载

在服务器执行 `exit` 回到本地 PowerShell，或另开一个本地 PowerShell。替换用户名、服务器和路径：

```powershell
scp `
  YOUR_USER@SERVER_HOST:/path/to/work/pokemon-tcg-sl0-sl1-handoff-v3/pokemon-gru-validation-results-v1.tar.gz `
  YOUR_USER@SERVER_HOST:/path/to/work/pokemon-tcg-sl0-sl1-handoff-v3/pokemon-gru-validation-results-v1.tar.gz.sha256 `
  ".\"
```

本地收到后校验：

```powershell
$actual = (Get-FileHash -Algorithm SHA256 ".\pokemon-gru-validation-results-v1.tar.gz").Hash.ToLower()
$expected = (Get-Content ".\pokemon-gru-validation-results-v1.tar.gz.sha256").Split()[0].ToLower()
if ($actual -ne $expected) { throw "SHA-256 校验失败" }
"OK: result archive SHA-256 verified"
```

校验成功后，把结果压缩包和校验文件交回本项目。

## 9. 本指南能够与不能够直接得到的结果

严格照做可以得到：

- 首轮 GRU best/last 的同口径详细 test；
- `SL-0-history` 全量 checkpoint 与详细 test；
- 第二 seed GRU 的完整 checkpoint 与详细 test；
- 是否值得开发在线运行时的离线结论。

严格照做仍不能直接得到：

- 固定 Arena 胜率；
- 在线回退率、整局 p95 延迟；
- NumPy 提交包表现。

这三项依赖尚未实现的 GRU 在线运行时。离线结果通过后，下一份操作指南应先完成 PyTorch/NumPy 一致性测试和代理接入，再运行 Arena；不能把离线 top-1 当作最终实战结论。

## 10. 常见问题与处理

### 10.1 `No such file or directory`

```bash
pwd
ls
```

必须位于 `pokemon-tcg-sl0-sl1-handoff-v3` 根目录。否则重新 `cd`，不要在上级目录运行训练。

### 10.2 `ModuleNotFoundError: No module named 'src'`

训练使用 `python -m src.train.模块名`。测试使用：

```bash
PYTHONPATH=. python tests/测试脚本.py
```

同时确认当前目录正确。

### 10.3 `torch.cuda.is_available()` 为 `False`

常见原因是仍在登录节点、没有申请 GPU，或者装了 CPU 版 PyTorch。先运行：

```bash
nvidia-smi
```

如果该命令失败，联系管理员或先申请 GPU 节点。不要改成 CPU 跑全量训练。

### 10.4 CUDA out of memory

先停止命令并运行 `nvidia-smi`，确认没有遗留进程。GRU 将 batch 从 `32` 降为 `16` 或 `8`；history 从 `256` 降为 `128`，必要时增加 `--grad-accum 2`。必须记录实际参数。

### 10.5 终端断线

若使用 Slurm，先检查作业状态；若使用 `tmux`：

```bash
tmux attach -t pokemon
```

只有进程确实停止且已有合法 `last.pt` 时才使用 `--resume`。不要重新写入首轮 GRU 目录。

### 10.6 `dataset hash mismatch`

数据、manifest 和 checkpoint 不是同一版本。不要关闭哈希检查或手改 JSON；重新解压 V3，确认没有混入旧 V2 文件。

### 10.7 JSON 报告不存在或无法解析

评估没有正常完成。查看 traceback 和日志，修复后重跑同一评估命令。评估是只读操作，可以安全重跑；不要创建空 JSON。

### 10.8 不确定能否继续

先停止，把以下信息交回：

- 最后 50～100 行日志；
- 已生成的 JSON；
- `nvidia-smi` 输出；
- 实际执行的完整命令；
- `ls -lh` 产物目录输出。

不要跳过哈希、测试或停止条件。
