# Laya 游戏决策与五子棋训练调研

> 核查日期：2026-09-21
>
> 目标：回答“`laya-mlx` 的模型是否专门训练过贪吃蛇、如果要做五子棋人机对战是否需要训练、应该怎样训练”，并把已被源代码证明的事实与面向五子棋的工程建议分开。
>
> 资料范围：`mizorewww/laya-mlx` 当前公开仓库、其声明的上游 `NandhaKishorM/laya`、Convai Innovations 的 Hugging Face 模型卡、上游公开微调 notebook 和 `LocalLLaMA/typed-decisions` 数据集卡。没有运行训练，没有启动服务，也没有下载额外训练数据。

## 结论先行

1. **当前 `laya-mlx` 没有把 Laya 模型训练成贪吃蛇模型。** Snake 是一个推理演示：现成的 Laya checkpoint 接收宿主程序计算好的文字特征，返回 `UP/DOWN/LEFT/RIGHT` 的 `choice` 概率。Snake 规则、合法性、Hamiltonian cycle 安全规划和最终动作接管都在 Python 代码里完成。
2. **“模型没有专门学过五子棋”指的是公开资料中未发现作者针对五子棋训练或棋力验证的证据。** 这不意味着预训练语料完全不含五子棋知识，也不代表模型绝对不会给出相关落子建议；它只表示没有可核查的五子棋专用数据、参数更新和棋力评测。Laya 的 backbone 来自通用语言模型，Laya 还经过 typed decision/RLCD 训练，但这些已公开的训练目标没有被证明是在 15×15 棋盘上优化胜负策略。
3. **做一个能玩的五子棋 demo，不必先训练。** 可以仿照 Snake，把棋盘编码为有限的文字特征，生成少量合法候选点，让 Laya 做候选点排序，再由宿主程序执行落子、胜负判定和合法性保护。这证明的是“模型参与决策”的产品形态，棋力主要来自特征工程、候选点生成和安全/搜索逻辑。
4. **做一个棋力可靠的人机对战，需要五子棋专用增强。** 最稳妥的第一版是用规则引擎或搜索引擎生成教师分布，再按 Laya 的 `choice` 输入格式做监督/RLCD 微调；若只用对局最终输赢做自博弈强化学习，则需要额外实现游戏环境和序列策略优化，上游公开 notebook 并没有提供可直接运行的五子棋 RL 命令。
5. **`laya-mlx` 不是训练框架。** 其 README 明确把 RLCD training/fine-tuning 留给上游，转换命令是参数名/数据类型转换，不是再训练。公开的上游微调流程基于 PyTorch、Transformers、CUDA 和 `torchrun`，示例硬件是 Kaggle 2×T4；训练完成后才考虑转换成 MLX，在 Apple Silicon 上推理。

## 1. 这几个“训练”概念要分开

### 1.1 通用 backbone 预训练

Laya 不是从零开始学字词。模型卡把英文 checkpoint 描述为 ModernBERT-large backbone（421M 总参数），多语言 checkpoint 为 mmBERT-base（322M），typed-decisions checkpoint 使用 ModernBERT-large（421M）。Laya 的结构是在 backbone 上加 decision Transformer、option-marker scorer 和 act/escalate head。

上游模型卡公开说明的是 backbone/decision head 及 RLCD 微调方式，并没有在该项目页面给出完整的 backbone 预训练数据和重现命令。因此这里不能把“它懂一些自然语言”误说成“它已经具备可验证的五子棋棋力”。自然语言预训练可能提供某些相关知识，但棋力还需要棋盘状态、合法动作、胜负目标和相应的参数训练或搜索算法，并必须用独立对局评测验证。

来源：

- [Laya 模型卡的模型家族、架构与训练说明](https://huggingface.co/convaiinnovations/laya/blob/main/README.md)
- [上游 `laya/common.py`：decision model、option markers 和 `choice/score/noul` 类型](https://raw.githubusercontent.com/NandhaKishorM/laya/main/laya/common.py)
- [`laya-mlx` README：支持的三个 checkpoint 及其用途](https://github.com/mizorewww/laya-mlx/blob/main/README.md)

### 1.2 Laya 的 typed decision/RLCD 训练

Laya 的输出不是逐 token 生成文本，而是针对请求中给定的选项输出分布。三个基本类型是：

- `choice`：在给定选项中选一个，并返回每个选项的概率；
- `score`：在有序等级中输出分布和期望等级；
- `noul`：对命题为真的概率。

上游模型卡将 RLCD（Reinforcement Learning for Calibrated Decisions）描述为：对 logits 添加零均值高斯探索噪声，用严格 proper scoring rule 计算奖励（log、spherical；有序 score 还加入 ranked probability score），用 REINFORCE 和 group-mean baseline 更新。这个目标鼓励“概率分布诚实”，不是直接等价于“学会一款游戏”。

typed-decisions checkpoint 是一个后续 specialist：模型卡明确说它在四类 synthetic workflow 上微调——agent-trace observability、customer service、invoice processing、security incidents；官方测试表中，未微调的 `laya` 和 `laya-multilingual` 在这个 benchmark 上的准确率分别为 0.362 和 0.342，而 typed-decisions checkpoint 为 0.766。这个事实恰好说明：Laya 可以从通用决策能力进一步适配领域，但适配领域必须有该领域数据。

多语言 checkpoint 的模型卡还明确写出一组训练记录：RLCD、15,987 updates、4 epochs、约 4.97 小时。这是该 checkpoint 的 Laya 决策训练记录，不是游戏训练；英文根 checkpoint 的卡片则描述了 email triage、conversation trajectory（TD(λ=1.0)）和按选项数校准等 fine-tuned 行为。公开资料没有声称其中包含 Snake 或 Gomoku 对局。

来源：

- [Laya 模型卡的 Architecture/Training 段落](https://huggingface.co/convaiinnovations/laya/blob/main/README.md)
- [typed-decisions checkpoint 的任务范围、基线与限制](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/main/README.md)
- [multilingual checkpoint 的架构、RLCD 训练记录与限制](https://huggingface.co/convaiinnovations/laya-multilingual)
- [上游 `proper_reward` 实现](https://raw.githubusercontent.com/NandhaKishorM/laya/main/laya/common.py)

### 1.3 MLX 移植/转换不是训练

`laya-mlx` README 说明它是独立的 Apple Silicon MLX runtime；`convert` 导出包含 `model.safetensors`、配置、tokenizer 和 `mlx_config.json`，并明确称这是 parameter-name/dtype conversion，**不是 quantization 或 retraining**。模型卡也说明 `laya-typed-decisions-mlx` 是源 checkpoint 的 native MLX FP16 conversion，保留原始权重和问题格式。

所以流程的边界是：

```text
上游 PyTorch checkpoint
        │ 训练/微调（不是 laya-mlx 完成）
        ▼
五子棋专用 PyTorch checkpoint
        │ 参数映射/类型转换（laya-mlx convert，若架构兼容）
        ▼
Apple Silicon 上的 MLX 推理
```

来源：

- [`laya-mlx` README 的 checkpoint、训练归属和转换说明](https://github.com/mizorewww/laya-mlx/blob/main/README.md)
- [`laya-typed-decisions-mlx` 模型卡的 provenance/limits](https://huggingface.co/aac6fef/laya-typed-decisions-mlx)

## 2. 当前 Snake 到底做了什么

### 2.1 事实：没有 Snake 专用训练

项目自己的 Snake 文档把它称为 **feature-assisted neural decision demo**，并直接写明使用“existing Laya checkpoint without Snake training”。Snake benchmark 文档也写明 checkpoint was not trained on Snake here。仓库中没有 Snake dataset、Snake loss、Snake optimizer 或 Snake fine-tune 脚本；测试覆盖的是规则、推理、回放和 benchmark。

来源：

- [`docs/SNAKE_DEMO.md`：What the AI does](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md)
- [`docs/SNAKE_BENCHMARKS.md`：checkpoint 未在 Snake 上训练](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_BENCHMARKS.md)
- [`laya_mlx/model.py`：文件头明确标为 inference only](https://github.com/mizorewww/laya-mlx/blob/main/laya_mlx/model.py)

### 2.2 宿主程序先做规划，再把文字交给模型

每一步的实际链路如下：

1. `SnakeGame` 保存身体、食物、得分和 tick；计算墙、身体、反向走法等合法性。
2. 它预先构造 Hamiltonian cycle，并用 cycle index 判定是否会越过尾部或跳过食物；同时计算当前空格连通性。
3. `LayaPolicy` 把每个方向描述成“Blocked/Unsafe/Safe/Best route”等文字，生成一个 `choice` 问题；另生成两个 `noul` 问题，询问是否有安全路线、食物是否可达。
4. Laya 只是在这些文字描述上输出概率。它不是从像素或原始二维数组中自行发现碰撞和路线。
5. 默认 safety shield 先看模型的 top-1；如果该方向不在安全集合内，就执行安全集合中概率最高的方向。界面保留原始概率并记录 `intervened`。

直接阅读代码可以看到这些边界：

- [`snake/game.py`](https://github.com/mizorewww/laya-mlx/blob/main/laya_mlx/snake/game.py) 实现规则、Hamiltonian cycle、`safe` 和食物可达性；
- [`snake/policy.py`](https://github.com/mizorewww/laya-mlx/blob/main/laya_mlx/snake/policy.py) 生成 planner features、调用 `Agent.predict`、实施 shield；
- [`docs/SNAKE_DEMO.md`](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) 解释三问题批量推理和指标含义。

### 2.3 “关闭 shield”也不是原始棋盘智能

`--unassisted` 只关闭执行阶段的安全限制，让模型的 raw top-1 直接落子；它仍然接收 planner 生成的方向描述和安全信息。因此 raw top-1 结果也不能证明 Laya 已经学会从任意 Snake 棋盘规划。

这也是为什么项目的零死亡 benchmark 不能被解释成“模型自己学会了 Snake”：安全不变量由确定性代码维护，模型的作用是从候选描述中给出概率和选择。

## 3. 上游公开训练流程实际支持什么

### 3.1 公开的 notebook 是 typed-decisions 专项微调

上游仓库公开 notebook 的标题是 `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`，目标是把 `convaiinnovations/laya` 微调到 `LocalLLaMA/typed-decisions` 的 1,200 个训练 case、6,000 个 typed decisions；它不是 Snake 或 Gomoku 训练脚本。

Notebook 的数据预处理从每行读取 `state`、`questions`、`gold`：

- `state` 是 JSON 字符串，作为模型状态；
- `questions` 给出类型、instructions 和 criteria；
- `gold` 给出 label 及完整 probabilities；
- 预处理为 token ids、option marker 位置、question type、soft target distribution 和离散 label。

这是一个很适合五子棋改造的接口，因为五子棋教师也可以给出候选落点的**软分布**，而不仅是一个硬标签。

来源：[上游微调 notebook 的环境、数据读取和预处理单元](https://raw.githubusercontent.com/NandhaKishorM/laya/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)；GitHub 页面版本见[该 notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)。

### 3.2 notebook 的训练目标和超参数

公开代码使用 PyTorch/Transformers/CUDA：

- `torchrun --standalone --nproc_per_node=2`，DDP 使用两张 T4；
- gradient checkpointing，`max_tokens_per_batch=4096`；
- 4 epochs；每 GPU micro-batch 8，gradient accumulation 4，effective batch 64；
- group size 4 的 logits 探索；噪声 sigma 从 0.4 退火到 0.1；
- encoder 学习率 `2.5e-5`，head 学习率 `1.0e-4`，AdamW、weight decay 0.01、cosine schedule；
- loss 同时包含 RLCD policy-gradient loss 和对 teacher distribution 的 soft cross-entropy；
- 训练后在 notebook 中拟合温度并保存 checkpoint、encoder、tokenizer、配置。

这些数值是该 notebook 的**typed-decisions 示例配置**，不是五子棋的已验证超参数。五子棋的选项数、棋局分布、教师质量和状态长度都不同，不能直接照搬后宣称棋力。

来源：[notebook 的 DDP 训练脚本](https://raw.githubusercontent.com/NandhaKishorM/laya/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)；奖励函数的独立实现见[`laya/common.py`](https://raw.githubusercontent.com/NandhaKishorM/laya/main/laya/common.py)。

### 3.3 评估与校准

该 notebook 在官方 test split 上计算 accuracy、soft accuracy、Brier、KL、total variation、ECE、score MAE 和 latency；模型卡报告 typed-decisions checkpoint 的 400 test cases/2,000 decisions。

五子棋不能只看“赢了多少局”：若模型输出概率并用 confidence 做决策，还要保留 Brier/ECE；对棋步排序要看 top-1、top-k 和 teacher 分布相似度；对完整对局要看胜率、Elo、非法落子率和平均对局长度。

注意 notebook 的 calibration 示例从训练预处理结果抽取 `all_items[::15][:400]`。这不是一个足够严谨的五子棋校准方案，因为它仍来自训练数据。五子棋应该预留独立 calibration split；温度只能在训练完成后用这份未参与梯度更新的数据拟合。

### 3.4 公开流程的硬件/时间限制

typed-decisions 模型卡声称该 notebook 在 Kaggle 免费 2×T4 上约需 4–5 小时；README 也给出约 4–5 小时、4 epochs 的说法。notebook 某个 markdown 单元另写了“4 to 6 minutes total”，与上述时长和训练规模不一致。这里应按**小时级预算**规划，并把实际运行时间当作待测量项，不能把 notebook 的分钟文字当成承诺。

Apple Silicon 上的 `laya-mlx` 推理可以用于部署五子棋，但该项目没有公开 MLX 训练实现。若要训练，应使用上游 PyTorch 路线或自行验证 MLX 训练方案；不能把 `uv run laya-mlx convert` 当成训练命令。

## 4. 五子棋应该先不训练还是直接训练

### 4.1 第一阶段：不训练也能做可玩的 MVP

可以把五子棋 demo 做成和 Snake 同一类的“特征辅助决策”：

1. 宿主程序维护 15×15（或 9×9/13×13）棋盘、轮次、胜负和禁着规则。
2. 生成合法候选点，例如棋子邻域、立即成五点、阻挡对方成五点、活三/冲四相关点，再裁剪到小于约 20 个候选项。
3. 将每个候选点写成带坐标和局部特征的 criteria，例如“`H8`: 自己形成活三；对方下一手可成五；中心距离 0”。
4. 用 `choice` 询问“在这些合法候选点中哪一步最值得下”，执行概率最高的合法候选点。
5. 主程序在模型返回后再次做合法性和胜负检查；模型提出非法点时拒绝执行并回退到规则/搜索。

这个版本可以验证 UI、回合、候选解释、概率展示和本地 MLX 延迟，但不应宣称 Laya 已经具备专业棋力。由于模型卡建议 `choice` 选项控制在约 20 个以内，不能把 225 个全盘交叉点一次性塞进 question；候选点生成是必要的工程层。

### 4.2 第二阶段：用搜索/规则教师做五子棋专用微调

如果目标是“人机对战体验稳定”，建议先做离线教师数据，再微调，而不是一开始做纯胜负 RL：

1. 写一个无模型的规则引擎：合法落子、成五、禁手（如需要）、终局和棋盘规范化。
2. 选定教师：minimax/alpha-beta、MCTS、现成五子棋 engine，或多种深度的 ensemble。教师必须能对每个候选点给出访问次数、胜率或排序分布。
3. 从多种开局、棋力、先后手、候选生成器和随机种子采样整局，把每个中间局面保存为一个训练样本；终局标签和教师分布都要保存。
4. 训练输入以一个局面对应一个 `choice` 问题为主；可另外加入 `noul`（“此步是否立即获胜/是否必须防守”）和 `score`（局面优势等级），但先确保 `choice` 的主目标闭环。
5. 用上游 notebook 的序列构建和 decision head 训练思路改造 PyTorch pipeline，保留候选 marker、soft target 和 question type；不要直接声称上游 notebook 对自定义五子棋数据开箱即用。
6. 在独立 calibration split 上拟合温度；把训练得到的 PyTorch checkpoint 转成 MLX 后，逐层/逐样本验证 logits、argmax、概率误差和输出格式。

这里的教师分布比单一“最佳落点”更有价值：同一局面可能有多个等价好点，soft target 可以减少把任意一个等价点当成唯一真理的噪声。若只有硬标签，也可以先用交叉熵建立基线，但仍要单独评估概率校准。

### 4.3 第三阶段：只有在需要时才做自博弈 RL

上游 RLCD 的公开实现是带 gold distribution 的离线 typed decision 训练。它不是一个包含棋盘环境、轮次回报、对手采样和 credit assignment 的五子棋自博弈框架。

如果不想使用搜索教师，而是用对局输赢训练，需要自行实现：

- 环境状态、动作 mask、终局奖励和 draw 规则；
- 两侧策略采样及探索；
- 按落子序列回传结果的 policy-gradient/actor-critic 或其他自博弈算法；
- 对手池、开局随机化和防止策略坍缩的评测。

这条路在研究上可行，但不能把 `proper_reward` 直接当成完整的五子棋胜负奖励。实际工程上，先用搜索教师蒸馏，再用少量自博弈调整，通常更容易诊断，也更容易保证非法动作率为零。

## 5. 推荐的数据格式

### 5.1 上游 notebook 实际读取的外层格式

上游 notebook 读取的数据行至少要能提供 `state`、`questions` 和 `gold`；代码中它们是 JSON 字符串，解析后再构造 token ids、markers、question type、target distribution 和 label。五子棋可沿用这个形状：

```json
{
  "id": "game-0042-ply-5",
  "workflow": "gomoku",
  "state": "{\"size\":15,\"columns\":\"A-O\",\"to_move\":\"black\",\"board\":[\"...............\",\"...............\",\"...............\",\"...............\",\"...............\",\"...............\",\".......B.......\",\"......BW.......\",\".......W.......\",\"...............\",\"...............\",\"...............\",\"...............\",\"...............\",\"...............\"],\"last_move\":\"H8\"}",
  "questions": "{\"move\":{\"type\":\"choice\",\"instructions\":\"选择当前最强的合法落点\",\"criteria\":{\"G9\":\"连接左侧棋形；合法空点\",\"I9\":\"靠近右侧棋形；合法空点\",\"H10\":\"向下延伸；合法空点\"}}}",
  "gold": "{\"move\":{\"label\":\"G9\",\"probabilities\":{\"G9\":0.60,\"I9\":0.25,\"H10\":0.15}}}",
  "source": "mcts-depth-8",
  "game_id": "game-0042",
  "seed": 9182
}
```

示例采用 A–O 的 15×15 坐标：`H8` 是上一手白棋，当前轮到黑棋；棋盘中黑白各两子，`G9`、`I9`、`H10` 都是空的合法候选点。实际送入上游 notebook 时，`state`、`questions` 和 `gold` 仍按代码要求序列化为 JSON 字符串。

上面是**建议的五子棋字段示例**，不是仓库已经提供的五子棋 schema。关键要求是：`state` 只包含模型实际允许看到的状态；`questions.criteria` 的候选标签与 `gold.probabilities` 使用完全相同的键；候选点顺序和坐标规范要固定且可审计；`gold` 的概率分布要来自可复现教师或人工标注。

### 5.2 状态表示建议

不要只给一段含糊的自然语言棋盘。至少包括：

- 棋盘尺寸、当前执子方、上一手；
- 行列坐标约定，以及用固定字符/行列编码的全盘状态；
- 候选点列表；
- 每个候选点的合法性、立即成五/阻挡成五和局部棋型；教师统计默认只写入 `gold`、审计日志和评测文件，不作为部署时的模型输入；
- 是否存在禁手或特殊规则。

只有当线上部署也会对每个请求运行同等搜索，并且产品明确要让 Laya 重排搜索结果时，才可以把同等搜索统计作为输入；这测试的是“模型重排搜索候选”的能力，不是模型独立理解棋盘和承担棋力。推荐同时保留一个不带教师统计和模型特征的“原始状态”版本，用来检测特征泄漏：若只给教师直接算出的“最佳/安全”标记，模型可能只是复述标签，不能证明学会局面。

### 5.3 候选点和坐标泄漏

候选集应包含强点、次强点、看似合理但会输的点和随机合法点；只收录最佳点会让训练准确率虚高。候选点可以每局随机打乱顺序，并在验证时做坐标镜像/旋转，防止模型把“第三个 option”或固定坐标当作答案捷径。

若最终产品要让模型直接从 225 个位置选点，需要重新设计 head/token budget 或使用分阶段候选选择；这不属于当前 Laya checkpoint 已验证的能力。

## 6. 监督学习、RLCD 和自博弈的选择

| 路线 | 需要的标签/信号 | 当前项目是否直接支持 | 适合五子棋的用途 |
|---|---|---|---|
| 规则/搜索推理，不训练 | 合法性和启发式/搜索 | `laya-mlx` 可做推理，但搜索需自行写 | 最快做 MVP；棋力由搜索/规则承担 |
| 硬标签监督 | 每局面一个最佳候选 | 上游代码可改造，非现成命令 | 建立第一条可训练基线 |
| 软标签蒸馏 | 候选点概率/访问分布 | 与上游 notebook 的 target 形式最接近 | 推荐的第一版五子棋专用模型 |
| RLCD | gold distribution + proper reward | 上游 notebook 有 typed-decisions 示例 | 让分布和置信度更合理；仍需五子棋数据 |
| 自博弈 RL | 环境终局奖励和轨迹 | 当前仓库没有 | 研究型增强；需新建训练系统 |

建议顺序：规则引擎和搜索 baseline → 软标签 choice 蒸馏 → 独立校准 → 再评估是否值得自博弈。这样每一步都能回答“变强来自模型、教师、搜索还是 shield”。

## 7. 防止数据泄漏和错误评估

五子棋最容易出现的泄漏不是文件重复，而是同一局棋的相邻局面同时出现在 train/test。应遵循以下拆分：

1. **按完整 `game_id` 拆分**，同一盘棋的所有 ply 只能进入一个 split。
2. 对生成式数据按开局模板、随机种子、教师版本和对手策略分组拆分；相同开局的镜像/旋转版本也应放在同一组或做严格去重。
3. 预先固定 test 的教师版本和规则版本，训练过程中不读取 test 的概率、胜负或校准温度。
4. calibration 使用独立局面，不能从训练样本抽取；最终报告同时给未校准和校准结果。
5. 至少保留四组测试：随机/启发式对手、搜索教师的浅层/深层、未见过的开局族、坐标镜像/旋转后的局面。
6. 对模型比较时固定候选生成器和候选集；另做“全局搜索 + 模型排序”的对照，避免把候选生成器的棋力归因给 Laya。
7. 用严格的动作 mask 检查非法落子；非法率必须单列，不能让宿主回退后把结果算作模型正确。

`LocalLLaMA/typed-decisions` 数据集卡本身也强调 train/test 是独立生成、case id 和 state 不相交，并提醒该 benchmark 的 gold 是 teacher 采样均值，衡量的是对 teacher 的一致性而不是绝对真理。五子棋若用 engine teacher，同样应把结果称为“对教师/基线的评测”，不能自动等价为真实棋力。

来源：[typed-decisions 数据集卡的 schema、生成方法、拆分和评分含义](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/main/README.md)。

## 8. 硬件、成本和部署边界

- **本地 M 系列 Mac**：适合用 `laya-mlx` 做推理和五子棋交互；模型权重转换后在 MLX 上运行。M 系列上训练五子棋不是当前项目公开支持的路径。
- **训练**：可按上游公开 notebook 的 2×NVIDIA T4、PyTorch、CUDA、DDP 作为起点；官方模型卡给 typed-decisions 示例的时间是约 4–5 小时，但自定义五子棋数据和候选长度会改变显存与时长。
- **模型规模**：421M ModernBERT-large 或 322M mmBERT-base；训练时需要保存原始 checkpoint、tokenizer、配置、训练日志、teacher 版本和数据版本，不能只保存最终权重。
- **转换风险**：若五子棋只改变输入数据而保留同一 decision head 架构，才有机会复用现有参数映射。若要改变输出 head、棋盘编码、上下文长度或动作结构，必须把转换和数值验证当成单独工作，不能假定 `convert` 自动兼容。
- **预算假设**：先用几千至数万局面做小规模试验，测显存、吞吐、验证集趋势，再决定云 GPU 时长。不要在没有 baseline、泄漏检查和独立 test 的情况下扩张训练集。

## 9. 可执行路线和验收条件

### 阶段 A：五子棋规则和可玩性

- 完成棋盘、落子、胜负、重开、悔棋和非法点检查；
- 实现候选点生成，并记录候选点生成耗时和数量；
- 接入原始 `laya-mlx`，显示 choice 概率、执行点和回退原因；
- 验收：随机局面与随机动作测试无规则错误；模型不可使程序落非法点；可完成完整对局。

### 阶段 B：非训练基线

- 加入中心/连子/活三/冲四等启发式和一个可复现的搜索 baseline；
- 分别测 `Laya only`、`heuristic only`、`search only`、`Laya + safety/search fallback`；
- 验收：报告非法率、对随机和启发式对手胜率、p50/p95 推理延迟，而不是只展示一局演示。

### 阶段 C：教师数据和专用微调

- 固定坐标规范、规则版本、候选生成器、教师版本和随机种子；
- 生成按整局拆分的数据和独立 calibration/test；
- 先训练 `choice` 软标签 specialist，再按验证集决定是否加入 `noul/score`；
- 用上游 PyTorch notebook 的公开逻辑改造训练，不把它称为已有五子棋命令；
- 验收：相对未训练 Laya 和搜索 baseline 的独立对局胜率提升，候选打乱/镜像后性能不崩，概率 ECE/Brier 有记录。

### 阶段 D：MLX 部署

- 转换到 MLX 后做 checkpoint 文件清单、参数 shape、logit、argmax、概率误差和多次重复推理验证；
- 仅在验证通过后替换 demo 使用的 checkpoint；
- 验收：同一状态在 PyTorch 与 MLX 的选择一致率达到预先设定阈值，延迟和内存也单独记录。

## 10. 对本次五子棋主任务的直接建议

当前最合理的产品定位是“Laya 参与五子棋候选决策，宿主程序保证规则，搜索/启发式提供棋力下限”。可以先把游戏做出来并留下 `--baseline`、`--raw`、`--shield` 三种可比较模式；不要把第一版的对局结果描述为 Laya 已经通过五子棋训练。

如果后续确实要训练，优先训练一个候选点 `choice` specialist：候选数控制在约 20 个以内，教师提供软分布，按整局拆分，独立校准和测试；保持 Laya 的输入/输出结构不变，这样最有希望沿用上游 decision head 和后续 MLX 转换。自博弈 RL 应作为第二阶段研究任务，因为它不是当前仓库提供的能力。

## 一手来源索引

1. [`mizorewww/laya-mlx` README](https://github.com/mizorewww/laya-mlx/blob/main/README.md)
2. [`laya-mlx` Snake demo 文档](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md)
3. [`laya-mlx` Snake policy 实现](https://github.com/mizorewww/laya-mlx/blob/main/laya_mlx/snake/policy.py)
4. [`laya-mlx` Snake game/安全规划实现](https://github.com/mizorewww/laya-mlx/blob/main/laya_mlx/snake/game.py)
5. [`NandhaKishorM/laya` README](https://github.com/NandhaKishorM/laya/blob/main/README.md)
6. [Laya Hugging Face 模型卡](https://huggingface.co/convaiinnovations/laya/blob/main/README.md)
7. [Laya typed-decisions 模型卡](https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/main/README.md)
8. [Laya multilingual 模型卡](https://huggingface.co/convaiinnovations/laya-multilingual)
9. [上游 RLCD/decision head 源码 `laya/common.py`](https://raw.githubusercontent.com/NandhaKishorM/laya/main/laya/common.py)
10. [上游 2×T4 typed-decisions 微调 notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
11. [`LocalLLaMA/typed-decisions` 数据集卡](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/main/README.md)
