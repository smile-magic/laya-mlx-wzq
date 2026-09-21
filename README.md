# 一着 · Laya-MLX 五子棋

**中文** | [English](README.en.md)

一个在 Apple Silicon Mac 上运行的本地五子棋人机对战游戏。你执黑先手，Laya 执白；浏览器负责棋盘交互，Python 负责规则，MLX 在本机 GPU 上运行真实模型推理。

这个项目沿用 [laya-mlx Snake demo](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) 的思路：**程序提取局面特征 → 模型从候选动作中选择 → 战术规则检查 → 执行落子**。本项目没有对模型进行五子棋专项训练，也不把规则计算出的棋力归因给模型。

## 功能

- 15×15 自由五子棋：黑棋先行，横、竖、斜连续五颗或更多即胜；没有三三、四四、长连禁手。
- 每次 AI 落子真实调用一次 Laya `choice` 决策头，最多比较 6 个合法候选点。
- 展示候选点概率、模型首选、实际落点、推理耗时和战术规则介入。
- 支持悔棋一回合、重开、手数显示、最新落点与获胜连线标识。
- 鼠标、触摸及键盘操作；适配宽窄屏布局。目前游戏界面为中文，文档提供中英两种语言。
- 前端使用原生 HTML/CSS/Canvas，无需 Node.js、构建步骤或 CDN。
- 模型下载后离线游玩；不需要 API key、云端推理或付费模型服务。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 电脑 | Apple Silicon Mac（M 系列芯片） |
| 系统 | 上游声明 macOS 14+；实际支持还取决于当前 MLX wheel 与系统版本 |
| Python | 3.11 或更新版本；建议使用独立虚拟环境 |
| GPU | 本机可访问的 Metal GPU |
| 浏览器 | 较新的 Safari、Chrome、Edge 或其他现代浏览器 |
| 网络 | 首次安装依赖、下载模型时需要；之后可离线 |
| 磁盘 | 多语言 FP16 权重约 0.65 GB，另需依赖和缓存空间，建议预留数 GB |

本项目已在 M4 / 32 GB 内存、macOS 27.0、Python 3.14.7、MLX 0.32.2 上验证。这个记录不是最低内存要求，也不代表其他系统版本均已实测。当前启动路径不支持 Intel Mac、Windows 或 Linux。

## 安装：从零开始

### 1. 获取代码

```bash
git clone git@github.com:smile-magic/laya-mlx-wzq.git
cd laya-mlx-wzq
```

如果没有配置 SSH，也可以通过 HTTPS 克隆同一个仓库：

```bash
git clone https://github.com/smile-magic/laya-mlx-wzq.git
cd laya-mlx-wzq
```

仓库若为私有，需要先取得访问权限，并配置 GitHub SSH 密钥或 HTTPS 凭证。

### 2. 检查 Python 和芯片架构

```bash
uname -m
python3 --version
```

应分别看到 `arm64` 和 Python 3.11+。部分 Mac 的系统 `python3` 仍是 3.9；请换用已安装的新版 Python。若已安装 Homebrew，可用 `brew install python@3.12`，然后用 `python3.12` 替代后文创建环境时的 `python3`。不要使用 Rosetta 下的 x86_64 Python。

### 3. 创建环境并安装依赖

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

依赖文件固定 `laya-mlx==0.1.0`；MLX、tokenizers、NumPy、Hugging Face Hub 等由它安装。无需另外克隆上游源码，也不需要安装 PyTorch 或 Transformers 来运行游戏。

### 4. 下载模型（仅首次）

```bash
HF_HOME="$PWD/.hf-cache" .venv/bin/hf download \
  aac6fef/laya-multilingual-mlx \
  --local-dir models/laya
```

默认使用 [Laya Multilingual MLX](https://huggingface.co/aac6fef/laya-multilingual-mlx)，约 322M 参数，FP16。保留整个下载目录，不能只留下 `model.safetensors`；配置和 tokenizer 同样必需。公开模型通常不需要登录。

模型文件和缓存已被 `.gitignore` 排除，不会提交到仓库。游戏运行时启用 `HF_HUB_OFFLINE=1`，缺少模型时会报错，不会自动联网下载。

### 5. 启动游戏

```bash
./run.sh
```

首次加载会预热 GPU，终端打印“已就绪”后自动打开：

**<http://127.0.0.1:8766>**

也可在 Finder 中双击根目录的 **`启动五子棋.command`**。如果浏览器没有自动打开，手动访问上面的地址即可。

**停止服务：** 回到启动终端，按 `Ctrl+C`。仅关闭网页不会停止 Python 服务或释放模型资源。对局仅保存在网页内存中；刷新、关闭页面会丢失当前对局。

## 日常启动与自定义配置

安装完成后，每次只需要 `cd laya-mlx-wzq`，再运行 `./run.sh`。

```bash
# 不自动打开浏览器
./run.sh --no-browser

# 更换端口
./run.sh --port 8767

# 使用已下载的完整本地模型目录
./run.sh --model /absolute/path/to/laya-multilingual-mlx

# 复用已有 Python 环境（须已安装 laya-mlx）
LAYA_PYTHON=/absolute/path/to/venv/bin/python ./run.sh \
  --model /absolute/path/to/laya-multilingual-mlx

# 不经过启动脚本，直接运行
.venv/bin/python -B server.py --model models/laya --port 8766
```

默认模型路径是**本仓库内的 `models/laya`**，默认 Python 是**本仓库内的 `.venv/bin/python`**，不依赖开发者电脑上的任何目录。`run.sh` 会先切换到仓库根目录，所以通过它传入的相对模型路径也相对此目录解析；直接调用 `server.py` 时，显式相对路径相对当前工作目录解析。`--model` 接受本地目录，不接受远程模型 ID。

## 如何下棋

| 操作 | 行为 |
| --- | --- |
| 点击/轻触空的交叉点 | 你落一颗黑棋，AI 自动应手 |
| 悔棋一回合 | 撤销你与 AI 最近各一手；若仅有你的落子，则只撤销这一手 |
| 新的一局 | 清空棋盘；进行中的对局会先确认 |
| 显示手数 | 在棋子上显示落子顺序 |
| `Tab` 聚焦棋盘，方向键 | 移动键盘落点 |
| `Enter` / 空格 | 在当前键盘位置落子 |

AI 思考时锁定落子、悔棋和重开，避免回合冲突。推理失败后保留你的落子，支持重试或悔棋；不会静默改用随机或纯规则 AI。

## 决策机制与能力边界

```text
你的落子 → 校验棋谱 / 判定终局
                    ↓
       规则分析邻域空点：攻击、防守、棋形
                    ↓
             最多 6 个候选点
                    ↓
      本地 Laya Agent.predict → choice 概率
                    ↓
       短程应手检查 / 棋形质量约束 → 白棋落子
```

规则层先检查已有棋子两格邻域内的所有空点，再筛选最多六个候选。每个候选会检查落子后的对手立即取胜、活四或双重连五威胁；如果己方冲四迫使对手应手，还会检查该应手是否反而制造对手双杀。模型接收**完整文字棋盘、候选后果及棋形质量**，不是棋盘图像；它不生成自然语言推理过程。

- 优先级为：立即连五 → 对方无立即胜着时制造双重连五点 → 通过短程败招检查的落点。
- 提前处理对方活三发展成活四；并非等到对方下一手能连五才防守。
- 普通局面保留同一战术等级中棋形评分至少达到最佳值 65% 的点，再按模型概率选择。这是启发式质量约束，不是最优性证明。
- 如果全部候选均存在短程败招，会明确说明未找到解除全部威胁的走法。
- 若规则改变模型首选，界面显示首选、实际落点和介入原因。
- 候选概率是**选择倾向，不是胜率**。棋形标签和规则说明由代码生成。
- 短程检查覆盖有限的强制威胁，不是完整博弈树搜索；对更长的杀法和已无法防守的局面仍可能失败。

这是一个可玩、可检查的模型决策演示，不是专业五子棋引擎。本项目未对权重进行训练；上游 Snake demo 也使用现成权重而非 Snake 专用训练。关于如何进一步微调，请阅读[完整调研文档](docs/Laya游戏决策与五子棋训练调研.md)（中文）。

“推理耗时”包括同步 `Agent.predict`、分词与结果处理；“整步耗时”还包括规则分析，不包含首次模型加载、HTTP 传输和浏览器绘图。

## 本地运行与隐私

服务只监听 `127.0.0.1`，供这台电脑的浏览器使用，不对局域网开放。代码不上传棋谱，不保存对局文件，不使用外部字体或 CDN。终端输出普通 HTTP 请求日志。多个页面共用一个模型，推理串行执行；并发请求可能收到“模型正在处理另一局”，稍后重试即可。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 Python 环境 | 先创建 `.venv` 并安装依赖；或用 `LAYA_PYTHON` 指定可执行文件的绝对路径 |
| Python 版本太低 / 找不到 MLX wheel | 确认原生 arm64 Python 3.11+、macOS 版本满足当前 wheel 要求，并升级 pip |
| 模型不存在 / 配置缺失 | 完整执行下载命令；默认目录为 `models/laya`，或显式指定 `--model` |
| `Using SOCKS proxy ... socksio ...` | 执行 `.venv/bin/python -m pip install 'httpx[socks]'`，然后重试下载 |
| `No Metal device available` | 在正常 macOS 本机终端运行；沙箱、容器或虚拟机可能无法访问 GPU |
| `Address already in use` | 停止之前启动的服务，或使用 `./run.sh --port 8767` |
| `Permission denied` 执行脚本失败 | 执行 `chmod +x run.sh 启动五子棋.command`；也可直接用 Python 启动 |
| 下载连接失败 | 检查当前网络和代理后重试；模型完整下载前无法离线启动 |
| 页面显示连接异常 | 确认启动终端仍在运行；恢复服务后重试，或刷新以开新局 |
| AI 下得不够强 | 当前是未专项训练的特征辅助决策；可改进候选/搜索或按调研路线微调 |

## 项目结构

```text
game.py                  规则、局部棋形特征、候选与战术约束
server.py                本地 HTTP 服务与真实 Laya 推理
run.sh                   通用终端启动入口
启动五子棋.command        macOS 双击启动入口
requirements.txt         Python 依赖
web/                     HTML / CSS / JavaScript / Canvas 界面
tests/test_game.py       不依赖模型的规则与战术单元测试
docs/                    游戏决策与训练调研
README.md                中文说明
README.en.md             English documentation
models/laya/             下载后的模型（不提交）
.venv/                   本地 Python 环境（不提交）
```

## 开发与测试

规则测试不需要 MLX、模型或 GPU：

```bash
python3 -B -m unittest discover -s tests -v
```

默认运行 21 项不依赖模型的测试，另有 2 项可选 GPU 测试。覆盖规则边界、诊断中的 32 个战术局面、强制应手反杀、提前防活四、原始概率保留及棋盘编码。即使把最高模型概率故意给差点，也必须执行符合战术约束的点。

运行真实模型和输入完整性回归：

```bash
LAYA_TEST_MODEL="$PWD/models/laya" .venv/bin/python -B -m unittest discover -s tests -v
```

修复后 32 个战术局面的实际落子全部通过，模型原始首选为 30/32；这不是实战胜率。关于 Snake 安全层借鉴、修复细节和测试边界，见[决策修复说明](docs/DECISION_FIX.md)。

临时启动服务时可以用 `./run.sh --no-browser --port 8767`。验证完毕必须按 `Ctrl+C` 停止服务，并删除测试棋谱、截图、日志和临时环境；不要保留测试进程。

## 上游与延伸阅读

- [Laya-MLX](https://github.com/mizorewww/laya-mlx)：原生 MLX 推理运行时，Apache-2.0。
- [默认模型权重](https://huggingface.co/aac6fef/laya-multilingual-mlx)：模型与其基础模型的许可、来源以对应模型卡为准。
- [上游 Laya](https://github.com/NandhaKishorM/laya)：结构化决策模型及训练实现。
- [Laya 游戏决策与五子棋训练调研](docs/Laya游戏决策与五子棋训练调研.md)：Snake 是否训练、RLCD/监督/自博弈的区别、五子棋数据设计、硬件与评测路线；包含核查日期和一手来源链接。

本仓库是独立的五子棋应用，不是上述项目的官方游戏发行版；仓库不包含上游模型权重。
