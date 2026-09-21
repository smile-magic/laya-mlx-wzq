# Yizhao · Laya-MLX Gomoku

[中文](README.md) | **English**

A local human-versus-AI Gomoku game for Apple Silicon Macs. You play Black and move first; Laya plays White. The browser renders the board, Python implements the rules, and MLX runs real model inference on your Mac's GPU.

The design follows the [laya-mlx Snake demo](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md): **extract features in code → ask the model to choose a candidate → apply tactical constraints → play the move**. This project does not fine-tune Laya on Gomoku, and does not attribute the rules engine's strategy to the model.

## Features

- 15×15 freestyle Gomoku: Black moves first; five or more consecutive stones horizontally, vertically, or diagonally wins. No double-three, double-four, or overline restrictions.
- A real Laya `choice` inference call on every AI move, comparing up to six legal candidates.
- Live candidate probabilities, model preference, executed move, inference latency, and visible tactical interventions.
- Undo a round, restart, show move numbers, and highlight the latest move and winning stones.
- Mouse, touch, and keyboard input, with layouts for wide and narrow screens. **The game UI is currently Chinese; documentation is bilingual.**
- Plain HTML/CSS/Canvas frontend: no Node.js, build step, or CDN required.
- Offline play after downloading the model; no API key, cloud inference, or paid model service needed.

## Requirements

| Component | Requirement |
| --- | --- |
| Computer | Apple Silicon Mac, with an M-series chip |
| macOS | Upstream specifies macOS 14+; usable versions also depend on the available MLX wheels |
| Python | Python 3.11 or newer; a virtual environment is recommended |
| GPU | A locally accessible Metal GPU |
| Browser | A recent Safari, Chrome, Edge, or another modern browser |
| Network | Required for the initial dependency and model downloads; optional afterward |
| Storage | Approximately 0.65 GB for the multilingual FP16 weights, plus dependencies and caches; allow several GB |

The application has been verified on an M4 Mac with 32 GB RAM, macOS 27.0, Python 3.14.7, and MLX 0.32.2. This is a tested configuration, not a minimum RAM requirement or a claim that every other macOS version was tested. The current runtime path does not support Intel Macs, Windows, or Linux.

## Installation from scratch

### 1. Clone the repository

```bash
git clone git@github.com:smile-magic/laya-mlx-wzq.git
cd laya-mlx-wzq
```

If SSH is not configured, use HTTPS instead:

```bash
git clone https://github.com/smile-magic/laya-mlx-wzq.git
cd laya-mlx-wzq
```

A private repository requires access permission and working GitHub SSH keys or HTTPS credentials.

### 2. Check your Python version and architecture

```bash
uname -m
python3 --version
```

These should report `arm64` and Python 3.11+. Some Macs still resolve `python3` to the system Python 3.9; use a newer installation instead. If you already have Homebrew, one option is `brew install python@3.12`, then use `python3.12` in place of `python3` when creating the virtual environment below. Avoid an x86_64 Python running under Rosetta.

### 3. Create the environment and install dependencies

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The requirements pin `laya-mlx==0.1.0`, which installs MLX, tokenizers, NumPy, Hugging Face Hub, and its other dependencies. You do not need a separate upstream checkout, PyTorch, or Transformers to run this game.

### 4. Download the model once

```bash
HF_HOME="$PWD/.hf-cache" .venv/bin/hf download \
  aac6fef/laya-multilingual-mlx \
  --local-dir models/laya
```

The default checkpoint is [Laya Multilingual MLX](https://huggingface.co/aac6fef/laya-multilingual-mlx), approximately 322M parameters in FP16. Keep the complete directory, including configuration and tokenizer files, not just `model.safetensors`. The public checkpoint normally does not require login.

Model weights and caches are excluded by `.gitignore`. The game sets `HF_HUB_OFFLINE=1`; if the checkpoint is missing, it reports an error instead of downloading anything during play.

### 5. Start the game

```bash
./run.sh
```

The server loads the model, warms up the GPU, and opens your browser after printing that it is ready:

**<http://127.0.0.1:8766>**

On macOS, you can also double-click **`启动五子棋.command`** in Finder after installation. If the browser does not open automatically, visit the address above yourself.

**To stop:** press `Ctrl+C` in the terminal that started the server. Closing the webpage does not stop Python or release the model. Games are held only in page memory; reloading or closing the page loses the current game.

## Everyday use and configuration

After installation, run `cd laya-mlx-wzq` followed by `./run.sh` each time you want to play.

```bash
# Start without opening the browser
./run.sh --no-browser

# Use a different port
./run.sh --port 8767

# Reuse an existing complete local checkpoint
./run.sh --model /absolute/path/to/laya-multilingual-mlx

# Reuse another Python environment with laya-mlx installed
LAYA_PYTHON=/absolute/path/to/venv/bin/python ./run.sh \
  --model /absolute/path/to/laya-multilingual-mlx

# Invoke Python directly
.venv/bin/python -B server.py --model models/laya --port 8766
```

The defaults are **`models/laya` inside this repository** and **`.venv/bin/python` inside this repository**. No developer-specific directory is required. `run.sh` changes into the repository root, so relative model paths passed to it resolve there. When invoking `server.py` directly, explicit relative paths resolve against your current working directory. `--model` accepts a local directory, not a remote Hub model ID.

## Controls

| Action | Result |
| --- | --- |
| Click/tap an empty intersection | Place Black; the AI responds automatically |
| 悔棋一回合 — Undo a round | Remove your last move and the AI response; if only your move is pending, remove that move |
| 新的一局 — New game | Clear the board; an ongoing game asks for confirmation |
| 显示手数 — Show move numbers | Display the move sequence on the stones |
| Focus the board with `Tab`, then arrow keys | Move the keyboard cursor |
| `Enter` / Space | Play at the keyboard cursor |

Move, undo, and restart controls are locked while inference is in progress. If inference fails, your move is kept and you may retry or undo. The game does not silently substitute random moves or a rules-only AI for the model.

## How decisions work

```text
Human move → validate history / check terminal state
                              ↓
             Analyze nearby empty intersections
                              ↓
                   Up to six candidates
                              ↓
           Local Laya Agent.predict → choice probabilities
                              ↓
       Immediate-win / must-block constraints → White's move
```

The rules layer examines empty points within two cells of existing stones. It computes local features such as immediate wins, next-move winning points, and open threes, then ranks candidates heuristically. The model receives **text descriptions and a state summary**, not board images. It does not generate a written reasoning trace.

- In ordinary positions, the highest-probability model candidate is played.
- If White can win immediately, only winning candidates are eligible. Otherwise, if Black threatens to win on the next move, only blocking candidates are eligible.
- If this constraint changes the model's top choice, the UI shows the proposed move, executed move, and intervention reason.
- Candidate probabilities express **selection preference, not game win probability**. Tactical labels and explanations are generated by code.
- Local features, six candidates, and single-move constraints do not guarantee long-term strategy. Multiple simultaneous winning threats may be impossible to block.

This is a playable, inspectable local decision-model demonstration, not a professional Gomoku engine. No model training is performed by this project. The upstream Snake demo also uses existing weights without Snake-specific training. For a proposed training path, see the [detailed research document](docs/Laya游戏决策与五子棋训练调研.md), currently in Chinese.

Inference latency includes synchronized `Agent.predict`, tokenization, and result processing. Total move latency also includes rules analysis; it excludes initial model loading, HTTP transport, and browser rendering.

## Local operation and privacy

The server listens only on `127.0.0.1`, serving browsers on the same computer rather than the LAN. The code does not upload game histories, write game files, or load external fonts/CDNs. Ordinary HTTP request logs are printed to the terminal. Multiple pages share one model; inference is serialized, and concurrent requests may receive a busy response that can be retried.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Python environment not found | Create `.venv` and install the requirements, or set `LAYA_PYTHON` to an absolute executable path |
| Python too old / no matching MLX wheel | Check native arm64 Python 3.11+, the macOS requirements of the available wheel, and an up-to-date pip |
| Model or configuration missing | Download the complete checkpoint into `models/laya`, or pass `--model` |
| `Using SOCKS proxy ... socksio ...` | Run `.venv/bin/python -m pip install 'httpx[socks]'`, then retry the download |
| `No Metal device available` | Run from a normal local macOS terminal; a sandbox, container, or VM may not expose the GPU |
| `Address already in use` | Stop your previous server or choose `./run.sh --port 8767` |
| `Permission denied` when launching | Run `chmod +x run.sh 启动五子棋.command`, or invoke Python directly |
| Download connection error | Check your network/proxy and retry; offline play requires a complete checkpoint |
| Connection error in the webpage | Check that the terminal server is still running; restore it and retry, or reload for a new game |
| Weak AI moves | This is a feature-assisted model without project-specific fine-tuning; improve candidate generation/search or follow the research roadmap |

## Repository layout

```text
game.py                  Rules, local features, candidates, and constraints
server.py                Local HTTP server and real Laya inference
run.sh                   Terminal launcher
启动五子棋.command        macOS double-click launcher
requirements.txt         Python dependencies
web/                     HTML / CSS / JavaScript / Canvas frontend
tests/test_game.py       Model-independent rules and tactical tests
docs/                    Decision-model and training research
README.md                Chinese documentation
README.en.md             English documentation
models/laya/             Downloaded model; excluded from Git
.venv/                   Local environment; excluded from Git
```

## Development and tests

Rules tests do not require MLX, model weights, or a GPU:

```bash
python3 -B -m unittest discover -s tests -v
```

The 14 tests cover wins and overlines in every direction, board edges, invalid/duplicate/post-terminal moves, open threes and fours, legal candidates, win-before-block priority, and tactical overrides. Real-model responses, forced blocks and wins, browser moves, undo, restart, and connection-failure feedback have also been checked.

Use `./run.sh --no-browser --port 8767` for a temporary validation server. Stop it with `Ctrl+C` after validation, and delete generated test histories, screenshots, logs, and temporary environments. Do not leave test processes running.

## Upstream projects and research

- [Laya-MLX](https://github.com/mizorewww/laya-mlx): native MLX inference runtime, Apache-2.0.
- [Default model weights](https://huggingface.co/aac6fef/laya-multilingual-mlx): refer to the model and base-model cards for their licenses and provenance.
- [Upstream Laya](https://github.com/NandhaKishorM/laya): typed decision models and training implementation.
- [Game decisions and Gomoku training research](docs/Laya游戏决策与五子棋训练调研.md): whether Snake was trained, RLCD versus supervision and self-play, Gomoku data design, hardware, and evaluation. Includes a verification date and primary-source links; written in Chinese.

This is an independent Gomoku application, not an official game release from those projects. The repository does not include upstream model weights.
