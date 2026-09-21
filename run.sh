#!/bin/sh
set -eu
CDPATH= cd -- "$(dirname -- "$0")"
GOMOKU_PYTHON="${LAYA_PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$GOMOKU_PYTHON" ]; then
    printf '%s\n' 'Python environment not found / 未找到 Python 环境。' \
        'Run the installation steps in README.md or README.en.md first.' \
        '请先按 README 安装；也可设置 LAYA_PYTHON 为 Python 可执行文件的绝对路径。' >&2
    exit 1
fi
exec "$GOMOKU_PYTHON" -B server.py "$@"
