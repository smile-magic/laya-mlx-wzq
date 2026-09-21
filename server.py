"""Local-only Gomoku UI and a serialized, real Laya inference endpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser

from game import WHITE, candidates, coordinate, replay, select_move, winning_line

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/laya"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def decision_prompt(board, moves, options):
    """Full board plus compact, explicit candidate consequences, not a trace."""
    strongest = max(1, max(x["score"] for x in options))
    criteria = {}
    for option in options:
        if option["tier"] == 4:
            description = "Win immediately."
        elif option["tier"] == 3:
            description = "Create two winning points; Black cannot block both."
        elif option["tier"] == 0:
            description = "Black can win immediately after this move."
        elif option["tier"] == 1:
            description = "Black can create two winning points after this move."
        elif option["future_wins"]:
            description = "Force Black to block one winning point."
        else:
            description = (f"No immediate losing reply detected. "
                           f"Create {option['attack'].open_threes} open threes.")
        quality = round(max(0, option["score"]) / strongest * 100)
        eligibility = "Eligible" if option["eligible"] else "Avoid"
        criteria[option["label"]] = f"{eligibility}. {description} Shape quality {quality}/100."
    question = {"move": {"type": "choice",
                         "instructions": "Choose White's best eligible Gomoku move. Prefer winning, then defense, then stronger shape.",
                         "criteria": criteria}}
    rows = "\n".join(f"{r + 1:02d} " + " ".join(".BW"[cell] for cell in row)
                     for r, row in enumerate(board))
    state = ("Freestyle Gomoku, 15x15. Five or more in a row wins. "
             "You are WHITE (W), human is BLACK (B), dot means empty. "
             "Candidate consequences and shape quality are computed by rules, not win probabilities. "
             "A losing reply check is shallow, not a guarantee of long-term safety. "
             f"Move count: {len(moves)}. Black last played "
             f"{coordinate(moves[-1]['r'], moves[-1]['c'])}.\n"
             "Columns A B C D E F G H I J K L M N O; rows 1 to 15:\n" + rows)
    return state, question


class Policy:
    def __init__(self, model):
        # Imported here so rules/tests remain usable without a GPU or MLX.
        from laya_mlx import Agent
        self.agent = Agent(model, dtype="float16", device="gpu", batch_size=1)
        self.lock = threading.Lock()

    def decide(self, moves):
        started = time.perf_counter()
        board, winner, line = replay(moves)
        if winner or len(moves) == 225:
            return {"move": None, "winner": winner, "line": line, "draw": winner is None}
        if not moves or len(moves) % 2 != 1:
            raise ValueError("需要先由黑棋落子，再轮到白棋。")
        options = candidates(board)
        state, question = decision_prompt(board, moves, options)
        inference_started = time.perf_counter()
        result = self.agent.predict(state, question)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        probabilities = result["answers"]["move"]["probabilities"]
        if (set(probabilities) != set(question["move"]["criteria"]) or
                any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()) or
                sum(probabilities.values()) <= 0):
            raise RuntimeError("模型返回的候选概率无效。")
        proposed, executed, reason = select_move(options, probabilities)
        r, c = executed["r"], executed["c"]
        board[r][c] = WHITE
        line = winning_line(board, r, c)
        return {
            "move": {"r": r, "c": c}, "winner": WHITE if line else None,
            "line": line, "draw": not line and len(moves) + 1 == 225,
            "analysis": {
                "proposed": proposed["label"], "executed": executed["label"],
                "intervened": proposed["label"] != executed["label"], "reason": reason,
                "inference_ms": round(inference_ms, 1),
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
                "input_tokens": result.get("usage", {}).get("input_tokens"),
                "candidates": [{"label": x["label"], "r": x["r"], "c": x["c"],
                                "probability": probabilities[x["label"]],
                                "attack": x["attack"].summary, "defense": x["defense"].summary,
                                "eligible": x["eligible"], "tactic": x["tactic"]}
                               for x in sorted(options, key=lambda x: probabilities[x["label"]], reverse=True)],
            },
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def respond(self, code, body, content_type="application/json; charset=utf-8"):
        data = json.dumps(body, ensure_ascii=False).encode() if isinstance(body, dict) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return (self.headers.get("Host") in hosts and
                (not origin or origin in {f"http://{host}" for host in hosts}))

    def do_GET(self):
        if not self.local_request():
            return self.respond(403, {"error": "仅允许本机页面访问。"})
        if self.path == "/api/health":
            return self.respond(200, {"ready": True, "model": "Laya multilingual · 322M · FP16", "offline": True})
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path not in files:
            return self.respond(404, {"error": "页面不存在。"})
        name, mime = files[self.path]
        return self.respond(200, (ROOT / "web" / name).read_bytes(), mime)

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, {"error": "仅允许本机页面访问。"})
        if self.path != "/api/move":
            return self.respond(404, {"error": "接口不存在。"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16000:
                raise ValueError("请求大小不正确。")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or set(body) != {"moves"}:
                raise ValueError("请求需要完整落子记录。")
            # Validate history before occupying the GPU.
            replay(body["moves"])
        except (ValueError, UnicodeDecodeError) as error:
            return self.respond(400, {"error": str(error)})
        if not self.server.policy.lock.acquire(blocking=False):
            return self.respond(429, {"error": "模型正在处理另一局，请稍后重试。"})
        try:
            result = self.server.policy.decide(body["moves"])
        except ValueError as error:
            return self.respond(400, {"error": str(error)})
        except Exception as error:
            print(f"Inference failed: {error!r}", flush=True)
            return self.respond(500, {"error": "模型推理失败，请查看终端信息并重试。"})
        finally:
            self.server.policy.lock.release()
        return self.respond(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    args.model = args.model.expanduser().resolve()
    if not 0 <= args.port <= 65535:
        parser.error("端口必须在 0–65535 之间。Port must be between 0 and 65535.")
    if not (args.model / "model.safetensors").is_file():
        parser.error(f"Local model not found / 找不到本地模型：{args.model}\n"
                     "Download the checkpoint first / 请先下载权重：\n"
                     "  hf download aac6fef/laya-multilingual-mlx --local-dir models/laya")
    # Bind before allocating a model so an occupied port fails immediately.
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print("正在加载本地 Laya 模型并预热 GPU…", flush=True)
        server.policy = Policy(args.model)
        server.policy.decide([{"r": 7, "c": 7}])
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"五子棋已就绪：{url}\n全部推理在本机运行。按 Ctrl+C 停止服务。", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务已停止。", flush=True)


if __name__ == "__main__":
    main()
