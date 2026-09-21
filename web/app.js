"use strict";
const $ = id => document.getElementById(id);
const canvas = $("board");
const ctx = canvas.getContext("2d");
const N = 15, PAD = 0.068, SPAN = 1 - 2 * PAD;
let moves = [], analyses = [], busy = false, ready = false, failed = false;
let winner = null, winLine = [], hover = null, cursor = {r: 7, c: 7};
let keyboardFocus = false;
const label = (r, c) => `${String.fromCharCode(65 + c)}${r + 1}`;
const boardState = () => {
  const b = Array.from({length: N}, () => Array(N).fill(0));
  moves.forEach((m, i) => { b[m.r][m.c] = i % 2 + 1; });
  return b;
};
const finished = () => winner !== null || moves.length === N * N;

function detectWinner() {
  winner = null; winLine = [];
  if (!moves.length) return;
  const b = boardState(), last = moves.at(-1), color = b[last.r][last.c];
  for (const [dr, dc] of [[0,1],[1,0],[1,1],[1,-1]]) {
    const line = [[last.r,last.c]];
    for (const sign of [-1,1]) {
      let r = last.r + dr * sign, c = last.c + dc * sign;
      while (r >= 0 && r < N && c >= 0 && c < N && b[r][c] === color) {
        line.push([r,c]); r += dr * sign; c += dc * sign;
      }
    }
    if (line.length >= 5) { winner = color; winLine = line; return; }
  }
}

function draw() {
  const size = canvas.getBoundingClientRect().width;
  if (!size) return;
  const dpr = window.devicePixelRatio || 1;
  const pixels = Math.round(size * dpr);
  if (canvas.width !== pixels || canvas.height !== pixels) { canvas.width = pixels; canvas.height = pixels; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0,0,size,size);
  const pad = size * PAD, cell = size * SPAN / (N - 1);
  const at = v => pad + cell * v;
  const bg = ctx.createLinearGradient(0,0,size,size);
  bg.addColorStop(0,"#f0e6d0"); bg.addColorStop(1,"#e7d9bb");
  ctx.fillStyle = bg; ctx.fillRect(0,0,size,size);
  // Restrained wood grain; deterministic and decorative only.
  ctx.strokeStyle = "rgba(146,115,62,.035)"; ctx.lineWidth = .6;
  for (let y = 5; y < size; y += 7) { ctx.beginPath();ctx.moveTo(0,y);ctx.bezierCurveTo(size*.3,y-3,size*.7,y+4,size,y+1);ctx.stroke(); }
  ctx.strokeStyle = "#998969"; ctx.lineWidth = .65;
  for (let i = 0; i < N; i++) {
    ctx.beginPath();ctx.moveTo(at(0),at(i));ctx.lineTo(at(14),at(i));ctx.stroke();
    ctx.beginPath();ctx.moveTo(at(i),at(0));ctx.lineTo(at(i),at(14));ctx.stroke();
  }
  ctx.strokeStyle = "#8b7b5f";ctx.lineWidth = 1.2;ctx.strokeRect(pad,pad,cell*14,cell*14);
  ctx.fillStyle = "#a08d6b";ctx.font = `${Math.max(8, size*.016)}px -apple-system, sans-serif`;
  ctx.textAlign = "center";ctx.textBaseline = "middle";
  for (let i = 0; i < N; i++) {ctx.fillText(String.fromCharCode(65+i),at(i),pad*.43);ctx.fillText(String(i+1),pad*.4,at(i));}
  ctx.fillStyle = "#8b795b";
  for (const [r,c] of [[3,3],[3,11],[7,7],[11,3],[11,11]]) {ctx.beginPath();ctx.arc(at(c),at(r),Math.max(2,size*.004),0,Math.PI*2);ctx.fill();}
  const showNumbers = $("numbers").checked;
  moves.forEach((m,i) => {
    const x=at(m.c),y=at(m.r),rad=cell*.428,black=i%2===0;
    ctx.save();ctx.shadowColor="rgba(32,26,13,.26)";ctx.shadowBlur=cell*.14;ctx.shadowOffsetY=cell*.09;
    const g=ctx.createRadialGradient(x-rad*.35,y-rad*.4,0,x,y,rad);
    g.addColorStop(0,black?"#555952":"#fffefa");g.addColorStop(.7,black?"#292e29":"#f4f2e9");g.addColorStop(1,black?"#1e231f":"#d9d6c8");
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);ctx.fill();ctx.restore();
    if (showNumbers) {ctx.fillStyle=black?"#e7eade":"#616a5b";ctx.font=`500 ${cell*.32}px -apple-system, sans-serif`;ctx.fillText(String(i+1),x,y+.5);}
    else if (i===moves.length-1) {ctx.fillStyle=black?"#d9aa78":"#ac5a3e";ctx.beginPath();ctx.arc(x,y,cell*.08,0,Math.PI*2);ctx.fill();}
    if(winLine.some(([r,c])=>r===m.r&&c===m.c)){ctx.strokeStyle="#b96542";ctx.lineWidth=Math.max(1.5,cell*.05);ctx.beginPath();ctx.arc(x,y,rad*.82,0,Math.PI*2);ctx.stroke();}
  });
  const focus = keyboardFocus ? cursor : hover;
  if(focus && !busy && ready && !failed && !finished() && moves.length%2===0 && !boardState()[focus.r][focus.c]) {
    ctx.fillStyle="rgba(41,48,38,.2)";ctx.beginPath();ctx.arc(at(focus.c),at(focus.r),cell*.4,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle="#647158";ctx.lineWidth=1;ctx.strokeRect(at(focus.c)-cell*.47,at(focus.r)-cell*.47,cell*.94,cell*.94);
  }
}

function setError(message) {
  $("error").textContent=message || "";$("error").classList.toggle("hidden", !message);
}

function renderAnalysis() {
  const analysis = analyses.at(-1)?.data;
  $("lastMove").textContent = analysis?.executed || "—";
  $("latency").replaceChildren(document.createTextNode(analysis ? String(analysis.inference_ms) : "—"));
  const unit=document.createElement("small");unit.textContent=" ms";$("latency").append(unit);
  $("candidateEmpty").classList.toggle("hidden", !!analysis || busy);
  $("thinking").classList.toggle("hidden", !busy);
  $("candidates").replaceChildren();
  const note=$("decisionNote");note.classList.toggle("hidden", !analysis);note.classList.toggle("shield", !!analysis?.intervened);
  if (!analysis) return;
  for (const option of analysis.candidates) {
    const row=document.createElement("div");row.className="candidate"+(option.label===analysis.executed?" chosen":"");
    const top=document.createElement("div");top.className="candidate-top";
    const name=document.createElement("span");name.className="candidate-name";name.textContent=option.label;
    const detail=document.createElement("span");detail.className="candidate-detail";
    detail.textContent=option.label===analysis.executed?"本次落点":option.attack;
    const percent=document.createElement("span");percent.className="candidate-percent";percent.textContent=`${(option.probability*100).toFixed(1)}%`;
    top.append(name,detail,percent);
    const bar=document.createElement("div");bar.className="bar";
    const fill=document.createElement("div");fill.className="bar-fill";fill.style.width=`${Math.min(100,Math.max(0,option.probability*100))}%`;
    bar.append(fill);row.append(top,bar);$("candidates").append(row);
  }
  note.textContent=analysis.intervened
    ? `战术规则介入：模型首选 ${analysis.proposed} → 实际落点 ${analysis.executed}。${analysis.reason}。`
    : `${analysis.reason} · 整步耗时 ${Math.round(analysis.total_ms)} ms`;
}

function render() {
  $("moveCount").textContent=String(moves.length).padStart(2,"0");
  $("newGame").disabled=!ready||busy;$("undo").disabled=!moves.length||busy;
  $("retry").classList.toggle("hidden", !failed || busy);
  const yourTurn=ready&&!busy&&!failed&&!finished();
  $("yourTurn").textContent="轮到你";$("yourTurn").classList.toggle("hidden",!yourTurn);
  $("aiTurn").classList.toggle("hidden",!busy);
  $("statusDot").className="dot"+(ready&&!failed?" ready":"");
  $("status").textContent=!ready?"模型未就绪，请检查终端服务。":busy?"Laya 正在思考，请稍候…":failed?"AI 暂未落子，可重试或悔棋。":winner===1?"你赢了，漂亮的一局。":winner===2?"Laya 连成五子，再来一局？":finished()?"棋盘已满，本局和棋。":"轮到你，在交叉点落下一颗黑子。";
  canvas.setAttribute("aria-busy",String(busy));
  renderAnalysis();draw();
}

async function requestAI() {
  busy=true;failed=false;setError("");render();
  try {
    const response=await fetch("/api/move",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({moves}),signal:AbortSignal.timeout(60000)});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"模型暂时无法响应。");
    $("connectionText").textContent="本地模型已连接";$("connectionDot").className="dot ready";
    if(result.move) {
      const {r,c}=result.move;
      if(!Number.isInteger(r)||!Number.isInteger(c)||r<0||r>=N||c<0||c>=N||boardState()[r][c])throw new Error("模型返回了无效落点。");
      if(!result.analysis)throw new Error("缺少模型决策信息，请重试。");
      moves.push(result.move);analyses.push({ply:moves.length,data:result.analysis});
    } else if(!result.winner&&!result.draw) throw new Error("模型没有返回落点，请重试。");
    detectWinner();
  } catch(error) {
    if(error instanceof TypeError||error.name==="TimeoutError") {
      $("connectionText").textContent="服务连接异常";$("connectionDot").className="dot error-dot";
    }
    failed=true;setError(error.name==="TimeoutError"?"推理超时，等待片刻后可重试。":error.message==="Failed to fetch"?"无法连接本地服务。请重新运行启动脚本，然后重试。":error.message);
  } finally {busy=false;render();}
}

async function place(r,c) {
  if(!ready||busy||failed||finished()||moves.length%2!==0||boardState()[r][c])return;
  moves.push({r,c});hover=null;cursor={r,c};detectWinner();render();
  if(!finished())await requestAI();
}

function pointerPosition(event) {
  const rect=canvas.getBoundingClientRect(),cell=rect.width*SPAN/(N-1),pad=rect.width*PAD;
  const x=(event.clientX-rect.left-pad)/cell,y=(event.clientY-rect.top-pad)/cell;
  const c=Math.round(x),r=Math.round(y);
  return r>=0&&r<N&&c>=0&&c<N&&Math.abs(x-c)<.48&&Math.abs(y-r)<.48?{r,c}:null;
}
canvas.addEventListener("pointermove",event=>{keyboardFocus=false;hover=pointerPosition(event);draw();});
canvas.addEventListener("pointerleave",()=>{hover=null;draw();});
canvas.addEventListener("click",event=>{const position=pointerPosition(event);keyboardFocus=false;if(position)place(position.r,position.c);});
canvas.addEventListener("keydown",event=>{
  const directions={ArrowUp:[-1,0],ArrowDown:[1,0],ArrowLeft:[0,-1],ArrowRight:[0,1]};
  if(directions[event.key]) {
    event.preventDefault();keyboardFocus=true;const [dr,dc]=directions[event.key];
    cursor={r:Math.min(14,Math.max(0,cursor.r+dr)),c:Math.min(14,Math.max(0,cursor.c+dc))};
    $("focusPosition").textContent=`当前：${label(cursor.r,cursor.c)}`;draw();
  } else if(event.key==="Enter"||event.key===" "){event.preventDefault();place(cursor.r,cursor.c);}
});
canvas.addEventListener("focus",()=>{keyboardFocus=true;draw();});
canvas.addEventListener("blur",()=>{keyboardFocus=false;draw();});
$("numbers").addEventListener("change",draw);
$("retry").addEventListener("click",()=>{if(failed&&!busy)requestAI();});
$("undo").addEventListener("click",()=>{
  if(busy||!moves.length)return;
  moves.splice(moves.length-(moves.length%2===0?2:1));
  analyses=analyses.filter(a=>a.ply<=moves.length);failed=false;setError("");detectWinner();render();
});
function reset() {moves=[];analyses=[];winner=null;winLine=[];failed=false;hover=null;cursor={r:7,c:7};$("focusPosition").textContent="";setError("");render();}
$("newGame").addEventListener("click",()=>{if(busy)return;if(moves.length&&!finished())$("resetDialog").showModal();else reset();});
$("resetDialog").addEventListener("close",()=>{if($("resetDialog").returnValue==="reset")reset();});
new ResizeObserver(draw).observe(canvas);
async function connect() {
  try {
    const response=await fetch("/api/health",{signal:AbortSignal.timeout(5000)});
    const health=await response.json();if(!response.ok||!health.ready)throw new Error("模型未就绪");
    ready=true;$("connectionText").textContent="本地模型已连接";$("connectionDot").className="dot ready";
  } catch {
    $("connectionText").textContent="服务未连接";$("connectionDot").className="dot error-dot";
    setError("请先运行「启动五子棋.command」，然后刷新页面。");
  }
  render();
}
render();connect();
