import json
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# 1. NEURAL NETWORK BOT (4 → 3 → 4 → 3)
# =========================================================
class PongBot(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_bot(steps=1200, batch_size=256):
    model = PongBot()
    opt = optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    W, H = 900.0, 520.0

    for _ in range(steps):
        ball_y = torch.rand(batch_size) * H
        ball_vy = (torch.rand(batch_size) * 2 - 1) * 8
        paddle_y = torch.rand(batch_size) * H
        ball_x = torch.rand(batch_size) * W

        target = torch.where(
            ball_y < paddle_y - 10, torch.zeros(batch_size, dtype=torch.long),
            torch.where(ball_y > paddle_y + 10,
                        torch.full((batch_size,), 2, dtype=torch.long),
                        torch.ones(batch_size, dtype=torch.long))
        )

        X = torch.stack([
            ball_y / H,
            ball_vy / 8,
            paddle_y / H,
            ball_x / W
        ], dim=1)

        opt.zero_grad()
        loss = loss_fn(model(X), target)
        loss.backward()
        opt.step()

    model.eval()

    def export(layer):
        return {"w": layer.weight.tolist(), "b": layer.bias.tolist()}

    return {
        "layers": [
            export(model.fc1),
            export(model.fc2),
            export(model.fc3)
        ]
    }


# =========================================================
# 2. STREAMLIT PAGE
# =========================================================
st.set_page_config(layout="wide", page_title="Pong Neural Network Bot")
st.title("Pong Game – Neural Network Bot")

st.markdown("""
**Player 1:** 🤖 BOT  
**Player 2:** 🧑 Keyboard (W = UP, A = DOWN)  
**Pause / Resume:** SPACE  
👉 Click inside the game to activate keyboard
""")

bot_json = json.dumps(train_bot())

# =========================================================
# 3. HTML + JAVASCRIPT GAME (REAL GAME LOGIC)
# =========================================================
html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
body {{ margin:0; background:#0f0f0f; }}
canvas {{ display:block; margin:auto; background:#000; }}
#top {{
    color:#ccc;
    text-align:center;
    font-family:Arial;
    padding:8px;
}}
</style>
</head>

<body>
<div id="top">
Player 1: <b>BOT</b> | Player 2: <b>W / A</b> | Pause: <b>Space</b>
</div>

<canvas id="c" width="900" height="520"></canvas>

<script>
const BOT = {bot_json};
const c = document.getElementById("c");
const ctx = c.getContext("2d");

let paused = false;

let paddleL = {{ x:20, y:210 }};
let paddleR = {{ x:870, y:210 }};
let ball = {{ x:450, y:260, vx:6, vy:4 }};
let scoreL = 0, scoreR = 0;

function relu(x) {{ return Math.max(0, x); }}

function matvec(W,b,x){{
  return W.map((r,i)=>r.reduce((s,v,j)=>s+v*x[j], b[i]));
}}

function forward(inp){{
  let z1 = matvec(BOT.layers[0].w, BOT.layers[0].b, inp).map(relu);
  let z2 = matvec(BOT.layers[1].w, BOT.layers[1].b, z1).map(relu);
  let z3 = matvec(BOT.layers[2].w, BOT.layers[2].b, z2);
  return z3.indexOf(Math.max(...z3));
}}

document.addEventListener("keydown", e => {{
  if(e.code === "Space") paused = !paused;
  if(e.key === "w") paddleR.y -= 15;
  if(e.key === "a") paddleR.y += 15;
}});

function reset() {{
  ball.x = 450; ball.y = 260;
  ball.vx *= -1;
}}

function update() {{
  if(paused) return;

  let inp = [
    ball.y / 520,
    ball.vy / 8,
    paddleL.y / 520,
    ball.x / 900
  ];

  let action = forward(inp);
  if(action === 0) paddleL.y -= 6;
  if(action === 2) paddleL.y += 6;

  ball.x += ball.vx;
  ball.y += ball.vy;

  if(ball.y < 0 || ball.y > 520) ball.vy *= -1;

  if(ball.x < 30 && ball.y > paddleL.y && ball.y < paddleL.y + 100) ball.vx *= -1;
  if(ball.x > 860 && ball.y > paddleR.y && ball.y < paddleR.y + 100) ball.vx *= -1;

  if(ball.x < 0) {{ scoreR++; reset(); }}
  if(ball.x > 900) {{ scoreL++; reset(); }}
}}

function draw() {{
  ctx.clearRect(0,0,900,520);

  ctx.fillStyle = "#fff";
  ctx.fillRect(paddleL.x, paddleL.y, 10, 100);
  ctx.fillRect(paddleR.x, paddleR.y, 10, 100);

  ctx.beginPath();
  ctx.arc(ball.x, ball.y, 8, 0, Math.PI*2);
  ctx.fill();

  ctx.font = "20px Arial";
  ctx.fillText(scoreL, 420, 30);
  ctx.fillText(scoreR, 460, 30);

  if(paused){{
    ctx.font = "40px Arial";
    ctx.fillText("PAUSED", 350, 260);
  }}
}}

function loop() {{
  update();
  draw();
  requestAnimationFrame(loop);
}}

loop();
</script>
</body>
</html>
"""

components.html(html, height=620)

st.caption("Neural Network Pong Bot – PyTorch + Streamlit")
