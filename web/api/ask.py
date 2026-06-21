# api/ask.py
# ---------------------------------------------------------------------------
# Backend for the Quant Firm judge chatbot. Stdlib only — no pip installs.
#
# Front-end contract (wired in quant_firm_viewer.html):
#   POST /api/ask   body: {"question": "...", "history": [optional]}  -> {"answer": "..."}
#
# CONFIG (Vercel project env vars; key stays server-side):
#   LLM_API_KEY   = your Fireworks key            (required)
#   LLM_MODEL     = pin one model (optional)       -- if unset, tries FALLBACK_MODELS
#   LLM_BASE_URL  = OpenAI-compatible base         (optional; default Fireworks)
#
# To run on the model you trained instead (HUD gateway):
#   LLM_BASE_URL = https://inference.beta.hud.ai/v1
#   LLM_API_KEY  = your HUD key
#   LLM_MODEL    = Qwen/Qwen3-8B
# ---------------------------------------------------------------------------

import os, re, json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.fireworks.ai/inference/v1")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "").strip()

# Fireworks serverless models confirmed available (June 2026). Tried in order
# until one responds; the first that isn't NOT_FOUND wins.
FALLBACK_MODELS = [
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/qwen3p7-plus",
    "accounts/fireworks/models/minimax-m3",
]

SYSTEM = """You are the assistant for Quant Firm (codename Oracle) — a verifiable-reward \
reinforcement-learning environment built on HUD for the Frontier/RSI RL Environments Hackathon. \
Answer questions from judges accurately and concisely (2-4 sentences), using ONLY the facts below. \
If asked something outside this record — a live stock price, a figure not given here, or an \
unrelated topic — say it's outside what you can speak to and point to the verified demo above for \
filing-grounded numbers. Never invent figures. Be candid about the negative result; it is the contribution.

WHAT IT IS
- An agent answers financial-analysis questions (e.g., a company's FY2022 gross margin) that are computed, cited to a real SEC 10-K, and independently audited.
- Ground truth is SEC EDGAR / XBRL only. Exa is an agent-side discovery tool for finding filings; it never grades.
- The reward is correctness against the filing — never trading P&L. (Filings are static, so the market being open or closed is irrelevant.)

THE REWARD (verifiable, programmatic — no LLM judge)
- A weighted rubric scores each answer: gross margin 0.45, revenue 0.15, citation (resolvable accession) 0.20, MD&A driver 0.20.
- An independent auditor subagent recomputes every figure from the filing and rejects anything that doesn't reconcile — the defense against reward hacking. In the demo, an honest run reconciles to 43.31% and scores 1.00; a faked 60% margin is rejected and collapses to 0.35.

THE STACK
- Policy model: Qwen/Qwen3-8B. Training: hosted GRPO/LoRA via HUD's TrainingClient (Tinker backend) with LocalRuntime rollouts. Tools: FastMCP specialist servers. Discovery: Exa. Ground truth: SEC EDGAR/XBRL.

THE HONEST RESULT (the contribution)
- Hand-feeding answers (oracle tools) saturated the model: 8/10 tasks had zero within-group variance -> NO-GO.
- Exa self-retrieval restored spread on 9/10 tasks (90%) -> the variance gate passed (GO).
- But GRPO still trained flat (10 steps, mean reward about 0.333, within noise): the spread was exogenous — retrieval luck the policy did not control — so the gradient had nothing to climb.
- A redesigned controllable-variance reward was then saturated by Qwen3-8B (already mastered). So the task is untrainable for a capable 8B both ways.

THE TRAINABILITY LAW
- A verifiable RL reward is trainable only if its variance is policy-controllable, not exogenous: I(A_policy ; R | X_exogenous) > 0. Passing a variance gate (Var(R) > 0) is NOT the same as having spread a gradient can climb.
- The reusable takeaway is a controllable-variance gate that runs BEFORE spending compute. This is the precondition that GRPO fixes like RC-GRPO (arXiv:2602.03025) and binary-reward gradient-starvation analyses (arXiv:2605.07689) implicitly assume.

WHO USES IT
- Analysts use it directly — the person who reads 10-Ks, computes margins, and writes the note.
- A systematic quant consumes it upstream, as a verified fundamental-feature pipeline feeding their model: we verify the number, they make the trade. That is why the reward is correctness, never P&L."""


def _post(model, messages):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 700,
    }).encode("utf-8")
    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + LLM_API_KEY,
                 "x-api-key": LLM_API_KEY},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=55) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text


def ask_llm(question, history=None):
    if not LLM_API_KEY:
        return ("The backend is deployed but no model key is set. "
                "Set LLM_API_KEY in the server environment to enable live answers.", None)
    messages = [{"role": "system", "content": SYSTEM}]
    if isinstance(history, list):
        messages += history[-6:]
    messages.append({"role": "user", "content": question})

    candidates = [LLM_MODEL] if LLM_MODEL else FALLBACK_MODELS
    last_detail = ""
    for model in candidates:
        try:
            return _post(model, messages), model
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            last_detail = body[:300]
            # model unavailable on this account -> try the next candidate
            if e.code == 404 or "NOT_FOUND" in body or "not found" in body.lower():
                continue
            raise RuntimeError("model error (%s): %s" % (e.code, last_detail))
    raise RuntimeError("no available model. tried: %s | last: %s"
                       % (", ".join(candidates), last_detail))


class handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        self._json(200, {"ok": True, "service": "quant-firm chat"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                self._json(400, {"error": "missing 'question'"})
                return
            answer, used = ask_llm(question, body.get("history"))
            self._json(200, {"answer": answer, "model": used})
        except Exception as e:
            self._json(502, {"error": str(e)})
