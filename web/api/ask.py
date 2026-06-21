# api/ask.py
# ---------------------------------------------------------------------------
# Backend for the Quant Firm judge chatbot. Stdlib only — no pip installs.
#
# Front-end contract (already wired in quant_firm_viewer.html):
#   POST /api/ask      body: {"question": "...", "history": [optional]}
#   200  ->            {"answer": "..."}
#
# DEPLOY (Vercel, same-origin — simplest, no CORS):
#   1. Put quant_firm_landing.html + quant_firm_viewer.html at the repo root
#      you deploy, and this file at  api/ask.py .
#   2. In the Vercel project settings, set environment variables:
#        LLM_API_KEY   = your Fireworks key   (required; stays server-side)
#        LLM_MODEL     = a model on your account (optional; default below)
#        LLM_BASE_URL  = OpenAI-compatible base (optional; default Fireworks)
#   3. Deploy. The viewer calls /api/ask on the same origin, so no CORS setup.
#
# Run it on the SAME model you trained instead (on-brand) — point it at the
# HUD gateway:
#        LLM_BASE_URL = https://inference.beta.hud.ai/v1
#        LLM_API_KEY  = your HUD key
#        LLM_MODEL    = Qwen/Qwen3-8B
#
# If you host the static files somewhere WITHOUT serverless functions
# (e.g. GitHub Pages), deploy this elsewhere and set ORACLE_ENDPOINT in the
# viewer to its full URL (CORS headers below already allow that).
# ---------------------------------------------------------------------------

import os, json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.fireworks.ai/inference/v1")
LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
LLM_MODEL    = os.environ.get("LLM_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct")

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


def ask_llm(question, history=None):
    if not LLM_API_KEY:
        return ("The backend is deployed but no model key is set. "
                "Set LLM_API_KEY in the server environment to enable live answers.")
    messages = [{"role": "system", "content": SYSTEM}]
    if isinstance(history, list):
        messages += history[-6:]
    messages.append({"role": "user", "content": question})
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 500,
    }).encode("utf-8")
    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + LLM_API_KEY},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


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
            answer = ask_llm(question, body.get("history"))
            self._json(200, {"answer": answer})
        except urllib.error.HTTPError as e:
            self._json(502, {"error": "model error", "detail": e.read().decode("utf-8", "ignore")[:300]})
        except Exception as e:
            self._json(500, {"error": str(e)})
