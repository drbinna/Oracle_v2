"""
Rubric grader = the verifiable reward function.

Scores an agent's JSON answer against the weighted rubric. Every criterion is a
programmatic check (numeric tolerance, citation match, keyword presence) — no
LLM-as-judge anywhere. The weighted sum gives a DENSE reward in [0,1]: some
criteria pass, some fail, which is exactly the within-group variance GRPO needs.
"""
from __future__ import annotations
import json
import re


def _coerce(answer):
    """Agent output may be a dict or a JSON string (possibly fenced)."""
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return {}
    s = re.sub(r"^```(json)?|```$", "", answer.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    try:
        return json.loads(m.group(0) if m else s)
    except Exception:
        return {}


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").replace("$", "").strip())
    except Exception:
        return None


def _norm_accession(s: str) -> str:
    return re.sub(r"[^0-9]", "", str(s))


def _score_criterion(c: dict, ans: dict) -> float:
    t = c["type"]
    if t == "numeric":
        v = _num(ans.get(c["field"]))
        if v is None:
            return 0.0
        return 1.0 if abs(v - c["target"]) <= c["tol"] else 0.0
    if t == "citation":
        want = _norm_accession(c["accession"])
        cites = ans.get(c["field"]) or []
        if isinstance(cites, str):
            cites = [cites]
        return 1.0 if any(_norm_accession(x) == want for x in cites) else 0.0
    if t == "keywords":
        opts = [o.lower() for o in c.get("any_of", [])]
        if not opts:
            return 0.0  # criterion not yet populated -> contributes 0 (honest)
        blob = " ".join(map(str, ans.get(c["field"], []) or [])).lower()
        return 1.0 if any(o in blob for o in opts) else 0.0
    return 0.0


def grade(answer, rubric: dict) -> dict:
    ans = _coerce(answer)
    total_w = sum(c["weight"] for c in rubric["criteria"])
    breakdown, reward = [], 0.0
    for c in rubric["criteria"]:
        s = _score_criterion(c, ans)
        reward += s * c["weight"]
        breakdown.append({"id": c["id"], "section": c["section"],
                          "weight": c["weight"], "score": s})
    reward = reward / total_w if total_w else 0.0
    return {"reward": round(reward, 4), "breakdown": breakdown}
