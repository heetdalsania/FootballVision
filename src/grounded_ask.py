"""
Grounded Ask — natural-language questions answered only from measured data

This ports the grounding pattern from Soccer Companion (`src/lib/grounding/
wikipedia.ts`, `buildGroundingBlock`) into FootballVision, with one change
that matters: Soccer Companion grounded a language model against **Wikipedia
text**, which is text checking text. Here the grounding source is the **output
of the computer-vision pipeline** — player coordinates in metres derived from
pixels, team assignments, pitch control, compactness, pressing.

That makes the arbitration genuinely cross-modal. It is also the direct
follow-on to the WACV finding that models fabricate specific evidence and that
a second language model cannot detect it: the answer proposed there was
evidence the model did not get to write, and detector output is exactly that.

Design rules, inherited from the Soccer Companion prompts:
  * A FACTS block is assembled from measurements only.
  * The model is told those are the ONLY facts it may state.
  * If a question cannot be answered from the block, it must say so rather
    than guess. That refusal path is the whole point.

Backends: local Ollama by default (no API key, no per-query cost), or
Anthropic when ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
# Smaller first, deliberately. This is an interactive panel and the task is
# reasoning over a short table of numbers, not open-ended generation, so
# latency matters more than model size — especially since the video pipeline
# is competing for the same GPU. The 14B stays as a fallback.
OLLAMA_MODELS = [
    "qwen2.5-coder:7b",
    "qwen3-vl:8b-instruct",
    "qwen2.5-coder:14b-instruct-q5_K_M",
]

SYSTEM_PROMPT = """You are a football analyst assistant reading measurements taken from video by a computer-vision pipeline.

You will be given a FACTS block containing everything the pipeline measured for the current moment of the match.

Rules you must follow:
- The FACTS block is the ONLY source of information you may state. Treat it as the complete truth about this moment.
- Never state a specific number, position, or claim that is not in the FACTS block. If you happen to "know" something about football that is not measured here, do not assert it about this match.
- If the question cannot be answered from the FACTS, say plainly that the data does not cover it. Do not guess, and do not fill the gap with plausible-sounding detail.
- You may reason over the facts (compare two numbers, note which team is higher, describe shape) as long as every number you cite appears in the block.
- Coordinates are in metres on a 105 x 68 m pitch. x runs goal to goal, y runs across the width. Team A defends the low-x end.
- Read each metric in the direction the FACTS block states. Several are inverted: a SMALLER compactness number means a team is MORE compact, and a SMALLER pressing distance means TIGHTER engagement. Do not assume a bigger number is "more" of the quality being named.
- Be brief and concrete. Two or three sentences unless more is genuinely needed. Write like an analyst talking to a coach, not like a chatbot."""


def build_facts_block(pitch: Dict) -> Tuple[str, bool]:
    """
    Assemble the FACTS block from one frame's measurements.

    Returns (block, has_data). Pure function: no model call, so the grounding
    layer can be tested without a backend.
    """
    if not pitch:
        return "FACTS: none. No frame has been analysed yet.", False

    players = pitch.get("players") or []
    outfield = [p for p in players if p.get("role") == "player"]
    if not outfield:
        return "FACTS: none. No players are currently detected on the pitch.", False

    lines: List[str] = []
    counts = pitch.get("counts") or {}
    tc = pitch.get("team_counts") or {}

    lines.append("DETECTED THIS FRAME")
    lines.append(f"- outfield players: {len(outfield)}")
    lines.append(f"- team A players: {tc.get('0', 0)}, team B players: {tc.get('1', 0)}")
    if counts.get("goalkeeper"):
        lines.append(f"- goalkeepers: {counts['goalkeeper']}")
    if counts.get("referee"):
        lines.append(f"- referees: {counts['referee']}")

    cal = pitch.get("calibrated")
    lines.append(
        f"- pitch calibration: {'solved' if cal else 'NOT solved'}"
        + (f" from {pitch.get('n_keypoints')} detected landmarks" if cal else "")
    )
    if not cal:
        lines.append("  (positions below are unreliable while calibration is unsolved)")

    ball = pitch.get("ball")
    if ball:
        lines.append(f"- ball position: x={ball['x']} m, y={ball['y']} m")
    else:
        lines.append("- ball: not detected this frame")

    concepts = pitch.get("concepts") or {}
    teams = concepts.get("teams") or {}
    for key, label in (("0", "TEAM A"), ("1", "TEAM B")):
        s = teams.get(key)
        if not s:
            continue
        lines.append(f"\n{label} SHAPE")
        lines.append(f"- players measured: {s['n']}")
        lines.append(f"- centroid: x={s['centroid'][0]} m, y={s['centroid'][1]} m")
        lines.append(f"- line height (centroid x): {s['line_height_m']} m "
                     f"(higher x = further up the pitch toward team B's goal)")
        lines.append(f"- compactness: {s['compactness_m']} m mean distance to centroid "
                     f"(SMALLER = MORE compact / tighter; larger = more spread out)")
        lines.append(f"- width across pitch: {s['width_m']} m (larger = more spread laterally)")
        lines.append(f"- depth along pitch: {s['depth_m']} m (larger = more stretched goal-to-goal)")

    control = concepts.get("control") or {}
    if control:
        lines.append("\nPITCH CONTROL (share of pitch nearest to each team)")
        lines.append(f"- team A: {control.get('0')}%")
        lines.append(f"- team B: {control.get('1')}%")

    # Precomputed comparisons. A small local model reliably gets the *direction*
    # right once it is stated, but still slips on subtraction (it reported a
    # 14.48 - 11.0 gap as 3.44). Arithmetic is free and exact in Python, so do
    # it here rather than asking the model to.
    a, b = teams.get("0"), teams.get("1")
    if a and b:
        lines.append("\nDIRECT COMPARISONS (computed exactly, use these rather than doing arithmetic)")
        comp_a, comp_b = a["compactness_m"], b["compactness_m"]
        tighter = "team A" if comp_a < comp_b else "team B"
        lines.append(f"- more compact (tighter): {tighter}, by {abs(round(comp_a - comp_b, 2))} m")
        lh_a, lh_b = a["line_height_m"], b["line_height_m"]
        higher = "team A" if lh_a > lh_b else "team B"
        lines.append(f"- higher line (further up the pitch): {higher}, by {abs(round(lh_a - lh_b, 2))} m")
        w_a, w_b = a["width_m"], b["width_m"]
        wider = "team A" if w_a > w_b else "team B"
        lines.append(f"- wider across the pitch: {wider}, by {abs(round(w_a - w_b, 2))} m")

    if concepts.get("pressing_m") is not None:
        lines.append(
            f"\nPRESSING\n- mean distance from each player to nearest opponent: "
            f"{concepts['pressing_m']} m (smaller means tighter engagement)"
        )

    # Individual positions last: useful for "where is X" questions, and the
    # model can count them, but they are the bulkiest part of the block.
    lines.append("\nINDIVIDUAL PLAYER POSITIONS (track id, team, x, y in metres)")
    for p in sorted(outfield, key=lambda q: (q.get("team", -1), q.get("x", 0))):
        t = {0: "A", 1: "B"}.get(p.get("team"), "?")
        lines.append(f"- id {p['id']} team {t}: x={p['x']}, y={p['y']}")
    for p in players:
        if p.get("role") in ("goalkeeper", "referee"):
            lines.append(f"- {p['role']} id {p['id']}: x={p['x']}, y={p['y']}")

    block = ("FACTS (measured from video by the vision pipeline — the ONLY "
             "facts you may state about this moment):\n" + "\n".join(lines))
    return block, True


def _ollama_model() -> Optional[str]:
    """First available preferred model from the local Ollama instance."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            have = {m["name"] for m in json.load(r).get("models", [])}
    except Exception:
        return None
    for m in OLLAMA_MODELS:
        if m in have:
            return m
    # Any instruct/chat model rather than the embedder.
    for m in sorted(have):
        if "embed" not in m:
            return m
    return None


def _ask_ollama(question: str, facts: str) -> Tuple[Optional[str], Optional[str]]:
    model = _ollama_model()
    if not model:
        return None, "No local Ollama model available"
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{facts}\n\nQUESTION: {question}"},
        ],
    }
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return (data.get("message") or {}).get("content", "").strip(), None
    except Exception as exc:
        return None, f"Ollama request failed: {exc}"


def _ask_anthropic(question: str, facts: str) -> Tuple[Optional[str], Optional[str]]:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, "ANTHROPIC_API_KEY not set"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"{facts}\n\nQUESTION: {question}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip(), None
    except Exception as exc:
        return None, f"Anthropic request failed: {exc}"


def answer(question: str, pitch: Dict, backend: str = "auto") -> Dict:
    """
    Answer a question using only the current frame's measurements.

    Returns {ok, answer, facts, backend, error}. `facts` is always returned so
    the caller can show exactly what the model was allowed to see — the
    provenance is as much the point as the answer.
    """
    question = (question or "").strip()
    if not question:
        return {"ok": False, "error": "Ask a question first."}

    facts, has_data = build_facts_block(pitch)
    if not has_data:
        return {
            "ok": True,
            "answer": "There is nothing measured yet, so I cannot answer that. "
                      "Start an analysis session and wait for players to be detected.",
            "facts": facts,
            "backend": "none",
            "grounded": False,
        }

    order = ["anthropic", "ollama"] if backend == "auto" and os.environ.get("ANTHROPIC_API_KEY") \
        else ["ollama", "anthropic"] if backend == "auto" else [backend]

    errors = []
    for b in order:
        text, err = (_ask_ollama if b == "ollama" else _ask_anthropic)(question, facts)
        if text:
            return {"ok": True, "answer": text, "facts": facts,
                    "backend": b, "grounded": True}
        errors.append(f"{b}: {err}")

    return {"ok": False, "error": "; ".join(errors), "facts": facts, "grounded": True}


if __name__ == "__main__":
    # Exercise the grounding layer without a model, then with one if available.
    demo_pitch = {
        "calibrated": True, "n_keypoints": 9,
        "counts": {"player": 20, "goalkeeper": 1, "referee": 3, "ball": 1},
        "team_counts": {"0": 10, "1": 10, "-1": 0},
        "ball": {"x": 52.4, "y": 33.1},
        "players": [
            {"id": i, "role": "player", "team": 0 if i < 10 else 1,
             "x": 30 + i * 2, "y": 20 + (i % 5) * 8}
            for i in range(20)
        ],
        "concepts": {
            "teams": {
                "0": {"n": 10, "centroid": [39.5, 34.0], "compactness_m": 14.6,
                      "width_m": 31.9, "depth_m": 18.0, "line_height_m": 39.5},
                "1": {"n": 10, "centroid": [59.5, 34.0], "compactness_m": 16.6,
                      "width_m": 45.5, "depth_m": 22.0, "line_height_m": 59.5},
            },
            "control": {"0": 45.5, "1": 54.5},
            "pressing_m": 5.3,
        },
    }
    block, ok = build_facts_block(demo_pitch)
    print(f"has_data={ok}  block_lines={len(block.splitlines())}")
    print("\n".join(block.splitlines()[:12]))
    print("...")
    print("\nlocal model:", _ollama_model())
    for q in ["Which team is pressing higher up the pitch?",
              "What was the score at half time?"]:
        r = answer(q, demo_pitch)
        print(f"\nQ: {q}\nbackend={r.get('backend')} ok={r.get('ok')}")
        print("A:", (r.get('answer') or r.get('error'))[:400])
