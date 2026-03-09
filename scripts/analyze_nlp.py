#!/usr/bin/env python3
"""NLP analysis of negotiation game logs.

Three-stage pipeline:
  1. parse     — Deterministic extraction of game logs into structured JSON + CSV
  2. annotate  — LLM behavioral annotation per game (cached)
  3. query     — Natural language questions answered with citations

Usage:
  python scripts/analyze_nlp.py parse results/experiments/run_A [results/experiments/run_B ...] --name my_analysis
  python scripts/analyze_nlp.py annotate results/analysis/my_analysis --api-key $KEY [--max-games 10]
  python scripts/analyze_nlp.py query results/analysis/my_analysis "Does the profiler help?" --api-key $KEY

Output goes to results/analysis/<name>/ with provenance tracking.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_BASE = BASE_DIR / "results" / "analysis"

DEFAULT_MODEL = "api-gpt-oss-120b"
DEFAULT_API_BASE = "https://tritonai-api.ucsd.edu/v1"


# ── Game discovery ────────────────────────────────────────────────────

def discover_games(run_dirs):
    """Find all game directories (containing results.json) under given run dirs.
    Skips framework_logs subdirectories."""
    games = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"  Warning: {run_dir} not found, skipping", file=sys.stderr)
            continue
        for results_json in sorted(run_path.rglob("results.json")):
            if "framework_logs" in results_json.parts:
                continue
            games.append(results_json.parent)
    return games


def experiment_name_for(game_dir, run_dirs):
    """Determine which experiment a game belongs to."""
    for run_dir in run_dirs:
        try:
            game_dir.relative_to(Path(run_dir))
            return Path(run_dir).name
        except ValueError:
            continue
    return "unknown"


# ── Interaction.log parsing ───────────────────────────────────────────

TURN_KEYS = [
    "message", "player answer", "newly proposed trade",
    "my resources", "my goals", "reason", "proposal count",
]


def parse_interaction_log(log_path):
    """Parse interaction.log into list of turn dicts with full conversation."""
    with open(log_path) as f:
        lines = f.readlines()

    turns = []
    current_turn = None
    current_key = None
    past_header = False  # skip "Game Settings" preamble

    for line in lines:
        stripped = line.rstrip("\n")

        # Section separator ------ marks end of header / between sections
        if re.match(r"^-{5,}", stripped):
            past_header = True
            continue

        if not past_header:
            continue

        # Iteration header
        m = re.match(r"^Current Iteration:\s*(.+)$", stripped)
        if m:
            val = m.group(1).strip()
            if val == "END":
                if current_turn:
                    turns.append(current_turn)
                current_turn = None
                break
            if val == "START":
                continue
            if current_turn:
                turns.append(current_turn)
            current_turn = {"iteration": int(val)}
            current_key = None
            continue

        # Turn number
        m = re.match(r"^Turn:\s*(\d+|None)$", stripped)
        if m and current_turn is not None:
            v = m.group(1)
            current_turn["turn"] = int(v) if v != "None" else None
            continue

        # Key-value pairs (try longest key first to avoid partial matches)
        if current_turn is not None:
            matched = False
            for key in sorted(TURN_KEYS, key=len, reverse=True):
                prefix = f"{key}:"
                if stripped.startswith(prefix):
                    current_key = key
                    current_turn[key] = stripped[len(prefix):].strip()
                    matched = True
                    break
            if not matched and current_key and stripped.strip():
                # Multi-line value continuation
                current_turn[current_key] += " " + stripped.strip()

    if current_turn:
        turns.append(current_turn)
    return turns


# ── Helpers ───────────────────────────────────────────────────────────

def extract_price(trade_str):
    """Extract ZUP amount from trade string like 'Player BLUE Gives ZUP: 12'."""
    if not trade_str or trade_str.strip() == "NONE":
        return None
    m = re.search(r"Player BLUE Gives ZUP:\s*(\d+)", trade_str)
    return int(m.group(1)) if m else None


def find_latest_framework_log(game_dir):
    """Return the latest framework_logs/<timestamp>/ directory."""
    fw_dir = game_dir / "framework_logs"
    if not fw_dir.exists():
        return None
    subdirs = sorted(
        [d for d in fw_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def parse_game_state_profiler_logs(game_state_path):
    """Extract complete profiler_logs from game_state.json.

    Returns list of dicts: {opponent_message: str, profiler_output: str}
    Each entry is one profiler call (one per our agent's turn in profiler mode).
    """
    with open(game_state_path) as f:
        state = json.load(f)

    # profiler_logs lives under the profiler player entry in the players list
    raw_logs = state.get("profiler_logs", [])
    if not raw_logs:
        players = state.get("players", [])
        for player in (players if isinstance(players, list) else []):
            if isinstance(player, dict) and "profiler_logs" in player:
                raw_logs = player["profiler_logs"]
                break
    parsed = []
    for entry in raw_logs:
        if isinstance(entry, list) and len(entry) == 2:
            opponent_msg = entry[0]
            profiler_out = entry[1]
            # opponent_msg is either a dict {role, content} or a string
            if isinstance(opponent_msg, dict):
                opponent_msg = opponent_msg.get("content", "")
            parsed.append({
                "opponent_message": str(opponent_msg),
                "profiler_output": str(profiler_out),
            })
    return parsed


# ── Filtering ────────────────────────────────────────────────────────

def parse_filters(filter_strings):
    """Parse filter strings like 'mode=profiler' into {key: [values]} dict."""
    filters = {}
    if not filter_strings:
        return filters
    for fs in filter_strings:
        if "=" not in fs:
            print(f"  Warning: ignoring malformed filter '{fs}' (expected key=value)",
                  file=sys.stderr)
            continue
        key, val = fs.split("=", 1)
        # Support comma-separated values for OR: persona=hardball,friendly
        filters[key.strip()] = [v.strip() for v in val.split(",")]
    return filters


def filter_games(games, filters):
    """Filter list of game dicts by key=value criteria.

    Each filter key must match one of the comma-separated values.
    All filters must match (AND across keys, OR within values).
    """
    if not filters:
        return games
    filtered = []
    for g in games:
        match = True
        for key, allowed_values in filters.items():
            game_val = str(g.get(key, ""))
            if game_val not in allowed_values:
                match = False
                break
        if match:
            filtered.append(g)
    return filtered


def parse_game_log_setup(game_log_path):
    """Extract model names and persona prompt from game.log SETUP section."""
    with open(game_log_path) as f:
        text = f.read()
    setup = {}
    for m in re.finditer(
        r"^\s+(Our (?:agent|negotiator)|Profiler brain|Opponent):\s+(.+)$",
        text, re.MULTILINE,
    ):
        setup[m.group(1).strip()] = m.group(2).strip()

    m = re.search(
        r"Opponent persona prompt:\n(.*?)(?=\n={10,}|\n\n[A-Z])", text, re.DOTALL
    )
    if m:
        setup["persona_prompt"] = m.group(1).strip().strip('"')
    return setup


def parse_game_log_profiler_blocks(game_log_path):
    """Extract [PROFILER ANALYSIS] blocks from game.log as raw text."""
    with open(game_log_path) as f:
        text = f.read()
    blocks = []
    for m in re.finditer(
        r"\[PROFILER ANALYSIS[^\]]*\]\n(.*?)(?=\n\[Player|\n={10,}|\Z)",
        text, re.DOTALL,
    ):
        blocks.append(m.group(1).strip())
    return blocks


# ── Single game parsing ──────────────────────────────────────────────

def make_game_id(game_dir, exp_name):
    """Deterministic game ID from path: experiment/scenario/persona/run/mode."""
    parts = game_dir.parts
    # path ends with .../scenario_X/vs_Y/run_N/mode/
    mode = parts[-1]
    run = parts[-2]
    persona = parts[-3]
    scenario = parts[-4]
    return f"{exp_name}/{scenario}/{persona}/{run}/{mode}"


def parse_single_game(game_dir, exp_name):
    """Parse one game directory into structured data.

    Returns (game_row, turn_rows, transcript_text, game_log_text).
    """
    game_dir = Path(game_dir)

    with open(game_dir / "results.json") as f:
        results = json.load(f)

    game_id = make_game_id(game_dir, exp_name)

    # Full conversation from interaction.log
    fw_dir = find_latest_framework_log(game_dir)
    turns_data = []
    transcript = ""
    if fw_dir:
        ilog = fw_dir / "interaction.log"
        if ilog.exists():
            turns_data = parse_interaction_log(ilog)
            with open(ilog) as f:
                transcript = f.read()

    # Setup metadata from game.log
    game_log = game_dir / "game.log"
    setup_meta = {}
    game_log_text = ""
    if game_log.exists():
        setup_meta = parse_game_log_setup(game_log)
        with open(game_log) as f:
            game_log_text = f.read()

    # Profiler data: prefer game_state.json (complete) over game.log (last block only)
    profiler_logs_full = []
    profiler_blocks = []
    if fw_dir:
        gs_path = fw_dir / "game_state.json"
        if gs_path.exists():
            profiler_logs_full = parse_game_state_profiler_logs(gs_path)
            profiler_blocks = [p["profiler_output"] for p in profiler_logs_full]
    if not profiler_blocks and game_log.exists():
        # Fallback to game.log (only captures last block)
        profiler_blocks = parse_game_log_profiler_blocks(game_log)

    self_role = results.get("self_role", "seller")
    our_player = "BLUE" if self_role == "buyer" else "RED"

    # First and last offer prices
    first_offer = None
    last_offer = None
    for t in turns_data:
        price = extract_price(t.get("newly proposed trade", ""))
        if price is not None:
            if first_offer is None:
                first_offer = price
            last_offer = price

    zopa = (results.get("buyer_wtp") or 0) - (results.get("seller_cost") or 0)
    our_out = (
        results.get("seller_outcome") if self_role == "seller"
        else results.get("buyer_outcome")
    )
    surplus_pct = (
        round(our_out / zopa * 100, 1)
        if zopa > 0 and our_out is not None
        else None
    )

    game_row = {
        "game_id": game_id,
        "experiment": exp_name,
        "scenario": results.get("scenario"),
        "seller_cost": results.get("seller_cost"),
        "buyer_wtp": results.get("buyer_wtp"),
        "zopa": zopa,
        "mode": results.get("mode"),
        "persona": results.get("persona"),
        "run": results.get("run"),
        "self_role": self_role,
        "final_response": results.get("final_response"),
        "deal_reached": results.get("deal_reached"),
        "deal_price": results.get("deal_price"),
        "seller_outcome": results.get("seller_outcome"),
        "buyer_outcome": results.get("buyer_outcome"),
        "our_outcome": our_out,
        "surplus_pct": surplus_pct,
        "num_turns": results.get("num_turns"),
        "first_offer_price": first_offer,
        "last_offer_price": last_offer,
        "our_model": setup_meta.get("Our agent") or setup_meta.get("Our negotiator"),
        "opponent_model": setup_meta.get("Opponent"),
        "profiler_model": setup_meta.get("Profiler brain"),
        "num_profiler_calls": len(profiler_logs_full),
        "persona_prompt": setup_meta.get("persona_prompt", ""),
        "game_dir": str(game_dir),
    }

    # Turn rows
    turn_rows = []
    profiler_idx = 0
    for t in turns_data:
        turn_num = t.get("turn")
        if turn_num is None:
            continue
        player = "RED" if turn_num == 0 else "BLUE"
        role = "seller" if turn_num == 0 else "buyer"
        is_ours = player == our_player

        profiler_text = None
        if (
            is_ours
            and results.get("mode") == "profiler"
            and profiler_idx < len(profiler_blocks)
        ):
            profiler_text = profiler_blocks[profiler_idx]
            profiler_idx += 1

        turn_rows.append({
            "game_id": game_id,
            "iteration": t.get("iteration"),
            "turn": turn_num,
            "player": player,
            "role": role,
            "is_our_agent": is_ours,
            "player_answer": t.get("player answer"),
            "proposed_price": extract_price(t.get("newly proposed trade", "")),
            "proposal_count": t.get("proposal count"),
            "reason": t.get("reason"),
            "message": t.get("message"),
            "profiler_analysis": profiler_text,
        })

    return game_row, turn_rows, transcript, game_log_text, profiler_logs_full


# ── CSV export ────────────────────────────────────────────────────────

GAME_FIELDS = [
    "game_id", "experiment", "scenario", "seller_cost", "buyer_wtp", "zopa",
    "mode", "persona", "run", "self_role",
    "final_response", "deal_reached", "deal_price",
    "seller_outcome", "buyer_outcome", "our_outcome", "surplus_pct",
    "num_turns", "first_offer_price", "last_offer_price",
    "our_model", "opponent_model", "profiler_model", "num_profiler_calls",
    "persona_prompt", "game_dir",
]

TURN_FIELDS = [
    "game_id", "iteration", "turn", "player", "role", "is_our_agent",
    "player_answer", "proposed_price", "proposal_count",
    "reason", "message", "profiler_analysis",
]


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── LLM client ────────────────────────────────────────────────────────

def get_client(api_key, base_url=DEFAULT_API_BASE):
    import openai
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def llm_call(client, model, system, user, temperature=0.3):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ── Annotate ──────────────────────────────────────────────────────────

ANNOTATE_SYSTEM = """\
You are an expert negotiation analyst. You analyze game logs from automated \
negotiation experiments between LLM agents.

Experimental setup:
- Two agents negotiate the price of an object. The seller (Player RED) has a \
production cost; the buyer (Player BLUE) has a willingness to pay (WTP). \
The ZOPA (zone of possible agreement) = WTP - cost.
- Three modes are tested: "baseline" (static weak agent), "profiler" (weak \
agent augmented with a strategy-detection profiler), "compare" (static strong \
agent as upper-bound reference).
- The opponent is assigned one of 5 personas: neutral, hardball, friendly, \
sycophant, stalling.

IMPORTANT — flag these as high-significance behaviors whenever they occur:
- **Economically irrational deals**: seller_outcome < 0 means the seller sold \
below their own production cost (losing money). buyer_outcome < 0 means the \
buyer paid more than their willingness to pay. These are always noteworthy.
- **Surplus > 100%**: surplus_pct > 100 means one party captured more than the \
entire ZOPA, which is only possible if the other party is losing money on the \
deal. Flag and explain why.
- **Persona-inconsistent behavior**: a "hardball" agent making generous offers, \
a "friendly" agent anchoring aggressively, a "stalling" agent accepting \
immediately, etc. When the opponent's actual behavior contradicts their assigned \
persona, flag it — this reveals the opponent LLM failing to follow instructions.
- **Information leaks**: an agent revealing its private cost, WTP, or internal \
reasoning (e.g., <reason> tags) in its public message to the other player.

Given a game transcript and metadata, provide structured behavioral analysis.
Cite exact quotes from the transcript as evidence.

Output ONLY valid JSON (no markdown fences) with this schema:
{
  "game_summary": "<1-2 sentence summary>",
  "notable_behaviors": [
    {
      "iteration": <int>,
      "actor": "our_agent | opponent | profiler",
      "behavior_type": "<anchoring | concession | bluffing | information_leak | \
stalling | accepting_too_early | accepting_too_late | rejecting_prematurely | \
counter_strategy | price_manipulation | rapport_building | other>",
      "description": "<what happened>",
      "evidence": "<exact quote from transcript>",
      "significance": "high | medium | low"
    }
  ],
  "profiler_assessment": {
    "present": <true/false>,
    "accuracy": "<accurate | partially_accurate | inaccurate | n/a>",
    "strategy_detected": "<what the profiler recommended>",
    "actual_opponent_behavior": "<what the opponent actually did>",
    "did_counter_strategy_help": <true/false/null>,
    "notes": "<analysis>"
  },
  "tactical_patterns": ["<pattern descriptions>"],
  "weaknesses": ["<identified weaknesses of our agent>"],
  "strengths": ["<identified strengths of our agent>"],
  "fairness_score": <-1.0 to 1.0, 0=even ZOPA split, positive=favors our agent>
}"""


def annotate_game(client, model, game_full):
    """Call LLM to annotate one game. Returns parsed annotation dict."""
    user = (
        f"Game ID: {game_full['game_id']}\n"
        f"Mode: {game_full.get('mode')}  |  Persona: {game_full.get('persona')}\n"
        f"Our role: {game_full.get('self_role')}  |  "
        f"Seller cost: {game_full.get('seller_cost')}  |  "
        f"Buyer WTP: {game_full.get('buyer_wtp')}  |  ZOPA: {game_full.get('zopa')}\n"
        f"Outcome: {game_full.get('final_response')}  |  "
        f"Deal price: {game_full.get('deal_price')}  |  "
        f"Seller: {game_full.get('seller_outcome')}  |  "
        f"Buyer: {game_full.get('buyer_outcome')}  |  "
        f"Turns: {game_full.get('num_turns')}\n\n"
        f"=== FULL TRANSCRIPT (interaction.log) ===\n"
        f"{game_full.get('transcript', '(not available)')}\n\n"
        f"=== GAME LOG (includes profiler analysis if present) ===\n"
        f"{game_full.get('game_log_text', '(not available)')}"
    )

    # Include complete profiler logs from game_state.json if available
    prof_logs = game_full.get("profiler_logs", [])
    if prof_logs:
        prof_parts = ["\n\n=== COMPLETE PROFILER LOGS (all calls, from game_state.json) ==="]
        for i, pl in enumerate(prof_logs, 1):
            prof_parts.append(f"\n--- Profiler Call {i}/{len(prof_logs)} ---")
            prof_parts.append(f"Opponent said: {pl['opponent_message'][:300]}")
            prof_parts.append(f"Profiler response:\n{pl['profiler_output']}")
        user += "\n".join(prof_parts)

    raw = llm_call(client, model, ANNOTATE_SYSTEM, user)

    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}


# ── Query ─────────────────────────────────────────────────────────────

QUERY_SYSTEM = """\
You are an expert negotiation research analyst. You have access to structured \
data and transcripts from automated negotiation experiments between LLM agents.

Experimental setup:
- A weaker agent (Llama-4-Scout) negotiates against a stronger opponent \
(GPT-OSS-120B) in a buy/sell game.
- Three modes: "baseline" (static weak agent), "profiler" (weak agent with \
real-time strategy-detection profiler), "compare" (static strong agent, \
upper-bound reference).
- Five opponent personas: neutral, hardball, friendly, sycophant, stalling.
- Multiple price scenarios with varying ZOPA (zone of possible agreement).
- surplus_pct = our_outcome / ZOPA * 100 (how much of available surplus \
our agent captured).

IMPORTANT — always check for and highlight these phenomena:
- **Economically irrational deals**: seller_outcome < 0 (seller sold below \
cost) or buyer_outcome < 0 (buyer overpaid). These are LLM agents and CAN \
behave irrationally — this is a key research finding, not a data error.
- **Surplus > 100%**: one party captured more than the full ZOPA, meaning the \
other party lost money. Explain which party acted irrationally and why.
- **Persona violations**: an agent behaving contrary to its assigned persona \
(e.g., "hardball" seller offering below cost, "stalling" agent accepting \
immediately). This reveals the opponent LLM failing to follow persona instructions.
- **Information leaks**: agents revealing private values (cost, WTP) or internal \
reasoning in public messages.

When answering:
1. EVERY quote or specific claim MUST include the game tag shown in the data \
(e.g., [G03 s12v19/hardball/profiler, iter 2]). Never present a quote without \
identifying which game it came from. The data prefixes each line with a tag \
like [G03] — use it.
2. Quote exact text from transcripts when relevant.
3. Distinguish patterns (across multiple games) from one-off observations.
4. If data is insufficient, say so.
5. Use numbers: deal rates, average surplus_pct, price comparisons.
6. Always flag irrational behavior and persona violations — these are among \
the most important findings in LLM negotiation research."""


def build_query_context(games_full, annotations):
    """Build context string for the query LLM from games + annotations."""
    parts = []

    # Summary statistics
    modes = set(g.get("mode") for g in games_full)
    personas = set(g.get("persona") for g in games_full)
    scenarios = set(g.get("scenario") for g in games_full)
    parts.append(
        f"Dataset: {len(games_full)} games across "
        f"{len(scenarios)} scenarios, {len(modes)} modes ({', '.join(sorted(modes))}), "
        f"{len(personas)} personas ({', '.join(sorted(personas))})\n"
    )

    for gi, g in enumerate(games_full):
        gid = g["game_id"]
        # Short tag: [G01 s10v20/hardball/baseline]
        short_id = f"G{gi+1:02d}"
        scenario = g.get("scenario", "?")
        persona = g.get("persona", "?")
        mode = g.get("mode", "?")
        tag = f"[{short_id} {scenario}/{persona}/{mode}]"

        parts.append(f"\n{'=' * 70}")
        parts.append(f"{tag} GAME: {gid}")
        parts.append(
            f"{tag} Mode: {mode}  |  Scenario: {scenario}  |  "
            f"Persona: {persona}  |  Our role: {g.get('self_role')}"
        )
        parts.append(
            f"{tag} Costs: seller={g.get('seller_cost')}, buyer_wtp={g.get('buyer_wtp')}, "
            f"ZOPA={g.get('zopa')}"
        )
        parts.append(
            f"{tag} Outcome: {g.get('final_response')}  |  Price: {g.get('deal_price')}  |  "
            f"Seller: {g.get('seller_outcome')}  |  Buyer: {g.get('buyer_outcome')}  |  "
            f"surplus_pct: {g.get('surplus_pct')}  |  Turns: {g.get('num_turns')}"
        )

        # Include turns
        for t in g.get("turns", []):
            label = "OUR" if t.get("is_our_agent") else "OPP"
            price_str = f"  price={t['proposed_price']}" if t.get("proposed_price") else ""
            parts.append(
                f"{tag}   [iter {t['iteration']}, {t['player']} {label}] "
                f"{t.get('player_answer', '?')}{price_str}"
            )
            if t.get("reason"):
                parts.append(f"{tag}     reason: {t['reason']}")
            if t.get("message"):
                parts.append(f"{tag}     message: {t['message']}")
            if t.get("profiler_analysis"):
                # Truncate long profiler blocks
                pa = t["profiler_analysis"]
                if len(pa) > 500:
                    pa = pa[:500] + "..."
                parts.append(f"{tag}     [PROFILER]: {pa}")

        # Include profiler logs if available
        prof_logs = g.get("profiler_logs", [])
        if prof_logs:
            parts.append(f"{tag}   [PROFILER CALLS: {len(prof_logs)} total]")
            for pi, pl in enumerate(prof_logs, 1):
                opp_msg = pl["opponent_message"][:200]
                prof_out = pl["profiler_output"][:400]
                parts.append(f"{tag}     Call {pi}: opp={opp_msg}")
                parts.append(f"{tag}       -> {prof_out}")

        # Include annotation summary if available
        ann = annotations.get(gid)
        if ann and not ann.get("parse_error"):
            parts.append(f"{tag}   [ANNOTATION] {ann.get('game_summary', '')}")
            for nb in ann.get("notable_behaviors", []):
                parts.append(
                    f"{tag}     - [{nb.get('significance')}] iter {nb.get('iteration')}: "
                    f"{nb.get('description')}"
                )
            pa = ann.get("profiler_assessment", {})
            if pa.get("present"):
                parts.append(
                    f"{tag}     Profiler accuracy: {pa.get('accuracy')} — "
                    f"{pa.get('notes', '')}"
                )

    return "\n".join(parts)


# ── Subcommands ───────────────────────────────────────────────────────

def cmd_parse(args):
    run_dirs = [Path(d) for d in args.run_dirs]
    game_dirs = discover_games(run_dirs)

    if not game_dirs:
        print("No games found.")
        sys.exit(1)
    if args.max_games:
        game_dirs = game_dirs[: args.max_games]

    name = args.name or datetime.now().strftime("analysis_%Y%m%d_%H%M%S")
    out_dir = ANALYSIS_BASE / name
    os.makedirs(out_dir, exist_ok=True)

    print(f"Parsing {len(game_dirs)} games → {out_dir}")

    all_games = []
    all_game_rows = []
    all_turn_rows = []
    experiments = set()

    for i, gd in enumerate(game_dirs):
        exp = experiment_name_for(gd, run_dirs)
        try:
            game_row, turn_rows, transcript, gl_text, prof_logs = parse_single_game(gd, exp)
            all_game_rows.append(game_row)
            all_turn_rows.extend(turn_rows)
            all_games.append({
                **game_row,
                "turns": turn_rows,
                "transcript": transcript,
                "game_log_text": gl_text,
                "profiler_logs": prof_logs,
            })
            experiments.add(exp)
        except Exception as e:
            print(f"  Warning: {gd}: {e}", file=sys.stderr)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(game_dirs)} parsed")

    # Manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "name": name,
        "experiments": sorted(experiments),
        "experiment_paths": [str(d) for d in run_dirs],
        "num_games": len(all_game_rows),
        "num_turns": len(all_turn_rows),
        "scenarios": sorted(set(g["scenario"] for g in all_game_rows if g.get("scenario"))),
        "modes": sorted(set(g["mode"] for g in all_game_rows if g.get("mode"))),
        "personas": sorted(set(g["persona"] for g in all_game_rows if g.get("persona"))),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Full parsed data (JSON) — includes transcripts for annotate/query
    with open(out_dir / "parsed_games.json", "w") as f:
        json.dump(all_games, f, indent=2)

    # CSVs
    write_csv(out_dir / "games.csv", GAME_FIELDS, all_game_rows)
    write_csv(out_dir / "turns.csv", TURN_FIELDS, all_turn_rows)

    print(f"\nDone: {len(all_game_rows)} games, {len(all_turn_rows)} turns")
    print(f"  {out_dir / 'manifest.json'}")
    print(f"  {out_dir / 'parsed_games.json'}")
    print(f"  {out_dir / 'games.csv'}")
    print(f"  {out_dir / 'turns.csv'}")
    print(f"  Experiments: {', '.join(sorted(experiments))}")
    print(f"  Modes:       {', '.join(manifest['modes'])}")
    print(f"  Personas:    {', '.join(manifest['personas'])}")


def cmd_annotate(args):
    analysis_dir = Path(args.analysis_dir)
    parsed_path = analysis_dir / "parsed_games.json"
    if not parsed_path.exists():
        print(f"No parsed_games.json in {analysis_dir}. Run 'parse' first.")
        sys.exit(1)

    with open(parsed_path) as f:
        games = json.load(f)

    filters = parse_filters(getattr(args, "filter", None))
    if filters:
        games = filter_games(games, filters)
        print(f"Filter applied: {len(games)} games match {filters}")

    if args.max_games:
        games = games[: args.max_games]

    # Load existing annotations
    ann_path = analysis_dir / "annotations.json"
    annotations = {}
    if ann_path.exists():
        with open(ann_path) as f:
            annotations = json.load(f)

    client = get_client(args.api_key, args.base_url)
    model = args.model

    print(f"Annotating {len(games)} games with {model}...")

    for i, g in enumerate(games):
        gid = g["game_id"]
        if gid in annotations and not args.force:
            print(f"  [{i+1}/{len(games)}] {gid} — cached")
            continue

        print(f"  [{i+1}/{len(games)}] {gid} — annotating...", end="", flush=True)
        try:
            ann = annotate_game(client, model, g)
            annotations[gid] = ann

            # Save incrementally
            with open(ann_path, "w") as f:
                json.dump(annotations, f, indent=2)

            if ann.get("parse_error"):
                print(" PARSE ERROR")
            else:
                print(f" OK — {ann.get('game_summary', '')[:80]}")
        except Exception as e:
            print(f" ERROR: {e}")

        # Basic rate limiting
        time.sleep(0.1)

    print(f"\nAnnotations: {ann_path} ({len(annotations)} games)")


def cmd_query(args):
    analysis_dir = Path(args.analysis_dir)
    parsed_path = analysis_dir / "parsed_games.json"
    if not parsed_path.exists():
        print(f"No parsed_games.json in {analysis_dir}. Run 'parse' first.")
        sys.exit(1)

    with open(parsed_path) as f:
        games = json.load(f)

    filters = parse_filters(getattr(args, "filter", None))
    if filters:
        games = filter_games(games, filters)
        print(f"Filter applied: {len(games)} games match {filters}")

    if args.max_games:
        games = games[: args.max_games]

    # Load annotations if available
    ann_path = analysis_dir / "annotations.json"
    annotations = {}
    if ann_path.exists():
        with open(ann_path) as f:
            annotations = json.load(f)

    client = get_client(args.api_key, args.base_url)
    model = args.model
    question = args.question

    print(f"Querying {len(games)} games with {model}...")
    print(f"Q: {question}")
    print("=" * 80)

    context = build_query_context(games, annotations)
    user_prompt = (
        f"Here is data from {len(games)} negotiation games:\n\n"
        f"{context}\n\n"
        f"{'=' * 40}\n"
        f"QUESTION: {question}\n\n"
        f"Provide a thorough analysis with specific citations."
    )

    answer = llm_call(client, model, QUERY_SYSTEM, user_prompt, temperature=0.3)
    print(answer)

    # Save query
    query_dir = analysis_dir / "queries"
    os.makedirs(query_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_path = query_dir / f"query_{ts}.json"
    with open(query_path, "w") as f:
        json.dump({
            "question": question,
            "answer": answer,
            "model": model,
            "timestamp": ts,
            "num_games": len(games),
            "game_ids": [g["game_id"] for g in games],
            "has_annotations": bool(annotations),
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Saved: {query_path}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NLP analysis of negotiation game logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # parse
    p = sub.add_parser("parse", help="Parse game logs into structured JSON + CSV")
    p.add_argument("run_dirs", nargs="+", help="Experiment run directories")
    p.add_argument("--name", default=None, help="Output folder name (default: timestamped)")
    p.add_argument("--max-games", type=int, default=None, help="Limit number of games")

    # annotate
    p = sub.add_parser("annotate", help="LLM-annotate games with behavioral analysis")
    p.add_argument("analysis_dir", help="Path to analysis output directory")
    p.add_argument("--api-key", required=True, help="API key for LLM")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=DEFAULT_API_BASE, help="API base URL")
    p.add_argument("--max-games", type=int, default=None, help="Limit number of games")
    p.add_argument("--filter", nargs="+", metavar="KEY=VAL",
                   help="Filter games (e.g. mode=profiler persona=hardball,friendly)")
    p.add_argument("--force", action="store_true", help="Re-annotate cached games")

    # query
    p = sub.add_parser("query", help="Natural language query over game data")
    p.add_argument("analysis_dir", help="Path to analysis output directory")
    p.add_argument("question", help="Natural language question")
    p.add_argument("--api-key", required=True, help="API key for LLM")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=DEFAULT_API_BASE, help="API base URL")
    p.add_argument("--max-games", type=int, default=None, help="Limit number of games")
    p.add_argument("--filter", nargs="+", metavar="KEY=VAL",
                   help="Filter games (e.g. mode=profiler persona=hardball,friendly)")

    args = parser.parse_args()

    if args.command == "parse":
        cmd_parse(args)
    elif args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
