"""Unified N-run experiment runner for baseline and profiler comparisons.

Directory layout per run:
    results/experiments/run_TIMESTAMP/
        config.json                     # experiment-wide config
        summary.log                     # comparison table across all scenarios
        vs_<persona>/
            run_<N>/
                baseline/
                    game.log            # full human-readable log
                    results.json        # numeric metrics
                profiler/
                    game.log
                    results.json

Usage:
    python scripts/run_experiment.py --mode both --num-runs 3
    python scripts/run_experiment.py --mode baseline --num-runs 5 --role buyer
    python scripts/run_experiment.py --mode profiler --num-runs 1
"""

import argparse
import json
import sys
import os
import traceback
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
from games.buy_sell_game.game import BuySellGame
from profiler_agent import ProfilerAgent
from constants import OPPONENT_PERSONAS

# ── Models ──────────────────────────────────────────────────────────
SELF_MODEL = "api-llama-4-scout"        # baseline negotiator
NEGOTIATOR_MODEL = "api-llama-4-scout"  # profiler negotiator
PROFILER_MODEL = "api-gpt-oss-120b"     # profiler brain
OPPONENT_MODEL = "api-gpt-oss-120b"     # opponent
COMPARE_MODEL = "api-gpt-oss-120b"      # compare: oss static agent (upper-bound reference)

ITERATIONS = 10
MAX_RETRIES = 3

# 20 (seller_cost, buyer_wtp) scenarios spanning a wide range of ZOPA widths,
# price levels, and deal feasibility. ZOPA = buyer_wtp - seller_cost; negative = no deal.
PRICE_SCENARIOS = [
    # Low price range
    (5,  35),   # ZOPA=30, very low prices
    (10, 50),   # ZOPA=40, wide, low prices
    (12, 22),   # ZOPA=10, narrow, low prices
    (12, 32),   # ZOPA=20, moderate ZOPA, low prices
    # Mid-low range
    (20, 45),   # ZOPA=25
    (25, 75),   # ZOPA=50, very wide
    (28, 38),   # ZOPA=10, narrow
    # Mid range
    (30, 70),   # ZOPA=40, wide
    (35, 65),   # ZOPA=30
    (40, 60),   # ZOPA=20 — original baseline
    (42, 58),   # ZOPA=16, slightly narrow
    (45, 55),   # ZOPA=10, narrow
    (50, 62),   # ZOPA=12, tight
    # High price range
    (60, 100),  # ZOPA=40, wide, high prices
    (70, 90),   # ZOPA=20, high prices
    (75, 85),   # ZOPA=10, narrow, high prices
    (80, 95),   # ZOPA=15, narrow-ish, high prices
    # No-deal scenarios
    (55, 45),   # ZOPA=-10, no deal, moderate prices
    (65, 55),   # ZOPA=-10, no deal, high prices
    (30, 20),   # ZOPA=-10, no deal, low prices
]

BASE_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiments"
)


# ── Logging helpers ──────────────────────────────────────────────────

def _indent(text, prefix="  "):
    """Indent every line of text."""
    return "\n".join(prefix + line for line in text.splitlines())


def write_game_log(log_dir, mode, persona_label, persona_prompt,
                   self_is_seller, run_idx, game, result,
                   profiler_agent=None, config=None):
    """Write game.log (full human-readable log) and results.json to log_dir."""
    if config is None:
        config = {}
    os.makedirs(log_dir, exist_ok=True)

    seller_name = game.players[0].agent_name   # Player RED
    buyer_name  = game.players[1].agent_name   # Player BLUE
    our_name  = seller_name if self_is_seller else buyer_name
    opp_name  = buyer_name  if self_is_seller else seller_name
    our_role  = "seller" if self_is_seller else "buyer"

    lines = []

    # ── SETUP ────────────────────────────────────────────────────────
    lines.append("=" * 80)
    lines.append("SETUP")
    lines.append("=" * 80)
    lines.append(f"Mode:               {mode}")
    lines.append(f"Our role:           {our_role}  (our agent = {our_name})")
    lines.append(f"Opponent role:      {'buyer' if self_is_seller else 'seller'}  (opponent = {opp_name})")
    lines.append(f"Persona (opponent): {persona_label}")
    lines.append(f"Run:                {run_idx}")
    lines.append("")
    lines.append("Models:")
    if mode in ("baseline", "compare"):
        lines.append(f"  Our agent:          {config.get('self_model', 'N/A')}")
    else:
        lines.append(f"  Our negotiator:     {config.get('negotiator_model', 'N/A')}")
        lines.append(f"  Profiler brain:     {config.get('profiler_model', 'N/A')}")
    lines.append(f"  Opponent:           {config.get('opponent_model', 'N/A')}")
    lines.append("")
    lines.append("Game parameters:")
    lines.append(f"  Seller cost:        {config.get('seller_cost', '?')} ZUP")
    lines.append(f"  Buyer WTP:          {config.get('buyer_wtp', '?')} ZUP")
    lines.append(f"  Max iterations:     {config.get('iterations', ITERATIONS)}")
    lines.append(f"  ZOPA:               [{config.get('seller_cost', '?')}, {config.get('buyer_wtp', '?')}] ZUP")
    lines.append("")
    lines.append("Opponent persona prompt:")
    if persona_prompt and persona_prompt.strip():
        lines.append(_indent(f'"{persona_prompt}"'))
    else:
        lines.append("  (neutral - no persona injected)")

    # ── CONVERSATION ─────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("CONVERSATION")
    lines.append("=" * 80)

    # game_state layout:
    #   [0]     = START state (skip)
    #   [1:-1]  = turn states, each has turn (0-indexed), player_complete_answer
    #   [-1]    = END state with "summary" (skip for conversation)
    # turn % 2 == 0 → seller (Player RED) speaks
    # turn % 2 == 1 → buyer  (Player BLUE) speaks

    profiler_logs = profiler_agent.profiler_logs if profiler_agent else []
    # profiler_logs[i] = (opponent_last_msg, profiler_analysis_string)
    # called each time the profiler agent speaks
    # if profiler_is_seller: speaks on turns 0, 2, 4 ... → profiler_logs[0,1,2...]
    # if profiler_is_buyer:  speaks on turns 1, 3, 5 ... → profiler_logs[0,1,2...]
    profiler_log_idx = 0

    turn_states = []
    for state in game.game_state:
        ci = state.get("current_iteration", "")
        if ci == "START" or ci == "END":
            continue
        if "summary" in state and "player_complete_answer" not in state:
            continue  # pure summary entry
        if "player_complete_answer" in state:
            turn_states.append(state)

    # Group into rounds: each round = seller turn (even) + optional buyer turn (odd)
    rounds = {}  # round_num (1-based) → {"seller": state, "buyer": state}
    for state in turn_states:
        t = state.get("turn", 0)
        round_num = t // 2 + 1
        if round_num not in rounds:
            rounds[round_num] = {}
        if t % 2 == 0:
            rounds[round_num]["seller"] = state
        else:
            rounds[round_num]["buyer"] = state

    if not rounds:
        lines.append("(no turns recorded)")
    else:
        for round_num in sorted(rounds.keys()):
            round_data = rounds[round_num]
            lines.append("")
            lines.append(f"{'─' * 60}")
            lines.append(f"ROUND {round_num}")
            lines.append(f"{'─' * 60}")

            # ── Seller turn ──────────────────────────────────────────
            seller_state = round_data.get("seller")
            if seller_state:
                is_our_turn = self_is_seller
                our_label = "OUR AGENT" if is_our_turn else "OPPONENT"

                # Show profiler analysis BEFORE our agent's turn (profiler mode only)
                if profiler_agent and self_is_seller and profiler_log_idx < len(profiler_logs):
                    opp_msg, profiler_output = profiler_logs[profiler_log_idx]
                    profiler_log_idx += 1
                    lines.append("")
                    lines.append(f"[PROFILER ANALYSIS — before {our_name}'s response]")
                    lines.append(_indent(profiler_output))

                answer = seller_state.get("player_public_info_dict", {}).get("player answer", "?")
                lines.append("")
                lines.append(f"[{seller_name} — {our_label} — {answer}]")
                complete = seller_state.get("player_complete_answer", "")
                if complete:
                    lines.append(_indent(complete))

            # ── Buyer turn ───────────────────────────────────────────
            buyer_state = round_data.get("buyer")
            if buyer_state:
                is_our_turn = not self_is_seller
                our_label = "OUR AGENT" if is_our_turn else "OPPONENT"

                # Show profiler analysis BEFORE our agent's turn (profiler mode, buyer role)
                if profiler_agent and not self_is_seller and profiler_log_idx < len(profiler_logs):
                    opp_msg, profiler_output = profiler_logs[profiler_log_idx]
                    profiler_log_idx += 1
                    lines.append("")
                    lines.append(f"[PROFILER ANALYSIS — before {our_name}'s response]")
                    lines.append(_indent(profiler_output))

                answer = buyer_state.get("player_public_info_dict", {}).get("player answer", "?")
                lines.append("")
                lines.append(f"[{buyer_name} — {our_label} — {answer}]")
                complete = buyer_state.get("player_complete_answer", "")
                if complete:
                    lines.append(_indent(complete))

    # ── RESULTS ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 80)
    lines.append("RESULTS")
    lines.append("=" * 80)

    final_response   = result.get("final_response", "N/A")
    seller_outcome   = result.get("seller_outcome")
    buyer_outcome    = result.get("buyer_outcome")
    num_turns        = result.get("num_turns", "N/A")
    deal_reached     = final_response == "ACCEPT"

    lines.append(f"Final response:     {final_response}")
    lines.append(f"Seller outcome:     {seller_outcome} ZUP profit" if seller_outcome is not None else "Seller outcome:     N/A")
    lines.append(f"Buyer outcome:      {buyer_outcome} ZUP surplus" if buyer_outcome is not None else "Buyer outcome:      N/A")
    lines.append(f"Number of turns:    {num_turns}")

    deal_price = None
    if deal_reached and seller_outcome is not None:
        deal_price = config.get("seller_cost", 0) + seller_outcome
        lines.append(f"Deal price:         {deal_price} ZUP")
        lines.append(f"  (seller profit = {deal_price} - {config.get('seller_cost', '?')} cost = {seller_outcome})")
        lines.append(f"  (buyer surplus  = {config.get('buyer_wtp', '?')} WTP - {deal_price} = {buyer_outcome})")
    else:
        lines.append("Deal price:         N/A (no deal)")

    # Write game.log
    log_path = os.path.join(log_dir, "game.log")
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Write results.json
    results_data = {
        "mode": mode,
        "persona": persona_label,
        "run": run_idx,
        "self_role": our_role,
        "seller_cost": config.get("seller_cost"),
        "buyer_wtp": config.get("buyer_wtp"),
        "scenario": config.get("scenario"),
        "final_response": final_response,
        "seller_outcome": seller_outcome,
        "buyer_outcome": buyer_outcome,
        "num_turns": num_turns,
        "deal_reached": deal_reached,
        "deal_price": deal_price,
    }
    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    return log_path


# ── Scenario runners ─────────────────────────────────────────────────

def run_baseline_scenario(persona_label, persona_prompt, self_is_seller, log_dir, config):
    """Run a single baseline game (static agent, no profiler). Returns (result, game)."""
    self_max_tokens = config.get("self_max_tokens", 800)
    if self_is_seller:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=config["self_model"], max_tokens=self_max_tokens)
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=config["opponent_model"], max_tokens=800)
        social = ["", persona_prompt]
    else:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=config["opponent_model"], max_tokens=800)
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=config["self_model"], max_tokens=self_max_tokens)
        social = [persona_prompt, ""]

    game = BuySellGame(
        players=[seller, buyer],
        iterations=config["iterations"],
        player_goals=[
            SellerGoal(cost_of_production=Valuation({"X": config["seller_cost"]})),
            BuyerGoal(willingness_to_pay=Valuation({"X": config["buyer_wtp"]})),
        ],
        player_starting_resources=[
            Resources({"X": 1}),
            Resources({MONEY_TOKEN: 1000}),
        ],
        player_conversation_roles=[
            f"You are {AGENT_ONE}.",
            f"You are {AGENT_TWO}.",
        ],
        player_social_behaviour=social,
        log_dir=log_dir,
    )
    game.run()

    final = game.game_state[-1]
    summary = final.get("summary", final)
    result = {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome":  summary.get("player_outcome", [None, None])[1],
        "num_turns":      len(game.game_state) - 2,  # subtract START and END
    }
    return result, game


def run_profiler_scenario(persona_label, persona_prompt, self_is_seller, log_dir, config):
    """Run a single profiler game (adaptive agent). Returns (result, game, profiler_agent)."""
    if self_is_seller:
        seller = ProfilerAgent(
            agent_name=AGENT_ONE,
            profiler_model=config["profiler_model"],
            negotiator_model=config["negotiator_model"],
        )
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=config["opponent_model"], max_tokens=800)
        social = ["", persona_prompt]
        profiler_agent = seller
    else:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=config["opponent_model"], max_tokens=800)
        buyer = ProfilerAgent(
            agent_name=AGENT_TWO,
            profiler_model=config["profiler_model"],
            negotiator_model=config["negotiator_model"],
        )
        social = [persona_prompt, ""]
        profiler_agent = buyer

    game = BuySellGame(
        players=[seller, buyer],
        iterations=config["iterations"],
        player_goals=[
            SellerGoal(cost_of_production=Valuation({"X": config["seller_cost"]})),
            BuyerGoal(willingness_to_pay=Valuation({"X": config["buyer_wtp"]})),
        ],
        player_starting_resources=[
            Resources({"X": 1}),
            Resources({MONEY_TOKEN: 1000}),
        ],
        player_conversation_roles=[
            f"You are {AGENT_ONE}.",
            f"You are {AGENT_TWO}.",
        ],
        player_social_behaviour=social,
        log_dir=log_dir,
    )
    game.run()

    final = game.game_state[-1]
    summary = final.get("summary", final)
    result = {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome":  summary.get("player_outcome", [None, None])[1],
        "num_turns":      len(game.game_state) - 2,
    }
    return result, game, profiler_agent


# ── Main experiment loop ─────────────────────────────────────────────

def run_experiments(mode, num_runs, role, num_scenarios=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = os.path.join(BASE_LOG_DIR, f"run_{timestamp}")
    os.makedirs(log_base, exist_ok=True)

    self_is_seller = (role == "seller")
    scenarios = PRICE_SCENARIOS[:num_scenarios]

    base_config = {
        "self_model":       SELF_MODEL,
        "negotiator_model": NEGOTIATOR_MODEL,
        "profiler_model":   PROFILER_MODEL,
        "opponent_model":   OPPONENT_MODEL,
        "compare_model":    COMPARE_MODEL,
        "iterations":       ITERATIONS,
        "timestamp":        timestamp,
        "mode":             mode,
        "role":             role,
        "num_runs":         num_runs,
        "scenarios":        scenarios,
    }

    # Save experiment-wide config
    config_path = os.path.join(log_base, "config.json")
    with open(config_path, "w") as f:
        json.dump(base_config, f, indent=2)

    modes_to_run = []
    if mode in ("all", "both", "baseline"):
        modes_to_run.append("baseline")
    if mode in ("all", "both", "profiler"):
        modes_to_run.append("profiler")
    if mode in ("all", "compare"):
        modes_to_run.append("compare")

    all_results = []

    for scenario_idx, (seller_cost, buyer_wtp) in enumerate(scenarios, start=1):
        scenario_label = f"s{seller_cost}v{buyer_wtp}"
        zopa = buyer_wtp - seller_cost
        scenario_config = {
            **base_config,
            "seller_cost": seller_cost,
            "buyer_wtp":   buyer_wtp,
            "scenario":    scenario_label,
        }

        print(f"\n{'#' * 70}")
        print(f"  SCENARIO {scenario_idx}/{len(scenarios)}: seller_cost={seller_cost}  buyer_wtp={buyer_wtp}  ZOPA={zopa:+d}")
        print(f"{'#' * 70}")

        for persona_label, persona_prompt in OPPONENT_PERSONAS.items():
            for run_idx in range(1, num_runs + 1):

                print(f"\n{'=' * 70}")
                print(f"  Persona: {persona_label}  |  Run: {run_idx}/{num_runs}")
                print(f"{'=' * 70}")

                for current_mode in modes_to_run:
                    # Directory: scenario_<label>/vs_<persona>/run_<N>/<mode>/
                    paired_log_dir = os.path.join(
                        log_base, f"scenario_{scenario_label}",
                        f"vs_{persona_label}", f"run_{run_idx}", current_mode
                    )
                    framework_log_dir = os.path.join(paired_log_dir, "framework_logs")

                    print(f"\n  [{current_mode.upper()}]  role={role}  scenario={scenario_label}  persona={persona_label}  run={run_idx}")

                    for attempt in range(MAX_RETRIES):
                        try:
                            if current_mode == "baseline":
                                result, game = run_baseline_scenario(
                                    persona_label, persona_prompt, self_is_seller,
                                    framework_log_dir, scenario_config
                                )
                                write_game_log(
                                    paired_log_dir, current_mode,
                                    persona_label, persona_prompt,
                                    self_is_seller, run_idx,
                                    game, result,
                                    profiler_agent=None,
                                    config=scenario_config,
                                )
                            elif current_mode == "compare":
                                # OSS vs OSS: swap self_model for compare_model
                                compare_config = {
                                    **scenario_config,
                                    "self_model": scenario_config["compare_model"],
                                }
                                result, game = run_baseline_scenario(
                                    persona_label, persona_prompt, self_is_seller,
                                    framework_log_dir, compare_config
                                )
                                write_game_log(
                                    paired_log_dir, current_mode,
                                    persona_label, persona_prompt,
                                    self_is_seller, run_idx,
                                    game, result,
                                    profiler_agent=None,
                                    config=compare_config,
                                )
                            else:
                                result, game, profiler_agent = run_profiler_scenario(
                                    persona_label, persona_prompt, self_is_seller,
                                    framework_log_dir, scenario_config
                                )
                                write_game_log(
                                    paired_log_dir, current_mode,
                                    persona_label, persona_prompt,
                                    self_is_seller, run_idx,
                                    game, result,
                                    profiler_agent=profiler_agent,
                                    config=scenario_config,
                                )

                            result.update({
                                "mode":        current_mode,
                                "persona":     persona_label,
                                "run":         run_idx,
                                "role":        role,
                                "scenario":    scenario_label,
                                "seller_cost": seller_cost,
                                "buyer_wtp":   buyer_wtp,
                            })
                            all_results.append(result)

                            print(f"    Result:  {result['final_response']}")
                            print(f"    Seller:  {result['seller_outcome']}    Buyer: {result['buyer_outcome']}    Turns: {result['num_turns']}")
                            print(f"    Log:     {paired_log_dir}/game.log")
                            break

                        except Exception as e:
                            print(f"    Attempt {attempt + 1}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
                            if attempt < MAX_RETRIES - 1:
                                print("    Retrying...")
                            else:
                                print(f"    All {MAX_RETRIES} attempts failed, skipping.")
                                traceback.print_exc()
                                all_results.append({
                                    "mode":        current_mode,
                                    "persona":     persona_label,
                                    "run":         run_idx,
                                    "role":        role,
                                    "scenario":    scenario_label,
                                    "seller_cost": seller_cost,
                                    "buyer_wtp":   buyer_wtp,
                                    "error":       str(e),
                                })

    # ── Summary log ─────────────────────────────────────────────────
    summary_path = os.path.join(log_base, "summary.log")
    _write_summary(summary_path, all_results, base_config, timestamp)
    print(f"\nSummary log: {summary_path}")
    _print_summary_table(all_results)

    return all_results


def _write_summary(path, all_results, config, timestamp):

    with open(path, "w") as f:
        f.write(f"Experiment: run_{timestamp}\n")
        f.write(f"Mode: {config['mode']}  |  Role: {config['role']}  |  Runs per persona: {config['num_runs']}\n")
        f.write(f"Baseline model:    {config['self_model']}\n")
        f.write(f"Compare model:     {config['compare_model']}\n")
        f.write(f"Negotiator model:  {config['negotiator_model']}\n")
        f.write(f"Profiler model:    {config['profiler_model']}\n")
        f.write(f"Opponent model:    {config['opponent_model']}\n")
        scenarios = config.get("scenarios", [])
        f.write(f"Scenarios ({len(scenarios)}): " + ", ".join(
            f"s{sc}v{bw}(ZOPA={bw-sc:+d})" for sc, bw in scenarios
        ) + "\n")
        f.write("=" * 110 + "\n\n")

        # Detailed results table
        f.write(f"{'Scenario':<16} {'Mode':<10} {'Persona':<12} {'Run':>3}  "
                f"{'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}\n")
        f.write("-" * 110 + "\n")

        for r in all_results:
            scenario = r.get("scenario", "?")
            if "error" in r:
                f.write(f"{scenario:<16} {r['mode']:<10} {r['persona']:<12} {r['run']:>3}  {'ERROR':<8}\n")
            else:
                f.write(
                    f"{scenario:<16} {r['mode']:<10} {r['persona']:<12} {r['run']:>3}  "
                    f"{r['final_response']:<8} {str(r['seller_outcome']):>6} "
                    f"{str(r['buyer_outcome']):>6} {str(r['num_turns']):>5}\n"
                )

        # Aggregate: mean seller_outcome per (scenario, mode)
        f.write("\n" + "=" * 110 + "\n")
        f.write("AGGREGATE: mean seller_outcome per (scenario, mode)\n")
        f.write("-" * 80 + "\n")

        agg = defaultdict(list)
        for r in all_results:
            if "error" not in r and r.get("seller_outcome") is not None:
                agg[(r["scenario"], r["mode"])].append(r["seller_outcome"])

        unique_scenarios = list(dict.fromkeys(r.get("scenario", "?") for r in all_results))
        modes            = list(dict.fromkeys(r["mode"] for r in all_results if "mode" in r))

        header = f"{'Scenario':<16}" + "".join(f"  {m:<22}" for m in modes)
        f.write(header + "\n")
        f.write("-" * (16 + 24 * len(modes)) + "\n")

        for scenario in unique_scenarios:
            row = f"{scenario:<16}"
            for m in modes:
                vals = agg.get((scenario, m), [])
                if vals:
                    mean = sum(vals) / len(vals)
                    row += f"  mean={mean:+.1f}  n={len(vals):<6}"
                else:
                    row += f"  {'N/A':<22}"
            f.write(row + "\n")

        # Aggregate: deal rate per (scenario, mode)
        f.write("\n" + "=" * 110 + "\n")
        f.write("AGGREGATE: deal rate per (scenario, mode)\n")
        f.write("-" * 80 + "\n")

        deal_agg = defaultdict(lambda: [0, 0])  # [deals, total]
        for r in all_results:
            if "error" not in r:
                key = (r.get("scenario", "?"), r["mode"])
                deal_agg[key][1] += 1
                if r.get("deal_reached"):
                    deal_agg[key][0] += 1

        f.write(header + "\n")
        f.write("-" * (16 + 24 * len(modes)) + "\n")

        for scenario in unique_scenarios:
            row = f"{scenario:<16}"
            for m in modes:
                deals, total = deal_agg.get((scenario, m), [0, 0])
                if total:
                    rate = deals / total
                    row += f"  {deals}/{total} ({rate:.0%})        "
                else:
                    row += f"  {'N/A':<22}"
            f.write(row + "\n")


def _print_summary_table(all_results):
    print(f"\n{'=' * 110}")
    print("SUMMARY")
    print(f"{'=' * 110}")
    print(f"{'Scenario':<16} {'Mode':<10} {'Persona':<12} {'Run':>3}  {'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}")
    print("-" * 110)
    for r in all_results:
        scenario = r.get("scenario", "?")
        if "error" in r:
            print(f"{scenario:<16} {r.get('mode','?'):<10} {r.get('persona','?'):<12} {r.get('run','?'):>3}  {'ERROR':<8}")
        else:
            print(
                f"{scenario:<16} {r.get('mode','?'):<10} {r.get('persona','?'):<12} {r.get('run','?'):>3}  "
                f"{r['final_response']:<8} "
                f"{str(r['seller_outcome']):>6} {str(r['buyer_outcome']):>6} "
                f"{str(r['num_turns']):>5}"
            )


# ── Entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline and/or profiler negotiation experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "both", "baseline", "profiler", "compare"],
        default="all",
        help="Which experiments to run: 'all'=baseline+profiler+compare, 'both'=baseline+profiler (default: all)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per persona per mode (default: 1)",
    )
    parser.add_argument(
        "--role",
        choices=["seller", "buyer"],
        default="seller",
        help="Role for our agent (default: seller)",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help=(
            "Number of price scenarios to run (default: all %d). "
            "Scenarios are taken in order from PRICE_SCENARIOS." % len(PRICE_SCENARIOS)
        ),
    )
    args = parser.parse_args()

    num_scenarios = args.num_scenarios if args.num_scenarios is not None else len(PRICE_SCENARIOS)
    scenarios_to_run = PRICE_SCENARIOS[:num_scenarios]

    print("Configuration:")
    print(f"  Mode:              {args.mode}")
    print(f"  Runs per persona:  {args.num_runs}")
    print(f"  Our role:          {args.role}")
    print(f"  Scenarios:         {num_scenarios}/{len(PRICE_SCENARIOS)}")
    for sc, bw in scenarios_to_run:
        print(f"    seller_cost={sc:>3}  buyer_wtp={bw:>3}  ZOPA={bw - sc:+d}")
    print(f"  Baseline model:    {SELF_MODEL}")
    print(f"  Compare model:     {COMPARE_MODEL}")
    print(f"  Negotiator model:  {NEGOTIATOR_MODEL}")
    print(f"  Profiler model:    {PROFILER_MODEL}")
    print(f"  Opponent model:    {OPPONENT_MODEL}")

    run_experiments(args.mode, args.num_runs, args.role, num_scenarios=num_scenarios)


if __name__ == "__main__":
    main()
