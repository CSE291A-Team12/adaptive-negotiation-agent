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

SELLER_COST = 40
BUYER_WTP = 60
ITERATIONS = 10
MAX_RETRIES = 3

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
    if mode == "baseline":
        lines.append(f"  Our agent:          {config.get('self_model', 'N/A')}")
    else:
        lines.append(f"  Our negotiator:     {config.get('negotiator_model', 'N/A')}")
        lines.append(f"  Profiler brain:     {config.get('profiler_model', 'N/A')}")
    lines.append(f"  Opponent:           {config.get('opponent_model', 'N/A')}")
    lines.append("")
    lines.append("Game parameters:")
    lines.append(f"  Seller cost:        {config.get('seller_cost', SELLER_COST)} ZUP")
    lines.append(f"  Buyer WTP:          {config.get('buyer_wtp', BUYER_WTP)} ZUP")
    lines.append(f"  Max iterations:     {config.get('iterations', ITERATIONS)}")
    lines.append(f"  ZOPA:               [{config.get('seller_cost', SELLER_COST)}, {config.get('buyer_wtp', BUYER_WTP)}] ZUP")
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
        deal_price = config.get("seller_cost", SELLER_COST) + seller_outcome
        lines.append(f"Deal price:         {deal_price} ZUP")
        lines.append(f"  (seller profit = {deal_price} - {config.get('seller_cost', SELLER_COST)} cost = {seller_outcome})")
        lines.append(f"  (buyer surplus  = {config.get('buyer_wtp', BUYER_WTP)} WTP - {deal_price} = {buyer_outcome})")
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
    if self_is_seller:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=config["self_model"])
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=config["opponent_model"], max_tokens=800)
        social = ["", persona_prompt]
    else:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=config["opponent_model"], max_tokens=800)
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=config["self_model"])
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

def run_experiments(mode, num_runs, role):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = os.path.join(BASE_LOG_DIR, f"run_{timestamp}")
    os.makedirs(log_base, exist_ok=True)

    self_is_seller = (role == "seller")

    config = {
        "self_model":       SELF_MODEL,
        "negotiator_model": NEGOTIATOR_MODEL,
        "profiler_model":   PROFILER_MODEL,
        "opponent_model":   OPPONENT_MODEL,
        "seller_cost":      SELLER_COST,
        "buyer_wtp":        BUYER_WTP,
        "iterations":       ITERATIONS,
        "timestamp":        timestamp,
        "mode":             mode,
        "role":             role,
        "num_runs":         num_runs,
    }

    # Save experiment-wide config
    config_path = os.path.join(log_base, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    modes_to_run = []
    if mode in ("both", "baseline"):
        modes_to_run.append("baseline")
    if mode in ("both", "profiler"):
        modes_to_run.append("profiler")

    all_results = []

    for persona_label, persona_prompt in OPPONENT_PERSONAS.items():
        for run_idx in range(1, num_runs + 1):

            print(f"\n{'=' * 70}")
            print(f"  Persona: {persona_label}  |  Run: {run_idx}/{num_runs}")
            print(f"{'=' * 70}")

            for current_mode in modes_to_run:
                # Paired directory: vs_<persona>/run_<N>/<mode>/
                paired_log_dir = os.path.join(
                    log_base, f"vs_{persona_label}", f"run_{run_idx}", current_mode
                )
                # The game framework writes its own game_state.json into a timestamped
                # subdir inside log_dir; we pass the same root so the framework files
                # land alongside our game.log.
                framework_log_dir = os.path.join(paired_log_dir, "framework_logs")

                print(f"\n  [{current_mode.upper()}]  role={role}  persona={persona_label}  run={run_idx}")

                for attempt in range(MAX_RETRIES):
                    try:
                        if current_mode == "baseline":
                            result, game = run_baseline_scenario(
                                persona_label, persona_prompt, self_is_seller,
                                framework_log_dir, config
                            )
                            write_game_log(
                                paired_log_dir, current_mode,
                                persona_label, persona_prompt,
                                self_is_seller, run_idx,
                                game, result,
                                profiler_agent=None,
                                config=config,
                            )
                        else:
                            result, game, profiler_agent = run_profiler_scenario(
                                persona_label, persona_prompt, self_is_seller,
                                framework_log_dir, config
                            )
                            write_game_log(
                                paired_log_dir, current_mode,
                                persona_label, persona_prompt,
                                self_is_seller, run_idx,
                                game, result,
                                profiler_agent=profiler_agent,
                                config=config,
                            )

                        result.update({
                            "mode":    current_mode,
                            "persona": persona_label,
                            "run":     run_idx,
                            "role":    role,
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
                                "mode":    current_mode,
                                "persona": persona_label,
                                "run":     run_idx,
                                "role":    role,
                                "error":   str(e),
                            })

    # ── Summary log ─────────────────────────────────────────────────
    summary_path = os.path.join(log_base, "summary.log")
    _write_summary(summary_path, all_results, config, timestamp)
    print(f"\nSummary log: {summary_path}")
    _print_summary_table(all_results)

    return all_results


def _write_summary(path, all_results, config, timestamp):
    with open(path, "w") as f:
        f.write(f"Experiment: run_{timestamp}\n")
        f.write(f"Mode: {config['mode']}  |  Role: {config['role']}  |  Runs per persona: {config['num_runs']}\n")
        f.write(f"Baseline model:    {config['self_model']}\n")
        f.write(f"Negotiator model:  {config['negotiator_model']}\n")
        f.write(f"Profiler model:    {config['profiler_model']}\n")
        f.write(f"Opponent model:    {config['opponent_model']}\n")
        f.write(f"Seller cost: {config['seller_cost']} ZUP  |  Buyer WTP: {config['buyer_wtp']} ZUP\n")
        f.write("=" * 90 + "\n\n")

        # Table header
        f.write(f"{'Label':<50} {'Mode':<10} {'Persona':<12} {'Run':>3}  "
                f"{'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}\n")
        f.write("-" * 100 + "\n")

        for r in all_results:
            if "error" in r:
                label = f"{r['mode']}_{r['role']}_vs_{r['persona']}_run{r['run']}"
                f.write(f"{label:<50} {r['mode']:<10} {r['persona']:<12} {r['run']:>3}  {'ERROR':<8}\n")
            else:
                label = f"{r['mode']}_{r['role']}_vs_{r['persona']}_run{r['run']}"
                f.write(
                    f"{label:<50} {r['mode']:<10} {r['persona']:<12} {r['run']:>3}  "
                    f"{r['final_response']:<8} {str(r['seller_outcome']):>6} "
                    f"{str(r['buyer_outcome']):>6} {str(r['num_turns']):>5}\n"
                )

        # Aggregate: mean seller outcome per (mode, persona)
        f.write("\n" + "=" * 90 + "\n")
        f.write("AGGREGATE: mean seller_outcome per (mode, persona)\n")
        f.write("-" * 60 + "\n")

        from collections import defaultdict
        agg = defaultdict(list)
        for r in all_results:
            if "error" not in r and r.get("seller_outcome") is not None:
                agg[(r["mode"], r["persona"])].append(r["seller_outcome"])

        personas = list(dict.fromkeys(r["persona"] for r in all_results if "persona" in r))
        modes    = list(dict.fromkeys(r["mode"]    for r in all_results if "mode"    in r))

        header = f"{'Persona':<14}" + "".join(f"  {m:<20}" for m in modes)
        f.write(header + "\n")
        f.write("-" * (14 + 22 * len(modes)) + "\n")

        for persona in personas:
            row = f"{persona:<14}"
            for m in modes:
                vals = agg.get((m, persona), [])
                if vals:
                    mean = sum(vals) / len(vals)
                    row += f"  mean={mean:+.1f}  n={len(vals):<6}"
                else:
                    row += f"  {'N/A':<20}"
            f.write(row + "\n")


def _print_summary_table(all_results):
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Label':<50} {'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}")
    print("-" * 90)
    for r in all_results:
        label = f"{r.get('mode','?')}_{r.get('role','?')}_vs_{r.get('persona','?')}_run{r.get('run','?')}"
        if "error" in r:
            print(f"{label:<50} {'ERROR':<8}")
        else:
            print(
                f"{label:<50} {r['final_response']:<8} "
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
        choices=["both", "baseline", "profiler"],
        default="both",
        help="Which experiments to run (default: both)",
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
    args = parser.parse_args()

    print("Configuration:")
    print(f"  Mode:              {args.mode}")
    print(f"  Runs per persona:  {args.num_runs}")
    print(f"  Our role:          {args.role}")
    print(f"  Baseline model:    {SELF_MODEL}")
    print(f"  Negotiator model:  {NEGOTIATOR_MODEL}")
    print(f"  Profiler model:    {PROFILER_MODEL}")
    print(f"  Opponent model:    {OPPONENT_MODEL}")
    print(f"  Seller cost:       {SELLER_COST} ZUP")
    print(f"  Buyer WTP:         {BUYER_WTP} ZUP")

    run_experiments(args.mode, args.num_runs, args.role)


if __name__ == "__main__":
    main()
