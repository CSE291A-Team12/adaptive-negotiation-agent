"""Unified N-run experiment runner for baseline and profiler comparisons.

Runs BuySellGame experiments across 5 opponent strategies with retry logic.
Supports baseline (static Mistral-small) and profiler (Mistral-small + GPT-OSS profiler).

Usage:
    python scripts/run_experiment.py --mode both --num-runs 3
    python scripts/run_experiment.py --mode baseline --num-runs 5 --role buyer
    python scripts/run_experiment.py --mode profiler --num-runs 1
"""

import argparse
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

# ── Models ──────────────────────────────────────────────────────────
SELF_MODEL = "api-mistral-small-3.2-2506"       # baseline negotiator
NEGOTIATOR_MODEL = "api-mistral-small-3.2-2506"  # profiler negotiator
PROFILER_MODEL = "api-gpt-oss-120b"              # profiler brain
OPPONENT_MODEL = "api-gpt-oss-120b"              # opponent

# ── Game params ─────────────────────────────────────────────────────
SELLER_COST = 40
BUYER_WTP = 60
ITERATIONS = 10
MAX_RETRIES = 3

BASE_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiments"
)

OPPONENT_PERSONAS = {
    "neutral": "",
    "hardball": (
        "You are an aggressive and tough negotiator. Never accept the first offer. "
        "Always push for the lowest price possible. Do not reveal your willingness to pay."
    ),
    "friendly": (
        "You are very kind and generous. Be friendly and helpful with the other player."
    ),
    "sycophant": (
        "You are very agreeable and submissive. Compliment the other player frequently. "
        "Accept offers quickly to maintain a good relationship. Avoid conflict at all costs."
    ),
    "stalling": (
        "You are a cautious negotiator. Delay committing to a final price as long as possible. "
        "Ask clarifying questions, request justifications, and make only small concessions. "
        "When you do propose a trade, always use a valid number."
    ),
}


def run_baseline_scenario(persona_label, persona_prompt, self_is_seller, log_dir):
    """Run a single baseline game (static agent, no profiler)."""
    if self_is_seller:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=SELF_MODEL)
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=OPPONENT_MODEL)
        social = ["", persona_prompt]
    else:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=OPPONENT_MODEL)
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=SELF_MODEL)
        social = [persona_prompt, ""]

    game = BuySellGame(
        players=[seller, buyer],
        iterations=ITERATIONS,
        player_goals=[
            SellerGoal(cost_of_production=Valuation({"X": SELLER_COST})),
            BuyerGoal(willingness_to_pay=Valuation({"X": BUYER_WTP})),
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
    return {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome": summary.get("player_outcome", [None, None])[1],
        "num_turns": len(game.game_state) - 1,
    }


def run_profiler_scenario(persona_label, persona_prompt, profiler_is_seller, log_dir):
    """Run a single profiler game (adaptive agent with profiler)."""
    if profiler_is_seller:
        seller = ProfilerAgent(
            agent_name=AGENT_ONE,
            profiler_model=PROFILER_MODEL,
            negotiator_model=NEGOTIATOR_MODEL,
        )
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=OPPONENT_MODEL)
        social = ["", persona_prompt]
    else:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=OPPONENT_MODEL)
        buyer = ProfilerAgent(
            agent_name=AGENT_TWO,
            profiler_model=PROFILER_MODEL,
            negotiator_model=NEGOTIATOR_MODEL,
        )
        social = [persona_prompt, ""]

    game = BuySellGame(
        players=[seller, buyer],
        iterations=ITERATIONS,
        player_goals=[
            SellerGoal(cost_of_production=Valuation({"X": SELLER_COST})),
            BuyerGoal(willingness_to_pay=Valuation({"X": BUYER_WTP})),
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
    return {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome": summary.get("player_outcome", [None, None])[1],
        "num_turns": len(game.game_state) - 1,
    }


def run_experiments(mode, num_runs, role):
    """Run experiments across all personas for the given mode and role."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = os.path.join(BASE_LOG_DIR, f"run_{timestamp}")
    os.makedirs(log_base, exist_ok=True)

    self_is_seller = role == "seller"
    all_results = []

    modes_to_run = []
    if mode in ("both", "baseline"):
        modes_to_run.append("baseline")
    if mode in ("both", "profiler"):
        modes_to_run.append("profiler")

    for current_mode in modes_to_run:
        print(f"\n{'#' * 60}")
        print(f"  MODE: {current_mode.upper()}  |  Role: {role}  |  Runs: {num_runs}")
        print(f"{'#' * 60}")

        for persona_label, persona_prompt in OPPONENT_PERSONAS.items():
            for run_idx in range(1, num_runs + 1):
                label = f"{current_mode}_{role}_vs_{persona_label}_run{run_idx}"
                log_dir = os.path.join(log_base, current_mode, label)

                print(f"\n--- {label} ---")

                for attempt in range(MAX_RETRIES):
                    try:
                        if current_mode == "baseline":
                            result = run_baseline_scenario(
                                persona_label, persona_prompt, self_is_seller, log_dir
                            )
                        else:
                            result = run_profiler_scenario(
                                persona_label, persona_prompt, self_is_seller, log_dir
                            )

                        result["mode"] = current_mode
                        result["persona"] = persona_label
                        result["run"] = run_idx
                        result["label"] = label
                        all_results.append(result)

                        print(f"  Result: {result['final_response']}")
                        print(f"  Seller: {result['seller_outcome']}  Buyer: {result['buyer_outcome']}  Turns: {result['num_turns']}")
                        break

                    except Exception as e:
                        print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
                        if attempt < MAX_RETRIES - 1:
                            print("  Retrying...")
                        else:
                            print(f"  All {MAX_RETRIES} attempts failed, skipping.")
                            traceback.print_exc()
                            all_results.append({
                                "mode": current_mode,
                                "persona": persona_label,
                                "run": run_idx,
                                "label": label,
                                "error": str(e),
                            })

    # ── Write summary log ───────────────────────────────────────────
    summary_path = os.path.join(log_base, "summary.log")
    with open(summary_path, "w") as f:
        f.write(f"Experiment run: {timestamp}\n")
        f.write(f"Mode: {mode} | Role: {role} | Runs per persona: {num_runs}\n")
        f.write(f"Baseline model: {SELF_MODEL} | Negotiator: {NEGOTIATOR_MODEL}\n")
        f.write(f"Profiler: {PROFILER_MODEL} | Opponent: {OPPONENT_MODEL}\n")
        f.write(f"Seller cost: {SELLER_COST} | Buyer WTP: {BUYER_WTP}\n")
        f.write("=" * 80 + "\n\n")

        for r in all_results:
            if "error" in r:
                f.write(f"{r['label']}: ERROR - {r['error']}\n")
            else:
                f.write(
                    f"{r['label']}: {r['final_response']} | "
                    f"seller={r['seller_outcome']} buyer={r['buyer_outcome']} "
                    f"turns={r['num_turns']}\n"
                )

    print(f"\nSummary log: {summary_path}")

    # ── Print summary table ─────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Label':<50} {'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}")
    print("-" * 80)
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<50} {'ERROR':<8}")
        else:
            print(
                f"{r['label']:<50} {r['final_response']:<8} "
                f"{r['seller_outcome']:>6} {r['buyer_outcome']:>6} {r['num_turns']:>5}"
            )

    return all_results


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

    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Runs per persona: {args.num_runs}")
    print(f"  Our role: {args.role}")
    print(f"  Baseline model: {SELF_MODEL}")
    print(f"  Negotiator model: {NEGOTIATOR_MODEL}")
    print(f"  Profiler model: {PROFILER_MODEL}")
    print(f"  Opponent model: {OPPONENT_MODEL}")

    run_experiments(args.mode, args.num_runs, args.role)


if __name__ == "__main__":
    main()
