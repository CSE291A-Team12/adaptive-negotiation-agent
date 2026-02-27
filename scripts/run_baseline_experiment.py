"""Baseline experiment: static Mistral-small vs GPT-OSS across 5 opponent strategies.

Seller (Player RED) = Mistral-small (our baseline agent, no profiler)
Buyer (Player BLUE) = GPT-OSS with persona

Seller cost: 40 ZUP  |  Buyer WTP: 60 ZUP  |  10 iterations

Usage:
    python scripts/run_baseline_experiment.py
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

from dotenv import load_dotenv

from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
from games.buy_sell_game.game import BuySellGame

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Models ──────────────────────────────────────────────────────────
SELF_MODEL = "api-mistral-small-3.2-2506"
OPPONENT_MODEL = "api-gpt-oss-120b"

# ── Game params ─────────────────────────────────────────────────────
SELLER_COST = 40
BUYER_WTP = 60
ITERATIONS = 10

BASE_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "baseline_experiment"
)

# ── Opponent personas (label, seller_persona, buyer_persona) ────────
# Static seller (our agent) vs 5 buyer strategies
SCENARIOS = [
    ("vs_neutral", "", ""),
    (
        "vs_hardball",
        "",
        "You are an aggressive and tough negotiator. Never accept the first offer. "
        "Always push for the lowest price possible. Do not reveal your willingness to pay.",
    ),
    (
        "vs_friendly",
        "",
        "You are very kind and generous. Be friendly and helpful with the other player.",
    ),
    (
        "vs_sycophant",
        "",
        "You are very agreeable and submissive. Compliment the other player frequently. "
        "Accept offers quickly to maintain a good relationship. Avoid conflict at all costs.",
    ),
    (
        "vs_stalling",
        "",
        "You are a cautious negotiator. Delay committing to a final price as long as possible. "
        "Ask clarifying questions, request justifications, and make only small concessions. "
        "When you do propose a trade, always use a valid number.",
    ),
]


def run_scenario(label, seller_persona, buyer_persona):
    log_dir = os.path.join(BASE_LOG_DIR, label)

    seller = ChatGPTAgent(agent_name=AGENT_ONE, model=SELF_MODEL)
    buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=OPPONENT_MODEL)

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
        player_social_behaviour=[seller_persona, buyer_persona],
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


if __name__ == "__main__":
    print(f"Baseline experiment: {SELF_MODEL} (seller) vs {OPPONENT_MODEL} (buyer)")
    print(f"Seller cost: {SELLER_COST} ZUP | Buyer WTP: {BUYER_WTP} ZUP")
    print(f"Logs: {BASE_LOG_DIR}")

    results = []

    for label, seller_persona, buyer_persona in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {label}")
        print(f"  Seller: {seller_persona or '(neutral)'}")
        print(f"  Buyer:  {buyer_persona or '(neutral)'}")
        print(f"{'=' * 60}")

        try:
            result = run_scenario(label, seller_persona, buyer_persona)
            result["scenario"] = label
            results.append(result)
            print(f"  Result: {result['final_response']}")
            print(f"  Seller profit: {result['seller_outcome']}")
            print(f"  Buyer surplus: {result['buyer_outcome']}")
            print(f"  Turns: {result['num_turns']}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append({"scenario": label, "error": str(e)})

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Scenario':<40} {'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['scenario']:<40} {'ERROR':<8}")
        else:
            print(
                f"{r['scenario']:<40} {r['final_response']:<8} "
                f"{r['seller_outcome']:>6} {r['buyer_outcome']:>6} {r['num_turns']:>5}"
            )
