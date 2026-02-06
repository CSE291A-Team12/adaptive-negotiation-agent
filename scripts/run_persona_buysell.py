"""Run BuySellGame with various social behaviour personas.

Tests how static persona prompting affects negotiation outcomes
between two GPT-4o-mini agents.

Usage:
    python scripts/run_persona_buysell.py
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

load_dotenv()

MODEL = "gpt-4o-mini"
BASE_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "persona_buysell"
)

# (label, seller_persona, buyer_persona)
SCENARIOS = [
    (
        "neutral_vs_neutral",
        "",
        "",
    ),
    (
        "hardball_seller_vs_neutral",
        "You are an aggressive and tough negotiator. Never accept the first offer. Always push for the highest price possible. Do not reveal your costs.",
        "",
    ),
    (
        "neutral_vs_hardball_buyer",
        "",
        "You are an aggressive and tough negotiator. Never accept the first offer. Always push for the lowest price possible. Do not reveal your willingness to pay.",
    ),
    (
        "hardball_vs_hardball",
        "You are an aggressive and tough negotiator. Never accept the first offer. Always push for the highest price possible. Do not reveal your costs.",
        "You are an aggressive and tough negotiator. Never accept the first offer. Always push for the lowest price possible. Do not reveal your willingness to pay.",
    ),
    (
        "friendly_seller_vs_hardball_buyer",
        "You are very kind and generous. Be friendly and helpful with the other player.",
        "You are an aggressive and tough negotiator. Never accept the first offer. Always push for the lowest price possible. Do not reveal your willingness to pay.",
    ),
    (
        "hardball_seller_vs_friendly_buyer",
        "You are an aggressive and tough negotiator. Never accept the first offer. Always push for the highest price possible. Do not reveal your costs.",
        "You are very kind and generous. Be friendly and helpful with the other player.",
    ),
]


def run_scenario(label, seller_persona, buyer_persona):
    log_dir = os.path.join(BASE_LOG_DIR, label)

    seller = ChatGPTAgent(agent_name=AGENT_ONE, model=MODEL)
    buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=MODEL)

    game = BuySellGame(
        players=[seller, buyer],
        iterations=10,
        player_goals=[
            SellerGoal(cost_of_production=Valuation({"X": 40})),
            BuyerGoal(willingness_to_pay=Valuation({"X": 60})),
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
        "num_turns": len(game.game_state) - 1,  # exclude END state
    }


if __name__ == "__main__":
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

    # Print summary table
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
