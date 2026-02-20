"""Batch runner: ProfilerAgent vs 5 buyer personas in both roles.

ProfilerAgent as Seller (5 scenarios):
  ProfilerAgent (AGENT_ONE) vs GPT-4o-mini buyer with persona

ProfilerAgent as Buyer (5 scenarios):
  GPT-4o-mini seller with persona vs ProfilerAgent (AGENT_TWO)
"""

import json
import sys
import os
import traceback

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

OPPONENT_MODEL = "gpt-4o-mini"
NEGOTIATOR_MODEL = "api-llama-4-scout"
PROFILER_MODEL = "api-gpt-oss-120b"

BASE_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "profiler_buysell"
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


def save_profiler_logs(profiler_agent, log_dir):
    """Save profiler logs as JSON alongside the game logs."""
    logs = []
    for opponent_msg, profiler_output in profiler_agent.profiler_logs:
        logs.append({
            "opponent_message": opponent_msg.get("content", ""),
            "profiler_output": profiler_output,
        })

    # Game creates a timestamped subdir — find the most recent one
    subdirs = sorted(
        [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))],
        reverse=True,
    )
    if subdirs:
        path = os.path.join(log_dir, subdirs[0], "profiler_logs.json")
    else:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "profiler_logs.json")

    with open(path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"  Profiler logs saved to {path}")


def run_scenario(label, opponent_persona, profiler_is_seller=True):
    log_dir = os.path.join(BASE_LOG_DIR, label)

    if profiler_is_seller:
        seller = ProfilerAgent(
            agent_name=AGENT_ONE,
            profiler_model=PROFILER_MODEL,
            negotiator_model=NEGOTIATOR_MODEL,
        )
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=OPPONENT_MODEL)
        seller_persona = ""
        buyer_persona = opponent_persona
        profiler_agent = seller
    else:
        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=OPPONENT_MODEL)
        buyer = ProfilerAgent(
            agent_name=AGENT_TWO,
            profiler_model=PROFILER_MODEL,
            negotiator_model=NEGOTIATOR_MODEL,
        )
        seller_persona = opponent_persona
        buyer_persona = ""
        profiler_agent = buyer

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

    save_profiler_logs(profiler_agent, log_dir)

    final = game.game_state[-1]
    summary = final.get("summary", final)
    return {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome": summary.get("player_outcome", [None, None])[1],
        "num_turns": len(game.game_state) - 1,
    }


if __name__ == "__main__":
    results = []

    for persona_label, persona_prompt in OPPONENT_PERSONAS.items():
        for profiler_is_seller in [True, False]:
            role = "seller" if profiler_is_seller else "buyer"
            label = f"{role}_vs_{persona_label}"

            print(f"\n{'=' * 60}")
            print(f"Scenario: {label}")
            print(f"  ProfilerAgent role: {role}")
            print(f"  Opponent persona:   {persona_label}")
            print(f"  Negotiator: {NEGOTIATOR_MODEL}")
            print(f"  Profiler:   {PROFILER_MODEL}")
            print(f"  Opponent:   {OPPONENT_MODEL}")
            print(f"{'=' * 60}")

            try:
                result = run_scenario(label, persona_prompt, profiler_is_seller)
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

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Scenario':<30} {'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5}"
    )
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['scenario']:<30} {'ERROR':<8}")
        else:
            print(
                f"{r['scenario']:<30} {r['final_response']:<8} "
                f"{r['seller_outcome']:>6} {r['buyer_outcome']:>6} "
                f"{r['num_turns']:>5}"
            )
