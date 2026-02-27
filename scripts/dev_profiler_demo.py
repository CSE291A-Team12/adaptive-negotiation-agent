"""Profiler demo: ProfilerAgent (Llama-3-8B + GPT-4o profiler) vs GPT-4o-mini.

Seller (Player RED) = ProfilerAgent  — Llama-3-8B negotiator, GPT-4o profiler
Buyer  (Player BLUE) = ChatGPTAgent  — GPT-4o-mini (static, no profiling)

Seller cost: 40 ZUP  |  Buyer WTP: 60 ZUP  |  10 iterations

Usage:
    python scripts/run_with_profiler.py
"""

import sys
import os
import traceback

# Add project root (for `negotiation_arena.xxx` imports),
# submodule root (for submodule-internal `negotiationarena.xxx` / `games.xxx` imports),
# and src/ (for `profiler_agent` / `profiler_prompt` imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

from dotenv import load_dotenv

# Load keys from .env
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env")
)

# Use submodule-relative imports (negotiationarena.xxx) to match the framework's
# internal imports — avoids isinstance() failures from duplicate module paths.
from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import (
    AGENT_ONE,
    AGENT_TWO,
    MONEY_TOKEN,
)
from games.buy_sell_game.game import BuySellGame
from profiler_agent import ProfilerAgent

OPPONENT_MODEL = "gpt-4o-mini"
NEGOTIATOR_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
PROFILER_MODEL = "gpt-4o"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "profiler_buysell")

if __name__ == "__main__":
    try:
        # --- Agents ---
        # Seller = ProfilerAgent (our adaptive agent)
        seller = ProfilerAgent(
            agent_name=AGENT_ONE,
            profiler_model=PROFILER_MODEL,
            negotiator_model=NEGOTIATOR_MODEL,
        )

        # Buyer = static GPT-4o-mini opponent
        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=OPPONENT_MODEL)

        # --- Game ---
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
            player_social_behaviour=["", ""],
            log_dir=LOG_DIR,
        )

        print(f"Running BuySellGame: ProfilerAgent vs {OPPONENT_MODEL}")
        print(f"  Negotiator : {NEGOTIATOR_MODEL}")
        print(f"  Profiler   : {PROFILER_MODEL}")
        print(f"  Opponent   : {OPPONENT_MODEL}")
        print("  Seller cost: 40 ZUP | Buyer WTP: 60 ZUP")
        print(f"  Logs: {LOG_DIR}")
        print("-" * 60)

        game.run()

        # --- Results ---
        final = game.game_state[-1]
        summary = final.get("summary", final)
        print("-" * 60)
        print("Game complete!")
        print(f"  Final response : {summary.get('final_response', 'N/A')}")
        print(f"  Final resources: {summary.get('final_resources', 'N/A')}")
        print(f"  Player outcomes: {summary.get('player_outcome', 'N/A')}")

        # --- Profiler logs ---
        print("\n" + "=" * 60)
        print("PROFILER LOGS (strategy detection each turn)")
        print("=" * 60)
        for i, (opponent_msg, profiler_output) in enumerate(seller.profiler_logs):
            print(f"\n--- Turn {i + 1} ---")
            print(f"Opponent said: {opponent_msg.get('content', '')[:200]}...")
            print(f"Profiler analysis:\n{profiler_output[:500]}")

    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
