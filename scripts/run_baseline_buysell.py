"""Baseline: two static-persona GPT-4o-mini agents in a BuySellGame.

Seller (Player RED) has production cost 40 ZUP for resource X.
Buyer (Player BLUE) is willing to pay up to 60 ZUP for resource X.
Both agents use default (neutral) personas -- no social behaviour injection.

Usage:
    python scripts/run_baseline_buysell.py
"""

import sys
import os
import traceback

# Add the submodule to the path so its packages are importable
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
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_buysell")

if __name__ == "__main__":
    try:
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
            player_social_behaviour=["", ""],
            log_dir=LOG_DIR,
        )

        print(f"Running BuySellGame: {MODEL} vs {MODEL}")
        print("Seller cost: 40 ZUP | Buyer WTP: 60 ZUP")
        print(f"Logs will be saved to: {LOG_DIR}")
        print("-" * 50)

        game.run()

        # Print summary from final game state
        final = game.game_state[-1]
        summary = final.get("summary", final)
        print("-" * 50)
        print("Game complete!")
        print(f"Final response: {summary.get('final_response', 'N/A')}")
        print(f"Final resources: {summary.get('final_resources', 'N/A')}")
        print(f"Player outcomes: {summary.get('player_outcome', 'N/A')}")

    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
