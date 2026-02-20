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
from negotiationarena.agents.llama2 import LLama2ChatAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import (
    AGENT_ONE,
    AGENT_TWO,
    MONEY_TOKEN,
)
from games.buy_sell_game.game import BuySellGame

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

possible_strategies = [
            "Neutral",
            "Hardball",
            "Friendly",
            "Sycophant",
            "Stalling",
        ]

SELF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
OPPONENT_MODEL = "api-gpt-oss-120b"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_buysell")
print("LOGGING AT: " , LOG_DIR)

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, "run_summary.log")
    MAX_RETRIES = 3 #the small LLama model has formatting issues so we will retry

    for strat in possible_strategies:
        for attempt in range(MAX_RETRIES):
            try:
                #lower the temperature and inreased max tokens 
                seller = ChatGPTAgent(agent_name=AGENT_ONE, model=OPPONENT_MODEL, max_tokens=800)
                buyer = LLama2ChatAgent(agent_name= AGENT_TWO, model=SELF_MODEL, max_tokens=800, temperature=0.3)

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
                    player_social_behaviour=["", strat],
                    log_dir=LOG_DIR,
                )
                game.run()

                final = game.game_state[-1]
                summary = final.get("summary", final)

                # Only log on success
                with open(log_file_path, "a") as f:
                    f.write(f"Running BuySellGame: {OPPONENT_MODEL} vs {SELF_MODEL}\n")
                    f.write(f"Opponent Model: {OPPONENT_MODEL}\n") 
                    f.write(f"Buyer Model: {SELF_MODEL}\n")
                    f.write(f"Opponent Strategy: {strat}\n")
                    f.write("-" * 50 + "\n")
                    f.write("Seller cost: 40 ZUP | Buyer WTP: 60 ZUP\n")
                    
                    """
                    IF WE WANT TO SEE THE CONVERSATION 
                    f.write("-" * 50 + "\n")
                    f.write("CONVERSATION LOG:\n")

                    #logging the entire conversation
                    for i, state in enumerate(game.game_state):
                        f.write(f"\n--- Turn {i + 1} ---\n")
                        for player_name, message in state.items():
                            if player_name != "summary":
                                f.write(f"{player_name}: {message}\n")
                                """
                    
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("Game complete!\n")
                    f.write(f"Final response: {summary.get('final_response', 'N/A')}\n")
                    f.write(f"Final resources: {summary.get('final_resources', 'N/A')}\n")
                    f.write(f"Player outcomes: {summary.get('player_outcome', 'N/A')}\n")
                    f.write("=" * 50 + "\n")

                break 

            except Exception as e:
                print(f"  Strategy '{strat}' attempt {attempt + 1}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retrying...")
                else:
                    print(f"  All {MAX_RETRIES} attempts failed for strategy '{strat}', skipping.")