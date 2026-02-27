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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

from dotenv import load_dotenv
from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.agents.llama2 import LLama2ChatAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
from games.buy_sell_game.game import BuySellGame

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_STRATEGIES = ["Neutral", "Hardball", "Friendly", "Sycophant", "Stalling"]


def run_buysell_experiment(
    log_iteration = 1,
    self_is_buyer = False,
    opponent_model="api-gpt-oss-120b",
    self_model="meta-llama/Meta-Llama-3-8B-Instruct",
    seller_cost=40,
    buyer_wtp=60,
    iterations=10,
    max_retries=3,
    temperature=0.3,
    max_tokens=800,
    
):
    """
    Run BuySellGame experiments across a set of strategies. Will run conversation 30 times each call

    Args:
        log_iteration: iteration number of experiment/used for naming log
        self_is_buyer: flag that indicates if self is buyer/seller
        opponent_model: Model string for the seller.
        self_model: Model string for the buyer (LLaMA-based).
        log_dir: Directory to write results. Defaults to ../results/baseline_buysell.
        strategies: List of strategy strings for the buyer. Defaults to all 5 standard strategies.
        seller_cost: Seller's cost of production.
        buyer_wtp: Buyer's willingness to pay.
        iterations: Max negotiation turns per game.
        max_retries: Number of retry attempts per strategy on failure.
        temperature: Sampling temperature for the buyer model.
        max_tokens: Max tokens for both agents.

    Returns:
        dict: A summary mapping each strategy to its result or error.
    """
    
    log_dir = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_buysell")
    
    """IF YOU WANT NEW STRATEGIES, FIX HERE"""
    strategies = DEFAULT_STRATEGIES

    os.makedirs(log_dir, exist_ok=True)
    print(f"LOGGING AT: {log_dir}")
    results = {}
    for _ in range(30):
        log_file_name = f"run_summary{log_iteration}.log"
        log_file_path = os.path.join(log_dir, log_file_name)
        
        log_iteration += 1

        for strat in strategies:
            """We give the strategy to opponent"""
            if self_is_buyer:
                player_social_behavior = ["", strat]
            else:
                player_social_behavior = [strat, ""]

            for attempt in range(max_retries):
                try:
                    if self_is_buyer:
                        seller = ChatGPTAgent(agent_name=AGENT_ONE, model=opponent_model, max_tokens=max_tokens)
                        buyer = LLama2ChatAgent(agent_name=AGENT_TWO, model=self_model, max_tokens=max_tokens, temperature=temperature)
                    
                    else: 
                        seller = LLama2ChatAgent(agent_name= AGENT_ONE, model=self_model, max_tokens=max_tokens, temperature=temperature)
                        buyer = ChatGPTAgent(agent_name=AGENT_TWO, model=opponent_model, max_tokens=max_tokens)


                    game = BuySellGame(
                        players=[seller, buyer],
                        iterations=iterations,
                        player_goals=[
                            SellerGoal(cost_of_production=Valuation({"X": seller_cost})),
                            BuyerGoal(willingness_to_pay=Valuation({"X": buyer_wtp})),
                        ],
                        player_starting_resources=[
                            Resources({"X": 1}),
                            Resources({MONEY_TOKEN: 1000}),
                        ],
                        player_conversation_roles=[
                            f"You are {AGENT_ONE}.",
                            f"You are {AGENT_TWO}.",
                        ],
                        player_social_behaviour= player_social_behavior,
                        log_dir=log_dir,
                    )
                    game.run()

                    final = game.game_state[-1]
                    summary = final.get("summary", final)

                    with open(log_file_path, "a") as f:
                        if self_is_buyer:
                            f.write(f"Running BuySellGame: {opponent_model} vs {self_model}\n")
                            f.write(f"Opponent Model: {opponent_model}\n")
                            f.write(f"Buyer Model: {self_model}\n")
                            f.write(f"Opponent Strategy: {strat}\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"Seller cost: {seller_cost} ZUP | Buyer WTP: {buyer_wtp} ZUP\n")\
                        
                        else:
                            f.write(f"Running BuySellGame:{self_model} vs {opponent_model} \n")
                            
                            f.write(f"Seller Model: {self_model}\n")
                            f.write(f"Opponent Model: {opponent_model}\n")
                            f.write(f"Opponent Strategy: {strat}\n")
                            f.write("-" * 50 + "\n")
                            f.write(f"Seller cost: {seller_cost} ZUP | Buyer WTP: {buyer_wtp} ZUP\n")\
                        

                        
                        """
                        IF WE WANT TO SEE THE CONVERSATION 
                        f.write("-" * 50 + "\n")
                        f.write("CONVERSATION LOG:\n")

                        #logging the entire conversation
                        for i, state in enumerate(game.game_state[:-1]):
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

                    results[strat] = {"status": "success", "summary": summary}
                    break

                except Exception as e:
                    print(f"  Strategy '{strat}' attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
                    if attempt < max_retries - 1:
                        print("  Retrying...")
                    else:
                        print(f"  All {max_retries} attempts failed for strategy '{strat}', skipping.")
                        results[strat] = {"status": "failed", "error": str(e)}

    return results


if __name__ == "__main__":
    run_buysell_experiment()