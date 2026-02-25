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


def run_scenario(label, 
                 OPPONENT_MODEL,
                 NEGOTIATOR_MODEL,
                 PROFILER_MODEL,
                 opponent_persona, 
                 profiler_is_seller=True, 
                 seller_cost=40, 
                 buyer_wtp=60, 
                 iterations = 10,
                 ):
    """
    label: 
    opponent_persona: strategy of opponent
    profiler_is_seller: if profiler is seller or not
    seller's cost: how much it took for seller to make
    buyer_wtp: buyer's willingness to pay
    iterations: number of back/forth between agents 
    """
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
        iterations= iterations,
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
        player_social_behaviour=[seller_persona, buyer_persona],
        log_dir=log_dir,
    )

    game.run()

    #save_profiler_logs(profiler_agent, log_dir)
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
    final = game.game_state[-1]
    summary = final.get("summary", final)

    
    return {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome": summary.get("player_outcome", [None, None])[1],
        "num_turns": len(game.game_state) - 1,
    }


def run_profiler_experiment(log_iteration,
                            opponent_model="api-gpt-oss-120b",
                            self_model="meta-llama/Meta-Llama-3-8B-Instruct",
                            profiler_model = "api-gpt-oss-120b",
                            seller_cost = 40, 
                            buyer_wtp = 60, 
                            max_retries=3,
                            iterations = 10):
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    
    OPPONENT_MODEL = opponent_model
    NEGOTIATOR_MODEL = self_model
    PROFILER_MODEL = profiler_model

    for _ in range(1):
        log_file_name = f"run_summary_{log_iteration}.log" 
        log_file_path = os.path.join(BASE_LOG_DIR, log_file_name)
        
        results = {}

        for persona_label, persona_prompt in OPPONENT_PERSONAS.items():
            for profiler_is_seller in [True, False]:
                role = "seller" if profiler_is_seller else "buyer"
                label = f"{role}_vs_{persona_label}"

                for attempt in range(max_retries):
                    try:
                        """if you want the conversation, go to run_scenario and uncomment the block right after game.run()"""
                        result = run_scenario(label,
                                              OPPONENT_MODEL,
                                              NEGOTIATOR_MODEL,
                                              PROFILER_MODEL,
                                              persona_prompt, 
                                              profiler_is_seller, 
                                              seller_cost=seller_cost, 
                                              buyer_wtp=buyer_wtp, 
                                              iterations = iterations)

                        with open(log_file_path, "a") as f:
                            f.write(f"Running ProfilerGame: {NEGOTIATOR_MODEL} vs {OPPONENT_MODEL}\n")
                            f.write(f"Negotiator Model: {NEGOTIATOR_MODEL}\n")
                            f.write(f"Profiler Model: {PROFILER_MODEL}\n")
                            f.write(f"Opponent Model: {OPPONENT_MODEL}\n")
                            f.write(f"Scenario: {label}\n")
                            f.write(f"Opponent Persona: {persona_label}\n")
                            f.write("-" * 50 + "\n")
                            f.write("Seller cost: 40 ZUP | Buyer WTP: 60 ZUP\n")
                            f.write("\n" + "-" * 50 + "\n")
                            f.write("Game complete!\n")
                            f.write(f"Final response: {result['final_response']}\n")
                            f.write(f"Seller outcome: {result['seller_outcome']}\n")
                            f.write(f"Buyer outcome: {result['buyer_outcome']}\n")
                            f.write(f"Turns: {result['num_turns']}\n")
                            f.write("=" * 50 + "\n")

                        results[label] = {"status": "success", **result}
                        break

                    except Exception as e:
                        print(f"  Scenario '{label}' attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
                        if attempt < max_retries - 1:
                            print("  Retrying...")
                        else:
                            print(f"  All {max_retries} attempts failed for '{label}', skipping.")
                            traceback.print_exc()
                            results[label] = {"status": "failed", "error": str(e)}

    return results