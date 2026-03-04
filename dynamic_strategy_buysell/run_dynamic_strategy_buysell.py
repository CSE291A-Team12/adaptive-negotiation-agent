"""
Run BuySellGame with a dynamic-strategy opponent.

Default setup:
  - Seller: ProfilerAgent
  - Buyer: DynamicStrategyChatGPTAgent

Usage:
  python dynamic_strategy_buysell/run_dynamic_strategy_buysell.py
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena")
)

from dotenv import load_dotenv

from dynamic_strategy_buysell.dynamic_strategy_agent import (
    DynamicStrategyChatGPTAgent,
)
from games.buy_sell_game.game import BuySellGame
from negotiationarena.constants import AGENT_ONE, AGENT_TWO, MONEY_TOKEN
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.valuation import Valuation
from profiler_agent import ProfilerAgent

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

OPPONENT_MODEL = "gpt-4o-mini"
NEGOTIATOR_MODEL = "api-llama-4-scout"
PROFILER_MODEL = "api-gpt-oss-120b"
BASE_LOG_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "dynamic_strategy_buysell"
)

# Move-indexed schedule on the dynamic opponent's own turns.
DYNAMIC_SCHEDULE = {
    1: "friendly",
    2: "hardball",
    4: "stalling",
}


def _latest_log_subdir(log_dir: str):
    if not os.path.isdir(log_dir):
        return None
    subdirs = sorted(
        [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    )
    if not subdirs:
        return None
    return os.path.join(log_dir, subdirs[-1])


def _save_json_in_latest_log(log_dir: str, filename: str, payload):
    target_dir = _latest_log_subdir(log_dir)
    if target_dir is None:
        os.makedirs(log_dir, exist_ok=True)
        target_dir = log_dir
    path = os.path.join(target_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    seller = ProfilerAgent(
        agent_name=AGENT_ONE,
        profiler_model=PROFILER_MODEL,
        negotiator_model=NEGOTIATOR_MODEL,
    )
    buyer = DynamicStrategyChatGPTAgent(
        agent_name=AGENT_TWO,
        model=OPPONENT_MODEL,
        strategy_schedule=DYNAMIC_SCHEDULE,
        default_strategy="neutral",
    )

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
        log_dir=BASE_LOG_DIR,
    )

    print("Running BuySellGame: ProfilerAgent vs DynamicStrategy opponent")
    print(f"  Negotiator model : {NEGOTIATOR_MODEL}")
    print(f"  Profiler model   : {PROFILER_MODEL}")
    print(f"  Dynamic opponent : {OPPONENT_MODEL}")
    print(f"  Strategy schedule: {DYNAMIC_SCHEDULE}")
    print(f"  Logs             : {BASE_LOG_DIR}")
    print("-" * 70)

    game.run()

    final = game.game_state[-1]
    summary = final.get("summary", final)
    print("-" * 70)
    print("Game complete!")
    print(f"  Final response : {summary.get('final_response', 'N/A')}")
    print(f"  Final resources: {summary.get('final_resources', 'N/A')}")
    print(f"  Player outcomes: {summary.get('player_outcome', 'N/A')}")

    dynamic_log_path = _save_json_in_latest_log(
        BASE_LOG_DIR,
        "dynamic_strategy_log.json",
        {
            "strategy_schedule": DYNAMIC_SCHEDULE,
            "strategy_history": buyer.strategy_history,
            "switch_events": buyer.switch_events,
        },
    )
    profiler_log_path = _save_json_in_latest_log(
        BASE_LOG_DIR,
        "profiler_logs.json",
        [
            {
                "opponent_message": msg.get("content", ""),
                "profiler_output": profiler_output,
            }
            for msg, profiler_output in seller.profiler_logs
        ],
    )

    print("\nSaved analysis logs:")
    print(f"  - {dynamic_log_path}")
    print(f"  - {profiler_log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

