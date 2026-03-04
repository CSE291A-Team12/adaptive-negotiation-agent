"""Dynamic strategy stress test runner for BuySellGame.

Usage:
  python scripts/run_profiler_buysell_dynamic.py
  python scripts/run_profiler_buysell_dynamic.py --role seller --schedule zigzag --num-runs 3
  python scripts/run_profiler_buysell_dynamic.py --role both --schedule all --num-runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

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


DEFAULT_SCHEDULES: Dict[str, Dict[int, str]] = {
    "friendly_to_hardball": {1: "friendly", 3: "hardball"},
    "hardball_to_friendly": {1: "hardball", 3: "friendly"},
    "stall_then_friendly": {1: "stalling", 4: "friendly"},
    "zigzag": {1: "friendly", 2: "hardball", 3: "friendly", 4: "hardball"},
}

DEFAULT_STRATEGY_LABELS = [
    "neutral",
    "hardball",
    "friendly",
    "stalling",
    "sycophant",
]

DEFAULT_LOG_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "profiler_buysell_dynamic",
)


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


def _build_players(
    profiler_role: str,
    schedule: Dict[int, str],
    default_strategy: str,
    switch_controller: str,
    switch_model: str,
    opponent_model: str,
    negotiator_model: str,
    profiler_model: str,
):
    if profiler_role == "seller":
        seller = ProfilerAgent(
            agent_name=AGENT_ONE,
            profiler_model=profiler_model,
            negotiator_model=negotiator_model,
        )
        buyer = DynamicStrategyChatGPTAgent(
            agent_name=AGENT_TWO,
            model=opponent_model,
            strategy_schedule=schedule,
            default_strategy=default_strategy,
            switch_controller=switch_controller,
            switch_model=switch_model,
        )
        profiler_agent = seller
        dynamic_agent = buyer
    elif profiler_role == "buyer":
        seller = DynamicStrategyChatGPTAgent(
            agent_name=AGENT_ONE,
            model=opponent_model,
            strategy_schedule=schedule,
            default_strategy=default_strategy,
            switch_controller=switch_controller,
            switch_model=switch_model,
        )
        buyer = ProfilerAgent(
            agent_name=AGENT_TWO,
            profiler_model=profiler_model,
            negotiator_model=negotiator_model,
        )
        profiler_agent = buyer
        dynamic_agent = seller
    else:
        raise ValueError(f"Unknown profiler role: {profiler_role}")

    return seller, buyer, profiler_agent, dynamic_agent


def run_single_game(
    case_log_dir: str,
    profiler_role: str,
    schedule_name: str,
    schedule: Dict[int, str],
    default_strategy: str,
    switch_controller: str,
    switch_model: str,
    opponent_model: str,
    negotiator_model: str,
    profiler_model: str,
    iterations: int,
    seller_cost: int,
    buyer_wtp: int,
):
    seller, buyer, profiler_agent, dynamic_agent = _build_players(
        profiler_role=profiler_role,
        schedule=schedule,
        default_strategy=default_strategy,
        switch_controller=switch_controller,
        switch_model=switch_model,
        opponent_model=opponent_model,
        negotiator_model=negotiator_model,
        profiler_model=profiler_model,
    )

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
        player_social_behaviour=["", ""],
        log_dir=case_log_dir,
    )

    game.run()

    final = game.game_state[-1]
    summary = final.get("summary", final)

    dynamic_log_path = _save_json_in_latest_log(
        case_log_dir,
        "dynamic_strategy_log.json",
        {
            "schedule_name": schedule_name,
            "strategy_schedule": schedule,
            "default_strategy": default_strategy,
            "switch_controller": switch_controller,
            "switch_model": switch_model,
            "profiler_role": profiler_role,
            "strategy_history": dynamic_agent.strategy_history,
            "switch_events": dynamic_agent.switch_events,
            "switch_decisions": dynamic_agent.switch_decisions,
        },
    )
    profiler_log_path = _save_json_in_latest_log(
        case_log_dir,
        "profiler_logs.json",
        [
            {
                "opponent_message": msg.get("content", ""),
                "profiler_output": profiler_output,
            }
            for msg, profiler_output in profiler_agent.profiler_logs
        ],
    )

    return {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome": summary.get("player_outcome", [None, None])[1],
        "num_turns": len(game.game_state) - 1,
        "dynamic_log_path": dynamic_log_path,
        "profiler_log_path": profiler_log_path,
        "switch_count": len(dynamic_agent.switch_events),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dynamic-strategy focused profiler experiments for BuySellGame."
    )
    parser.add_argument(
        "--role",
        choices=["seller", "buyer", "both"],
        default="both",
        help="Role played by ProfilerAgent (default: both).",
    )
    parser.add_argument(
        "--schedule",
        default="all",
        help=f"Schedule name from {{{', '.join(DEFAULT_SCHEDULES)}}} or 'all' (default).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Runs per (role, schedule) pair (default: 1).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per run on failure (default: 2).",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seller-cost", type=int, default=40)
    parser.add_argument("--buyer-wtp", type=int, default=60)
    parser.add_argument(
        "--default-strategy",
        choices=DEFAULT_STRATEGY_LABELS,
        default="neutral",
        help="Default strategy before first scheduled switch.",
    )
    parser.add_argument(
        "--switch-controller",
        choices=["schedule", "llm"],
        default="schedule",
        help="How strategy switching is decided (default: schedule).",
    )
    parser.add_argument(
        "--switch-model",
        default=None,
        help="Model used for LLM switch-controller; defaults to opponent model.",
    )
    parser.add_argument(
        "--opponent-model",
        default="gpt-4o-mini",
        help="Model used by DynamicStrategyChatGPTAgent.",
    )
    parser.add_argument(
        "--negotiator-model",
        default="api-llama-4-scout",
        help="Negotiator model for ProfilerAgent.",
    )
    parser.add_argument(
        "--profiler-model",
        default="api-gpt-oss-120b",
        help="Profiler model for ProfilerAgent.",
    )
    parser.add_argument(
        "--log-root",
        default=DEFAULT_LOG_ROOT,
        help="Root directory for all logs.",
    )
    args = parser.parse_args()

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1.")
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1.")
    if args.schedule != "all" and args.schedule not in DEFAULT_SCHEDULES:
        valid = ", ".join(sorted(DEFAULT_SCHEDULES.keys()))
        raise ValueError(f"Unknown schedule '{args.schedule}'. Valid: all, {valid}.")

    return args


def _resolve_roles(role_arg: str) -> List[str]:
    return ["seller", "buyer"] if role_arg == "both" else [role_arg]


def _resolve_schedules(schedule_arg: str) -> Dict[str, Dict[int, str]]:
    if schedule_arg == "all":
        return DEFAULT_SCHEDULES
    return {schedule_arg: DEFAULT_SCHEDULES[schedule_arg]}


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.log_root, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)

    roles = _resolve_roles(args.role)
    schedules = _resolve_schedules(args.schedule)

    switch_model = (
        args.switch_model
        if args.switch_model is not None
        else args.opponent_model
    )

    print("Dynamic strategy profiler experiment")
    print(f"  Run root         : {run_root}")
    print(f"  Roles            : {roles}")
    print(f"  Schedules        : {list(schedules.keys())}")
    print(f"  Runs per setting : {args.num_runs}")
    print(f"  Iterations       : {args.iterations}")
    print(f"  Seller cost/WTP  : {args.seller_cost}/{args.buyer_wtp}")
    print(f"  Opponent model   : {args.opponent_model}")
    print(f"  Switch controller: {args.switch_controller}")
    print(f"  Switch model     : {switch_model}")
    print(f"  Negotiator model : {args.negotiator_model}")
    print(f"  Profiler model   : {args.profiler_model}")
    print("-" * 80)

    results = []

    for role in roles:
        for schedule_name, schedule in schedules.items():
            for run_idx in range(1, args.num_runs + 1):
                case_name = f"{role}__{schedule_name}__run{run_idx}"
                case_log_dir = os.path.join(run_root, case_name)
                os.makedirs(case_log_dir, exist_ok=True)

                print(f"\n[{case_name}] schedule={schedule}")

                for attempt in range(1, args.max_retries + 1):
                    try:
                        result = run_single_game(
                            case_log_dir=case_log_dir,
                            profiler_role=role,
                            schedule_name=schedule_name,
                            schedule=schedule,
                            default_strategy=args.default_strategy,
                            switch_controller=args.switch_controller,
                            switch_model=switch_model,
                            opponent_model=args.opponent_model,
                            negotiator_model=args.negotiator_model,
                            profiler_model=args.profiler_model,
                            iterations=args.iterations,
                            seller_cost=args.seller_cost,
                            buyer_wtp=args.buyer_wtp,
                        )
                        result.update(
                            {
                                "status": "success",
                                "case_name": case_name,
                                "role": role,
                                "schedule_name": schedule_name,
                                "schedule": schedule,
                                "switch_controller": args.switch_controller,
                                "switch_model": switch_model,
                                "run_idx": run_idx,
                            }
                        )
                        results.append(result)
                        print(
                            "  success: "
                            f"{result['final_response']} | "
                            f"seller={result['seller_outcome']} "
                            f"buyer={result['buyer_outcome']} "
                            f"turns={result['num_turns']} "
                            f"switches={result['switch_count']}"
                        )
                        break
                    except Exception as e:
                        print(
                            f"  attempt {attempt}/{args.max_retries} failed: "
                            f"{type(e).__name__}: {e}"
                        )
                        if attempt == args.max_retries:
                            traceback.print_exc()
                            results.append(
                                {
                                    "status": "failed",
                                    "case_name": case_name,
                                    "role": role,
                                    "schedule_name": schedule_name,
                                    "schedule": schedule,
                                    "switch_controller": args.switch_controller,
                                    "switch_model": switch_model,
                                    "run_idx": run_idx,
                                    "error": str(e),
                                }
                            )

    summary = {
        "run_id": run_id,
        "config": vars(args),
        "results": results,
    }
    summary_path = os.path.join(run_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Case':<42} {'Status':<8} {'Resp':<8} {'Seller':>6} {'Buyer':>6} {'Sw':>3}")
    print("-" * 80)
    for item in results:
        if item["status"] == "failed":
            print(f"{item['case_name']:<42} {'FAILED':<8}")
            continue
        print(
            f"{item['case_name']:<42} {'OK':<8} "
            f"{item['final_response']:<8} {item['seller_outcome']:>6} "
            f"{item['buyer_outcome']:>6} {item['switch_count']:>3}"
        )
    print("-" * 80)
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
