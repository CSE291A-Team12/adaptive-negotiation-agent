"""Dynamic strategy stress-test runner for BuySellGame.

Directory layout per run:
    results/profiler_buysell_dynamic/run_TIMESTAMP/
        config.json
        summary.log
        summary.json
        role_<seller|buyer>/
            schedule_<name>/
                vs_<opponent_setting>/
                    run_<N>/
                        dynamic_profiler/
                            framework_logs/<timestamp>/{game_state.json, interaction.log}
                            game.log
                            results.json
                            dynamic_strategy_log.json
                            profiler_logs.json

Usage:
  python scripts/run_profiler_buysell_dynamic.py
  python scripts/run_profiler_buysell_dynamic.py --role seller --schedule zigzag --opponent-setting all --num-runs 3
  python scripts/run_profiler_buysell_dynamic.py --role both --schedule all --opponent-setting hardball --num-runs 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

from dotenv import load_dotenv

from constants import OPPONENT_PERSONAS
from dynamic_strategy_buysell.dynamic_strategy_agent import DynamicStrategyChatGPTAgent
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

MODE_NAME = "dynamic_profiler"


def _latest_log_subdir(log_dir: str):
    if not os.path.isdir(log_dir):
        return None
    subdirs = sorted(
        [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    )
    if not subdirs:
        return None
    return os.path.join(log_dir, subdirs[-1])


def _indent(text: str, prefix: str = "  "):
    return "\n".join(prefix + line for line in text.splitlines())


def _resolve_roles(role_arg: str) -> List[str]:
    return ["seller", "buyer"] if role_arg == "both" else [role_arg]


def _resolve_schedules(schedule_arg: str) -> Dict[str, Dict[int, str]]:
    if schedule_arg == "all":
        return DEFAULT_SCHEDULES
    return {schedule_arg: DEFAULT_SCHEDULES[schedule_arg]}


def _resolve_opponent_settings(setting_arg: str) -> Dict[str, str]:
    if setting_arg == "all":
        return OPPONENT_PERSONAS
    return {setting_arg: OPPONENT_PERSONAS[setting_arg]}


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


def _social_behaviour_for_opponent(
    profiler_role: str, opponent_setting_prompt: str
) -> List[str]:
    # Dynamic agent is opponent: buyer when profiler is seller, seller when profiler is buyer.
    if profiler_role == "seller":
        return ["", opponent_setting_prompt]
    return [opponent_setting_prompt, ""]


def run_single_game(
    case_log_dir: str,
    framework_log_dir: str,
    profiler_role: str,
    schedule_name: str,
    schedule: Dict[int, str],
    opponent_setting_label: str,
    opponent_setting_prompt: str,
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

    social_behaviour = _social_behaviour_for_opponent(
        profiler_role, opponent_setting_prompt
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
        player_social_behaviour=social_behaviour,
        log_dir=framework_log_dir,
    )

    game.run()

    final = game.game_state[-1]
    summary = final.get("summary", final)

    dynamic_log_path = os.path.join(case_log_dir, "dynamic_strategy_log.json")
    with open(dynamic_log_path, "w") as f:
        json.dump(
            {
                "schedule_name": schedule_name,
                "strategy_schedule": schedule,
                "opponent_setting": opponent_setting_label,
                "opponent_setting_prompt": opponent_setting_prompt,
                "default_strategy": default_strategy,
                "switch_controller": switch_controller,
                "switch_model": switch_model,
                "profiler_role": profiler_role,
                "strategy_history": dynamic_agent.strategy_history,
                "switch_events": dynamic_agent.switch_events,
                "switch_decisions": dynamic_agent.switch_decisions,
            },
            f,
            indent=2,
        )

    profiler_log_path = os.path.join(case_log_dir, "profiler_logs.json")
    with open(profiler_log_path, "w") as f:
        json.dump(
            [
                {
                    "opponent_message": msg.get("content", ""),
                    "profiler_output": profiler_output,
                }
                for msg, profiler_output in profiler_agent.profiler_logs
            ],
            f,
            indent=2,
        )

    return {
        "final_response": summary.get("final_response", "N/A"),
        "seller_outcome": summary.get("player_outcome", [None, None])[0],
        "buyer_outcome": summary.get("player_outcome", [None, None])[1],
        "num_turns": len(game.game_state) - 2,
        "dynamic_log_path": dynamic_log_path,
        "profiler_log_path": profiler_log_path,
        "framework_log_path": _latest_log_subdir(framework_log_dir),
        "switch_count": len(dynamic_agent.switch_events),
    }, game, profiler_agent, dynamic_agent


def write_dynamic_game_log(
    log_dir: str,
    run_idx: int,
    game,
    result: Dict[str, object],
    profiler_role: str,
    schedule_name: str,
    schedule: Dict[int, str],
    opponent_setting_label: str,
    opponent_setting_prompt: str,
    switch_controller: str,
    switch_model: str,
    dynamic_agent,
    profiler_agent,
    config: Dict[str, object],
):
    os.makedirs(log_dir, exist_ok=True)

    seller_name = game.players[0].agent_name
    buyer_name = game.players[1].agent_name
    our_name = seller_name if profiler_role == "seller" else buyer_name
    opp_name = buyer_name if profiler_role == "seller" else seller_name

    lines = []
    lines.append("=" * 80)
    lines.append("SETUP")
    lines.append("=" * 80)
    lines.append(f"Mode:               {MODE_NAME}")
    lines.append(f"Our role:           {profiler_role}  (our agent = {our_name})")
    lines.append(
        f"Opponent role:      {'buyer' if profiler_role == 'seller' else 'seller'}  (opponent = {opp_name})"
    )
    lines.append(f"Opponent setting:   {opponent_setting_label}")
    lines.append(f"Run:                {run_idx}")
    lines.append("")
    lines.append("Models:")
    lines.append(f"  Our negotiator:     {config.get('negotiator_model', 'N/A')}")
    lines.append(f"  Profiler brain:     {config.get('profiler_model', 'N/A')}")
    lines.append(f"  Opponent:           {config.get('opponent_model', 'N/A')}")
    lines.append(f"  Switch controller:  {switch_controller}")
    lines.append(f"  Switch model:       {switch_model}")
    lines.append("")
    lines.append("Game parameters:")
    lines.append(f"  Seller cost:        {config.get('seller_cost', '?')} ZUP")
    lines.append(f"  Buyer WTP:          {config.get('buyer_wtp', '?')} ZUP")
    lines.append(f"  Max iterations:     {config.get('iterations', '?')}")
    lines.append("")
    lines.append("Dynamic strategy:")
    lines.append(f"  Schedule name:      {schedule_name}")
    lines.append(f"  Default strategy:   {config.get('default_strategy', 'neutral')}")
    lines.append(f"  Schedule:           {schedule}")
    lines.append("")
    lines.append("Opponent setting prompt:")
    if opponent_setting_prompt and opponent_setting_prompt.strip():
        lines.append(_indent(f'"{opponent_setting_prompt}"'))
    else:
        lines.append("  (neutral - no extra prompt)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("CONVERSATION")
    lines.append("=" * 80)

    profiler_logs = profiler_agent.profiler_logs
    profiler_log_idx = 0

    turn_states = []
    for state in game.game_state:
        ci = state.get("current_iteration", "")
        if ci in ("START", "END"):
            continue
        if "summary" in state and "player_complete_answer" not in state:
            continue
        if "player_complete_answer" in state:
            turn_states.append(state)

    rounds = {}
    for state in turn_states:
        turn = state.get("turn", 0)
        round_num = turn // 2 + 1
        if round_num not in rounds:
            rounds[round_num] = {}
        rounds[round_num]["seller" if turn % 2 == 0 else "buyer"] = state

    for round_num in sorted(rounds.keys()):
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"ROUND {round_num}")
        lines.append("-" * 60)

        seller_state = rounds[round_num].get("seller")
        if seller_state:
            if profiler_role == "seller" and profiler_log_idx < len(profiler_logs):
                _, profiler_output = profiler_logs[profiler_log_idx]
                profiler_log_idx += 1
                lines.append("")
                lines.append(f"[PROFILER ANALYSIS - before {our_name}'s response]")
                lines.append(_indent(profiler_output))

            answer = seller_state.get("player_public_info_dict", {}).get(
                "player answer", "?"
            )
            label = "OUR AGENT" if profiler_role == "seller" else "OPPONENT"
            lines.append("")
            lines.append(f"[{seller_name} - {label} - {answer}]")
            complete = seller_state.get("player_complete_answer", "")
            if complete:
                lines.append(_indent(complete))

        buyer_state = rounds[round_num].get("buyer")
        if buyer_state:
            if profiler_role == "buyer" and profiler_log_idx < len(profiler_logs):
                _, profiler_output = profiler_logs[profiler_log_idx]
                profiler_log_idx += 1
                lines.append("")
                lines.append(f"[PROFILER ANALYSIS - before {our_name}'s response]")
                lines.append(_indent(profiler_output))

            answer = buyer_state.get("player_public_info_dict", {}).get(
                "player answer", "?"
            )
            label = "OUR AGENT" if profiler_role == "buyer" else "OPPONENT"
            lines.append("")
            lines.append(f"[{buyer_name} - {label} - {answer}]")
            complete = buyer_state.get("player_complete_answer", "")
            if complete:
                lines.append(_indent(complete))

    lines.append("")
    lines.append("=" * 80)
    lines.append("DYNAMIC SWITCH TRACE")
    lines.append("=" * 80)
    if dynamic_agent.switch_decisions:
        for decision in dynamic_agent.switch_decisions:
            lines.append(
                f"move={decision.get('move')} switch={decision.get('switch')} "
                f"next={decision.get('next_strategy')} reason={decision.get('reason', '')}"
            )
    else:
        lines.append("(no switch decisions recorded)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("RESULTS")
    lines.append("=" * 80)

    final_response = result.get("final_response", "N/A")
    seller_outcome = result.get("seller_outcome")
    buyer_outcome = result.get("buyer_outcome")
    num_turns = result.get("num_turns")
    deal_reached = final_response == "ACCEPT"
    deal_price = None
    if deal_reached and seller_outcome is not None:
        deal_price = int(config.get("seller_cost", 0)) + int(seller_outcome)

    lines.append(f"Final response:     {final_response}")
    lines.append(f"Seller outcome:     {seller_outcome}")
    lines.append(f"Buyer outcome:      {buyer_outcome}")
    lines.append(f"Number of turns:    {num_turns}")
    lines.append(
        f"Deal price:         {deal_price if deal_price is not None else 'N/A (no deal)'}"
    )
    lines.append(f"Switch count:       {result.get('switch_count', 0)}")

    game_log_path = os.path.join(log_dir, "game.log")
    with open(game_log_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    results_data = {
        "mode": MODE_NAME,
        "run": run_idx,
        "self_role": profiler_role,
        "opponent_setting": opponent_setting_label,
        "schedule_name": schedule_name,
        "strategy_schedule": schedule,
        "switch_controller": switch_controller,
        "switch_model": switch_model,
        "seller_cost": config.get("seller_cost"),
        "buyer_wtp": config.get("buyer_wtp"),
        "final_response": final_response,
        "seller_outcome": seller_outcome,
        "buyer_outcome": buyer_outcome,
        "num_turns": num_turns,
        "deal_reached": deal_reached,
        "deal_price": deal_price,
        "switch_count": result.get("switch_count", 0),
    }
    results_path = os.path.join(log_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    return game_log_path, results_path


def _write_summary(
    path: str, all_results: List[Dict[str, object]], config: Dict[str, object]
):
    with open(path, "w") as f:
        f.write(f"Experiment: run_{config.get('run_id', 'N/A')}\n")
        f.write(
            f"Mode: {MODE_NAME}  |  Roles: {config.get('roles')}  |  Runs per case: {config.get('num_runs')}\n"
        )
        f.write(
            f"Switch controller: {config.get('switch_controller')}  |  Switch model: {config.get('switch_model')}\n"
        )
        f.write(f"Negotiator model:  {config.get('negotiator_model')}\n")
        f.write(f"Profiler model:    {config.get('profiler_model')}\n")
        f.write(f"Opponent model:    {config.get('opponent_model')}\n")
        f.write(
            f"Opponent settings: {', '.join(config.get('opponent_settings', []))}\n"
        )
        f.write(f"Schedules: {', '.join(config.get('schedule_names', []))}\n")
        f.write("=" * 132 + "\n\n")

        f.write(
            f"{'Role':<8} {'Schedule':<24} {'OppSet':<10} {'Run':>3}  {'Result':<8} {'Seller':>6} {'Buyer':>6} {'Turns':>5} {'Sw':>4}\n"
        )
        f.write("-" * 132 + "\n")
        for r in all_results:
            if "error" in r:
                f.write(
                    f"{r.get('role','?'):<8} {r.get('schedule_name','?'):<24} {r.get('opponent_setting','?'):<10} {r.get('run_idx','?'):>3}  {'ERROR':<8}\n"
                )
            else:
                f.write(
                    f"{r.get('role','?'):<8} {r.get('schedule_name','?'):<24} {r.get('opponent_setting','?'):<10} {r.get('run_idx','?'):>3}  "
                    f"{r.get('final_response','?'):<8} {str(r.get('seller_outcome')):>6} "
                    f"{str(r.get('buyer_outcome')):>6} {str(r.get('num_turns')):>5} {str(r.get('switch_count')):>4}\n"
                )

        f.write("\n" + "=" * 132 + "\n")
        f.write("AGGREGATE: mean seller_outcome by (role, schedule, opponent_setting)\n")
        f.write("-" * 132 + "\n")

        agg = defaultdict(list)
        for r in all_results:
            if "error" in r:
                continue
            if r.get("seller_outcome") is not None:
                key = (r["role"], r["schedule_name"], r["opponent_setting"])
                agg[key].append(r["seller_outcome"])

        roles = list(dict.fromkeys([r.get("role", "?") for r in all_results]))
        schedules = list(dict.fromkeys([r.get("schedule_name", "?") for r in all_results]))
        settings = list(dict.fromkeys([r.get("opponent_setting", "?") for r in all_results]))

        f.write(f"{'Role':<8} {'Schedule':<24} {'OppSet':<10} {'Mean Seller':>12} {'N':>4}\n")
        f.write("-" * 132 + "\n")
        for role in roles:
            for schedule_name in schedules:
                for setting in settings:
                    vals = agg.get((role, schedule_name, setting), [])
                    if vals:
                        mean_val = sum(vals) / len(vals)
                        f.write(
                            f"{role:<8} {schedule_name:<24} {setting:<10} {mean_val:>12.2f} {len(vals):>4}\n"
                        )
                    else:
                        f.write(
                            f"{role:<8} {schedule_name:<24} {setting:<10} {'N/A':>12} {0:>4}\n"
                        )


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
        "--opponent-setting",
        default="all",
        help=f"Opponent setting from {{{', '.join(OPPONENT_PERSONAS)}}} or 'all' (default).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Runs per (role, schedule, opponent-setting) tuple (default: 1).",
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
    if args.opponent_setting != "all" and args.opponent_setting not in OPPONENT_PERSONAS:
        valid = ", ".join(sorted(OPPONENT_PERSONAS.keys()))
        raise ValueError(
            f"Unknown opponent-setting '{args.opponent_setting}'. Valid: all, {valid}."
        )

    return args


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.log_root, f"run_{run_id}")
    os.makedirs(run_root, exist_ok=True)

    roles = _resolve_roles(args.role)
    schedules = _resolve_schedules(args.schedule)
    opponent_settings = _resolve_opponent_settings(args.opponent_setting)

    switch_model = args.switch_model if args.switch_model is not None else args.opponent_model

    config = {
        "run_id": run_id,
        "mode": MODE_NAME,
        "role_arg": args.role,
        "roles": roles,
        "schedule_arg": args.schedule,
        "schedule_names": list(schedules.keys()),
        "opponent_setting_arg": args.opponent_setting,
        "opponent_settings": list(opponent_settings.keys()),
        "num_runs": args.num_runs,
        "max_retries": args.max_retries,
        "iterations": args.iterations,
        "seller_cost": args.seller_cost,
        "buyer_wtp": args.buyer_wtp,
        "default_strategy": args.default_strategy,
        "switch_controller": args.switch_controller,
        "switch_model": switch_model,
        "opponent_model": args.opponent_model,
        "negotiator_model": args.negotiator_model,
        "profiler_model": args.profiler_model,
    }

    with open(os.path.join(run_root, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Dynamic strategy profiler experiment")
    print(f"  Run root          : {run_root}")
    print(f"  Roles             : {roles}")
    print(f"  Schedules         : {list(schedules.keys())}")
    print(f"  Opponent settings : {list(opponent_settings.keys())}")
    print(f"  Runs per setting  : {args.num_runs}")
    print(f"  Iterations        : {args.iterations}")
    print(f"  Seller cost/WTP   : {args.seller_cost}/{args.buyer_wtp}")
    print(f"  Opponent model    : {args.opponent_model}")
    print(f"  Switch controller : {args.switch_controller}")
    print(f"  Switch model      : {switch_model}")
    print(f"  Negotiator model  : {args.negotiator_model}")
    print(f"  Profiler model    : {args.profiler_model}")
    print("-" * 90)

    results = []

    for role in roles:
        for schedule_name, schedule in schedules.items():
            for opponent_setting_label, opponent_setting_prompt in opponent_settings.items():
                for run_idx in range(1, args.num_runs + 1):
                    case_name = (
                        f"{role}__{schedule_name}__{opponent_setting_label}__run{run_idx}"
                    )
                    case_log_dir = os.path.join(
                        run_root,
                        f"role_{role}",
                        f"schedule_{schedule_name}",
                        f"vs_{opponent_setting_label}",
                        f"run_{run_idx}",
                        MODE_NAME,
                    )
                    framework_log_dir = os.path.join(case_log_dir, "framework_logs")
                    os.makedirs(framework_log_dir, exist_ok=True)

                    print(
                        f"\n[{case_name}] schedule={schedule} opponent_setting={opponent_setting_label}"
                    )

                    for attempt in range(1, args.max_retries + 1):
                        try:
                            result, game, profiler_agent, dynamic_agent = run_single_game(
                                case_log_dir=case_log_dir,
                                framework_log_dir=framework_log_dir,
                                profiler_role=role,
                                schedule_name=schedule_name,
                                schedule=schedule,
                                opponent_setting_label=opponent_setting_label,
                                opponent_setting_prompt=opponent_setting_prompt,
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

                            game_log_path, results_path = write_dynamic_game_log(
                                log_dir=case_log_dir,
                                run_idx=run_idx,
                                game=game,
                                result=result,
                                profiler_role=role,
                                schedule_name=schedule_name,
                                schedule=schedule,
                                opponent_setting_label=opponent_setting_label,
                                opponent_setting_prompt=opponent_setting_prompt,
                                switch_controller=args.switch_controller,
                                switch_model=switch_model,
                                dynamic_agent=dynamic_agent,
                                profiler_agent=profiler_agent,
                                config=config,
                            )

                            result.update(
                                {
                                    "status": "success",
                                    "case_name": case_name,
                                    "role": role,
                                    "schedule_name": schedule_name,
                                    "opponent_setting": opponent_setting_label,
                                    "schedule": schedule,
                                    "switch_controller": args.switch_controller,
                                    "switch_model": switch_model,
                                    "run_idx": run_idx,
                                    "game_log_path": game_log_path,
                                    "results_path": results_path,
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
                                error_payload = {
                                    "status": "failed",
                                    "case_name": case_name,
                                    "role": role,
                                    "schedule_name": schedule_name,
                                    "opponent_setting": opponent_setting_label,
                                    "schedule": schedule,
                                    "switch_controller": args.switch_controller,
                                    "switch_model": switch_model,
                                    "run_idx": run_idx,
                                    "log_dir": case_log_dir,
                                    "error": str(e),
                                }
                                with open(
                                    os.path.join(case_log_dir, "error.json"), "w"
                                ) as f:
                                    json.dump(error_payload, f, indent=2)
                                results.append(error_payload)

    summary = {
        "run_id": run_id,
        "config": config,
        "results": results,
    }
    summary_path = os.path.join(run_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    summary_log_path = os.path.join(run_root, "summary.log")
    _write_summary(summary_log_path, results, config)

    print("\n" + "=" * 98)
    print("SUMMARY")
    print("=" * 98)
    print(
        f"{'Case':<58} {'Status':<8} {'Resp':<8} {'Seller':>6} {'Buyer':>6} {'Sw':>3}"
    )
    print("-" * 98)
    for item in results:
        if item["status"] == "failed":
            print(f"{item['case_name']:<58} {'FAILED':<8}")
            continue
        print(
            f"{item['case_name']:<58} {'OK':<8} "
            f"{item['final_response']:<8} {item['seller_outcome']:>6} "
            f"{item['buyer_outcome']:>6} {item['switch_count']:>3}"
        )
    print("-" * 98)
    print(f"Summary JSON: {summary_path}")
    print(f"Summary LOG : {summary_log_path}")


if __name__ == "__main__":
    main()
