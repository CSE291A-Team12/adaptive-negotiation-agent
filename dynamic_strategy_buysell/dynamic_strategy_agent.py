from __future__ import annotations

import json
from typing import Dict, Tuple

from negotiationarena.agents.chatgpt import ChatGPTAgent


DEFAULT_STRATEGY_PROMPTS: Dict[str, str] = {
    "neutral": (
        "Be pragmatic and concise. Make reasonable concessions when needed, "
        "while maximizing your own payoff."
    ),
    "hardball": (
        "Be aggressive and tough. Push for a better price, make very small "
        "concessions, and avoid accepting early unless clearly favorable."
    ),
    "friendly": (
        "Be warm and cooperative. Use polite language, offer constructive "
        "compromises, and aim for a mutually acceptable agreement."
    ),
    "stalling": (
        "Delay commitment. Ask clarifying questions, request justification, "
        "and move the price slowly to gather information."
    ),
    "sycophant": (
        "Be highly agreeable and relationship-focused. Prefer harmony over "
        "conflict and consider accepting fair offers quickly."
    ),
}


class DynamicStrategyChatGPTAgent(ChatGPTAgent):
    """
    ChatGPT agent that can switch persona strategy during negotiation.

    Strategy switching is driven by `strategy_schedule`, keyed by the agent's
    own move index (1-based):
        {1: "friendly", 2: "hardball", 4: "stalling"}
    """

    def __init__(
        self,
        strategy_schedule: Dict[int, str],
        default_strategy: str = "neutral",
        strategy_prompts: Dict[str, str] | None = None,
        switch_controller: str = "schedule",
        switch_model: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strategy_prompts = (
            DEFAULT_STRATEGY_PROMPTS.copy()
            if strategy_prompts is None
            else strategy_prompts
        )
        self.default_strategy = default_strategy
        self.strategy_schedule = self._normalize_schedule(strategy_schedule)
        self.switch_controller = switch_controller
        self.switch_model = self.model if switch_model is None else switch_model
        self._validate_schedule()

        self.base_system_prompt = ""
        self.current_strategy = default_strategy
        self.move_count = 0

        # Diagnostics for downstream analysis
        self.strategy_history = []
        self.switch_events = []
        self.switch_decisions = []

    def _normalize_schedule(self, schedule: Dict[int, str]) -> Dict[int, str]:
        normalized = {}
        for move, strategy in schedule.items():
            move_i = int(move)
            if move_i < 1:
                raise ValueError("Strategy schedule moves must be >= 1.")
            normalized[move_i] = strategy
        return dict(sorted(normalized.items()))

    def _validate_schedule(self):
        if self.switch_controller not in {"schedule", "llm"}:
            raise ValueError(
                "switch_controller must be either 'schedule' or 'llm'."
            )
        if self.default_strategy not in self.strategy_prompts:
            raise ValueError(
                f"Unknown default strategy: {self.default_strategy}"
            )
        unknown = [
            strategy
            for strategy in self.strategy_schedule.values()
            if strategy not in self.strategy_prompts
        ]
        if unknown:
            raise ValueError(
                "Unknown strategy labels in schedule: "
                + ", ".join(sorted(set(unknown)))
            )

    def _controller_prompt(self):
        strategies = ", ".join(sorted(self.strategy_prompts.keys()))
        return (
            "You are a strategy-switch controller for a negotiation agent.\n"
            "Task: decide whether to switch strategy before this move.\n"
            f"Allowed strategies: {strategies}\n"
            "Return strict JSON only with keys:\n"
            '  "switch": boolean,\n'
            '  "next_strategy": string,\n'
            '  "reason": string\n'
            "If switch is false, next_strategy must equal current strategy."
        )

    def _extract_json(self, text: str):
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    def _strategy_from_llm(
        self, latest_message: str
    ) -> Tuple[str, Dict[str, object]]:
        payload = {
            "move": self.move_count,
            "current_strategy": self.current_strategy,
            "latest_opponent_message": latest_message,
        }

        raw = self.client.chat.completions.create(
            model=self.switch_model,
            messages=[
                {"role": "system", "content": self._controller_prompt()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
            ],
            temperature=0.0,
            max_tokens=120,
        ).choices[0].message.content

        parsed = self._extract_json(raw or "")
        if not isinstance(parsed, dict):
            decision = {
                "mode": "llm",
                "switch": False,
                "next_strategy": self.current_strategy,
                "reason": "invalid controller output",
                "raw": raw,
            }
            return self.current_strategy, decision

        should_switch = bool(parsed.get("switch", False))
        candidate = parsed.get("next_strategy", self.current_strategy)
        if candidate not in self.strategy_prompts:
            candidate = self.current_strategy
            should_switch = False

        next_strategy = candidate if should_switch else self.current_strategy
        decision = {
            "mode": "llm",
            "switch": bool(should_switch),
            "next_strategy": next_strategy,
            "reason": str(parsed.get("reason", "")).strip(),
            "raw": raw,
        }
        return next_strategy, decision

    def _strategy_for_move(self, move_idx: int) -> str:
        strategy = self.default_strategy
        for switch_move, switch_strategy in self.strategy_schedule.items():
            if move_idx >= switch_move:
                strategy = switch_strategy
            else:
                break
        return strategy

    def _render_system_prompt(self, strategy: str) -> str:
        return (
            f"{self.base_system_prompt}\n\n"
            "Dynamic Strategy Controller (internal instruction):\n"
            f"- Switch controller: {self.switch_controller}\n"
            f"- Current strategy: {strategy}\n"
            f"- Tactical guidance: {self.strategy_prompts[strategy]}\n"
            "- Always obey the game's required output tags and order."
        )

    def _activate_strategy_for_current_move(self, latest_message: str):
        if self.switch_controller == "llm":
            next_strategy, decision = self._strategy_from_llm(latest_message)
        else:
            next_strategy = self._strategy_for_move(self.move_count)
            decision = {
                "mode": "schedule",
                "switch": next_strategy != self.current_strategy,
                "next_strategy": next_strategy,
                "reason": "schedule rule",
            }

        previous = self.current_strategy

        self.current_strategy = next_strategy
        self.conversation[0]["content"] = self._render_system_prompt(
            next_strategy
        )
        self.switch_decisions.append(
            {"move": self.move_count, **decision}
        )
        self.strategy_history.append(
            {
                "move": self.move_count,
                "strategy": next_strategy,
                "controller": self.switch_controller,
            }
        )

        if previous != next_strategy:
            self.switch_events.append(
                {
                    "move": self.move_count,
                    "from": previous,
                    "to": next_strategy,
                    "controller": self.switch_controller,
                }
            )

    def init_agent(self, system_prompt, role):
        super().init_agent(system_prompt, role)
        self.base_system_prompt = self.conversation[0]["content"]
        self.move_count = 0
        self.current_strategy = self.default_strategy
        self.strategy_history = []
        self.switch_events = []
        self.switch_decisions = []
        self.conversation[0]["content"] = self._render_system_prompt(
            self.current_strategy
        )

    def step(self, message):
        self.move_count += 1
        latest_message = "" if message is None else str(message)
        self._activate_strategy_for_current_move(latest_message)
        return super().step(message)
