import sys
import os
import re
from copy import deepcopy
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "negotiation_arena"))
sys.path.insert(0, PROJECT_ROOT)

from negotiationarena.constants import AGENT_ONE, AGENT_TWO, REASONING_TAG, MESSAGE_TAG


def _mock_openai_client():
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "<message>I offer 10 ZUP.</message><reason>testing</reason>"
    client.chat.completions.create.return_value.choices = [choice]
    return client


@pytest.fixture()
def agent():
    """ProfilerAgent with mocked API clients — no real calls."""
    with patch.dict(os.environ, {"HF_TOKEN": "fake-hf", "OPENAI_API_KEY": "fake-oai"}), \
         patch("openai.OpenAI", return_value=_mock_openai_client()):
        import profiler_agent as pa
        a = pa.ProfilerAgent(
            profiler_model="gpt-4o",
            negotiator_model="meta-llama/Llama-3.1-8B-Instruct",
            agent_name=AGENT_TWO,
        )
        a.client = _mock_openai_client()
        a.profiler_client = _mock_openai_client()
        return a


class TestProfilerPrompt:
    """profiler_prompt() import path and output sanity."""

    def test_callable_and_returns_string(self):
        import profiler_prompt as pp
        result = pp.profiler_prompt(
            possible_strategies=["Collaborating", "Competing"],
            agent_name=AGENT_ONE,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_expected_tags(self):
        import profiler_prompt as pp
        result = pp.profiler_prompt(
            possible_strategies=["Collaborating"],
            agent_name=AGENT_ONE,
        )
        assert "<strategy>" in result or "strategy" in result.lower()
        assert f"<{REASONING_TAG}>" in result
        assert f"<{MESSAGE_TAG}>" in result


class TestRunProfilerFirstTurn:
    """run_profiler() behavior on the very first turn."""

    def test_chat_does_not_crash_on_first_turn(self, agent):
        """chat() should not raise after init_agent as AGENT_TWO."""
        agent.init_agent(system_prompt="You are a buyer.", role=" You go second.")
        assert len(agent.conversation) >= 1
        result = agent.chat()
        assert isinstance(result, str)

    def test_first_turn_logs_system_not_opponent(self, agent):
        """run_profiler() logs conversation[-1] as 'last opponent response', but
        on turn 1 for AGENT_TWO that's actually the system prompt."""
        agent.init_agent(system_prompt="You are a buyer.", role=" You go second.")
        agent.chat()
        logged_msg = agent.profiler_logs[0][0]
        assert logged_msg["role"] == "system"


class TestPromptTagOrdering:
    """Profiler prompt rules vs template tag ordering."""

    def test_rules_and_template_agree_on_order(self):
        """The RULES section and the final template block contradict on tag order."""
        import profiler_prompt as pp
        prompt = pp.profiler_prompt(
            possible_strategies=["Collaborating"],
            agent_name=AGENT_ONE,
        )
        template_match = re.findall(
            r"All the responses.*?```(.*?)```", prompt, re.DOTALL
        )
        assert len(template_match) == 1
        template = template_match[0]

        msg_pos = template.find(f"<{MESSAGE_TAG}>")
        reason_pos = template.find(f"<{REASONING_TAG}>")
        assert reason_pos < msg_pos, (
            f"Template has message at {msg_pos} before reason at {reason_pos}"
        )


class TestDeepCopy:
    """__deepcopy__ converts clients to strings for JSON serialization."""

    def test_deepcopy_converts_clients_to_strings(self, agent):
        """After deepcopy, client and profiler_client should be class-name strings (for JSON serialization)."""
        agent.init_agent(system_prompt="You are a buyer.", role=" You go second.")
        copied = deepcopy(agent)
        assert isinstance(copied.client, str), f"client should be string, got {type(copied.client)}"
        assert isinstance(copied.profiler_client, str), f"profiler_client should be string, got {type(copied.profiler_client)}"

    def test_original_clients_unchanged_after_deepcopy(self, agent):
        """Deepcopy should not affect the original agent's live client objects."""
        agent.init_agent(system_prompt="You are a buyer.", role=" You go second.")
        deepcopy(agent)
        assert not isinstance(agent.client, str), "original client was mutated"
        assert not isinstance(agent.profiler_client, str), "original profiler_client was mutated"
