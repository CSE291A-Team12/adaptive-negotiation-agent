import sys
import os

# Add project root (for `negotiation_arena.xxx` imports) and
# submodule root (for submodule-internal `negotiationarena.xxx` / `games.xxx` imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

import openai
import negotiationarena.agents.llama2 as Llama
import profiler_prompt

from copy import deepcopy
from negotiationarena.constants import AGENT_TWO, AGENT_ONE


class ProfilerAgent(Llama.LLama2ChatAgent):
    def __init__(self, profiler_model, negotiator_model, **kwargs):

        super().__init__(
            model=negotiator_model, **kwargs
        )  # can set temperature, max tokens, etc
        self.possible_strategies = [
            "Neutral",
            "Hardball",
            "Friendly",
            "Sycophant",
            "Stalling",
        ]
        self.profiler_model = profiler_model
        self.negotiator_model = negotiator_model
        self.profiler_logs = []  # for us to see what profiler agent is responding

        client = openai.OpenAI()
        self.client = client
        self.profiler_client = client

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ("client", "profiler_client") and not isinstance(v, str):
                v = v.__class__.__name__
            setattr(result, k, deepcopy(v, memo))
        return result

    def run_profiler(self):
        # Skip the first message (negotiator system prompt) so it doesn't
        # override the profiler's own instructions.
        conversation_without_system = [
            msg for msg in self.conversation
            if msg.get("role") != self.prompt_entity_initializer
        ]
        messages = [
            {"role": "system", "content": self.profiler_prompt},
        ] + conversation_without_system

        response = self.profiler_client.chat.completions.create(
            model=self.profiler_model,
            messages=messages,
            temperature=0.1,  # Keep profiling consistent
        )

        # Save to profiler logs the last response from opponent
        last_response = self.conversation[-1]
        self.profiler_logs.append((last_response, response.choices[0].message.content))

        return response.choices[0].message.content

    def run_negotiator(self, instructions):
        negotiator_prompt_with_instructions = (
            f"{self.negotiator_prompt}\n\n"
            f"Follow strategic instructions from your profiler: {instructions}"
        )

        response = self.client.chat.completions.create(
            model=self.negotiator_model,
            messages=[
                {"role": "system", "content": negotiator_prompt_with_instructions}
            ]
            + self.conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def init_agent(self, system_prompt, role):
        """
        Roles:
        user: messages from opponent to self
        assistant: message from self to send to opponent
        system: set of instruction/behavior for agent
        """

        self.negotiator_prompt = system_prompt
        self.profiler_prompt = profiler_prompt.profiler_prompt(
            agent_name=self.agent_name, possible_strategies=self.possible_strategies
        )

        if AGENT_ONE in self.agent_name:
            # we use the user role to tell the assistant that it has to start.

            self.update_conversation_tracking(
                self.prompt_entity_initializer, self.negotiator_prompt
            )
            self.update_conversation_tracking("user", role)
        elif AGENT_TWO in self.agent_name:
            self.negotiator_prompt += role
            self.update_conversation_tracking(
                self.prompt_entity_initializer, self.negotiator_prompt
            )
        else:
            raise ValueError("No Player 1 or Player 2 in role")

    def chat(self):

        strategy = self.run_profiler()
        response = self.run_negotiator(strategy)

        return response
