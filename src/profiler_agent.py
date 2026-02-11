import sys
import os

# Add the submodule to the path so its packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "negotiation_arena"))

import negotiation_arena.negotiationarena.agents.llama2 as Llama
import profiler_prompt

from negotiation_arena.negotiationarena.constants import AGENT_TWO, AGENT_ONE


class ProfilerAgent(Llama.LLama2ChatAgent):
    def __init__(self, profiler_model, negotiator_model, **kwargs):

        super().__init__(
            model=negotiator_model, **kwargs
        )  # can set temperature, max tokens, etc

        self.profiler_model = profiler_model
        self.negotiator_model = negotiator_model

        self.profiler_logs = []  # for us to see what profiler agents is responding

    def run_profiler(self):

        messages = [self.profiler_prompt, self.conversation]
        response = self.client.chat.completions.create(
            model=self.profiler_model,
            messages=messages,
            temperature=0.1,  # Keep profiling consistent
        )

        # Save to profiler logs the last response from opponent
        last_response = self.conversation[-1]
        self.profiler_logs.append((last_response, response.choices[0].message.content))

        return response.choices[0].message.content

    def run_negotiator(self, instructions):
        # negotiator prompt but with extra instruction from profiler
        negotiator_prompt_with_instructions = (
            f"{self.negotiator_prompt}\n\n"
            f" Follow strategic instructions from your profiler: {instructions}"
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
        system: set of instruction/behavior for agnet
        """

        self.negotiator_prompt = system_prompt
        self.profiler_prompt = (
            profiler_prompt.profiler_prompt()
        )  # still need to work on this function

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
            raise "No Player 1 or Player 2 in role"

    def chat(self):

        strategy = self.run_profiler()
        response = self.run_negotiator(strategy)

        return response
