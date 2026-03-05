from negotiation_arena.negotiationarena.constants import (
    AGENT_ONE,
    AGENT_TWO,
    MONEY_TOKEN,
    REASONING_TAG,
    MESSAGE_TAG,
)
# Refer to negotiationarena/game/prompt.py

STRATEGY_TAG = "strategy"
# TODO: need to change this up
"""
OTHER ARGUMENTS TO CONSIDER TO PUT INTO PROMPT, HOWEVER THESE VARIABLES EXIST IN GAME.PY SO MAY NEED EXTRA WORK
        resources_available_in_game,
        starting_initial_resources,
        player_goal,
        player_social_behaviour
"""


def profiler_prompt(
    possible_strategies,
    agent_name,
):
    prompt = f"""You are a strategist behind an agent that is playing a game where they are buying or selling an object. There is only one object for sale/purchase.
{AGENT_ONE} is going to sell one object. {AGENT_TWO} gives {MONEY_TOKEN} to buy resources. You will be strategizing on behalf of {agent_name}.

RULES:
```
1. Each response from you must always provide:

    i) A probability table. Among the list of possible strategies in {possible_strategies}, estimate the probability that the opponent is using each strategy given the history of the conversation. Probabilities must sum to 1. Your output needs to follow the format:
            <{STRATEGY_TAG}> strategy_name </{STRATEGY_TAG}> : probability

    ii) After the probability table, identify the MOST LIKELY opponent strategy (highest probability) and explicitly state the best counter-strategy to maximize {agent_name}'s payoff against it.

2. You can reason step by step on how you arrived at your conclusions:

<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> add as much text as you want.
This information will not be sent to the other player. It is just for you to keep track of your reasoning.

3. At each turn, send a message to {agent_name} using the following format:

<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}>

Your message MUST include:
    - The full probability table showing the likelihood of each opponent strategy
    - A clear statement of which strategy you believe the opponent is MOST LIKELY using and why
    - A specific, actionable counter-strategy for {agent_name} to use in their next move
```

All the responses you send should contain the following and in this order:
```
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{STRATEGY_TAG}> [full probability table, one strategy per line] </{STRATEGY_TAG}>
<{MESSAGE_TAG}> [add here — must reference the probabilities and state the counter-strategy explicitly] </{MESSAGE_TAG}>
```
"""
    return prompt