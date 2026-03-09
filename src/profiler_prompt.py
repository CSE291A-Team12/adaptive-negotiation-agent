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

    i) Identify the MOST LIKELY opponent strategy from {possible_strategies} based on the conversation so far. Use these behavioral signals:
        - Neutral: reasonable opening price, moderate concessions, no emotional language
        - Hardball: extreme anchor price, small or no concessions, pressure tactics, ultimatums
        - Friendly: generous opening, large concessions, warm language, eager to close quickly
        - Sycophant: agrees quickly, compliments frequently, willing to accept unfavorable deals
        - Stalling: asks clarifying questions, avoids committing to a price, tiny concessions, delays

    Output the most likely strategy using:
            <{STRATEGY_TAG}> strategy_name </{STRATEGY_TAG}>

    ii) Explicitly state the best counter-strategy to maximize {agent_name}'s payoff against the identified opponent strategy.

2. You can reason step by step on how you arrived at your conclusions:

<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> add as much text as you want.
This information will not be sent to the other player. It is just for you to keep track of your reasoning.

3. At each turn, send a message to {agent_name} using the following format:

<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}>

Your message MUST include:
    - A specific, actionable counter-strategy for {agent_name} to use in their next move
    - Do NOT mention your analysis of the opponent's strategy in the message — that information is private and belongs only in the <{REASONING_TAG}> section
```

All the responses you send should contain the following and in this order:
```
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{STRATEGY_TAG}> [most likely strategy name] </{STRATEGY_TAG}>
<{MESSAGE_TAG}> [actionable counter-strategy only — do NOT reveal your opponent analysis here] </{MESSAGE_TAG}>
```
"""
    return prompt