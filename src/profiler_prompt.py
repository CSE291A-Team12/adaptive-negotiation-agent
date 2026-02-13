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
    possible_strategies,  # NEW
    agent_name,  # NEW
):
    prompt = f"""You are a strategist behind an agent that is playing a game where they are buying or selling an object. There is only one object for sale/purcahse.

{AGENT_ONE} is going to sell one object. {AGENT_TWO} gives {MONEY_TOKEN} to buy resources. You will be strategizing on behalf of {agent_name}.

RULES:

```
1. Each response from you must provide :
    
    i) A probability table. Among the list of possible strategies in {possible_strategies}, estimate the probability that the opponent is using each strategy given the history 
of the conversation. Your output needs to follow the format:
            </{STRATEGY_TAG}> : probability \n

    ii) After the probability table, you must offer the best strategy to counter the opponent's response to maximize {agent_name}'s payoff.


2. You can reason step by step on how you arrived to :

<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> add as much text as you want

This information will not be sent to the other player. It is just for you to keep track of your reasoning.

3. At each turn send messages to {agent_name} by using the following format:

<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}> This message can be as much text as you want but 
keep it relevant and explicit. 

```
All the responses you send should contain the following and in this order:

```
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
```
"""

    return prompt
