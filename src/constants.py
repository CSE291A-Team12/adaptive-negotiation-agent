MONEY_TOKEN = "ZUP"

RESOURCES_TAG = "my resources"
GOALS_TAG = "my goals"
REASONING_TAG = "reason"
PLAYER_ANSWER_TAG = "player answer"
PROPOSED_TRADE_TAG = "newly proposed trade"
SPLIT_TAG = "split"
MESSAGE_TAG = "message"
VALUATION_TAG = "my valuation"
REFUSING_OR_WAIT_TAG = "NONE"
ACCEPTING_TAG = "ACCEPT"
REJECTION_TAG = "REJECT"
TURN_OR_MOVE_TAG = "move"
PROPOSAL_COUNT_TAG = "proposal count"
MY_NAME_TAG = "my name"

OTHER_PLAYER_PROPOSED_TRADE = "other player proposed trade"
OTHER_PLAYER_ANSWER = "other player answer"
OTHER_PLAYER_MESSAGE = "other player message"

AGENT_ONE = "Player RED"
AGENT_TWO = "Player BLUE"

_BREVITY = " Keep your <reason> to 2-3 sentences and your <message> to 1-2 sentences."

OPPONENT_PERSONAS = {
    "neutral": _BREVITY,
    "hardball": (
        "You are an aggressive and tough negotiator. Never accept the first offer. "
        "Always push for the lowest price possible. Do not reveal your willingness to pay."
        + _BREVITY
    ),
    "friendly": (
        "You are very kind and generous. Be friendly and helpful with the other player."
        + _BREVITY
    ),
    "sycophant": (
        "You are very agreeable and submissive. Compliment the other player frequently. "
        "Accept offers quickly to maintain a good relationship. Avoid conflict at all costs."
        + _BREVITY
    ),
    "stalling": (
        "You are a cautious negotiator. Delay committing to a final price as long as possible. "
        "Ask clarifying questions, request justifications, and make only small concessions. "
        "When you do propose a trade, always use a valid number."
        + _BREVITY
    ),
}
