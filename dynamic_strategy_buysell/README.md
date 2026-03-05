# Dynamic Strategy BuySell

This module adds a dynamic opponent for `BuySellGame` that changes negotiation
style during the same game.

## Files

- `dynamic_strategy_agent.py`: `DynamicStrategyChatGPTAgent` implementation.
- `run_dynamic_strategy_buysell.py`: runnable scenario with dynamic strategy.

## Strategy Schedule

The schedule is move-indexed on the dynamic opponent's own turns:

```python
DYNAMIC_SCHEDULE = {
    1: "friendly",
    2: "hardball",
    4: "stalling",
}
```

This means the opponent starts friendly, switches to hardball on its 2nd move,
then switches to stalling on its 4th move.

## Run

```bash
python dynamic_strategy_buysell/run_dynamic_strategy_buysell.py
```

## Output Logs

In each run directory under `results/dynamic_strategy_buysell/`:

- `dynamic_strategy_log.json`: ground-truth strategy history and switch events.
- `profiler_logs.json`: profiler outputs turn by turn.

