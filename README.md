# Adaptive Negotiation Agent with Real-Time Strategy Detection

Adaptive negotiation agent using GPT-OSS as a real-time strategy profiler to boost Llama-8B's bargaining performance against stronger opponents. Built on [NegotiationArena](https://github.com/vinid/NegotiationArena).

**CSE 291A -- Group 12**: Sean Ko, Vivian Chen, Wei Dai, Liza Babior, Malcolm Hsiu

## Setup

```bash
git clone --recurse-submodules git@github.com:lizababior/adaptive-negotiation-agent.git
cd adaptive-negotiation-agent
uv sync
cp .env.example .env  # Add your API keys
```

## Project Structure

```
src/               - Source code (agents, profiler, prompts)
configs/           - Experiment configs
scripts/           - Run and evaluate experiments
tests/             - Tests
notebooks/         - Exploration
results/           - Experiment logs (gitignored)
negotiation_arena/ - NegotiationArena (git submodule)
```
