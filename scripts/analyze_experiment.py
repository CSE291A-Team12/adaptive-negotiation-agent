"""Analyze and compare baseline vs profiler experiment results.

Loads all results.json files from a run directory and prints a formatted
comparison across personas, with per-metric stats and a verdict.

Usage:
    python scripts/analyze_experiment.py                          # latest run
    python scripts/analyze_experiment.py results/experiments/run_20260303_103307/
    python scripts/analyze_experiment.py --csv                    # also save CSV
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

BASE_RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiments"

PERSONA_ORDER = ["neutral", "hardball", "friendly", "sycophant", "stalling"]


# ── Math helpers ─────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else None

def std(values):
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def fmt_mean_std(values, plus_sign=True):
    """Format a list of values as 'mean ± std' or 'mean' if n<2."""
    if not values:
        return "  N/A "
    m = mean(values)
    s = std(values)
    sign = "+" if plus_sign and m >= 0 else ""
    if s is not None:
        return f"{sign}{m:.1f}±{s:.1f}"
    return f"{sign}{m:.1f}"

def fmt_pct(count, total):
    if total == 0:
        return " N/A "
    return f"{count}/{total} ({100*count//total}%)"


# ── Data loading ─────────────────────────────────────────────────────

def find_latest_run():
    if not BASE_RESULTS_DIR.exists():
        return None
    runs = sorted(BASE_RESULTS_DIR.glob("run_*/"), reverse=True)
    return runs[0] if runs else None

def load_results(run_dir):
    """Walk run_dir and collect every results.json into a list of dicts."""
    records = []
    for path in Path(run_dir).rglob("results.json"):
        # skip framework_logs subdirs (they don't contain our results.json)
        if "framework_logs" in path.parts:
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
            rec["_path"] = str(path)
            records.append(rec)
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}", file=sys.stderr)
    return records

def load_config(run_dir):
    config_path = Path(run_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# ── Stats computation ────────────────────────────────────────────────

def compute_persona_stats(records, mode, persona, self_role):
    """
    Returns a stats dict for a given (mode, persona) slice.

    Metrics use 'our_outcome': seller_outcome when role=seller, buyer_outcome when buyer.
    """
    subset = [r for r in records
              if r.get("mode") == mode
              and r.get("persona") == persona]
    if not subset:
        return None

    our_outcomes_all   = []   # includes REJECT (outcome=0)
    our_outcomes_deals = []   # only ACCEPT games
    opp_outcomes_deals = []
    deal_prices        = []
    turns_list         = []
    deal_count         = 0

    for r in subset:
        deal = r.get("deal_reached", False)
        s_out = r.get("seller_outcome")
        b_out = r.get("buyer_outcome")
        turns = r.get("num_turns")
        price = r.get("deal_price")

        our_out = s_out if self_role == "seller" else b_out
        opp_out = b_out if self_role == "seller" else s_out

        # Treat REJECT as 0 outcome for our agent (no trade = no change)
        if our_out is not None:
            our_outcomes_all.append(our_out)

        if deal:
            deal_count += 1
            if our_out is not None:
                our_outcomes_deals.append(our_out)
            if opp_out is not None:
                opp_outcomes_deals.append(opp_out)
            if price is not None:
                deal_prices.append(price)

        if turns is not None:
            turns_list.append(turns)

    return {
        "n":                  len(subset),
        "deal_count":         deal_count,
        "our_outcomes_all":   our_outcomes_all,   # for primary metric (incl. REJECTs as 0)
        "our_outcomes_deals": our_outcomes_deals, # conditional on deal
        "opp_outcomes_deals": opp_outcomes_deals,
        "deal_prices":        deal_prices,
        "turns":              turns_list,
    }


# ── Printing helpers ─────────────────────────────────────────────────

W = 100  # line width

def section(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def divider():
    print("─" * W)


# ── Main analysis ────────────────────────────────────────────────────

def analyze(run_dir, save_csv=False):
    run_dir = Path(run_dir)
    config  = load_config(run_dir)
    records = load_results(run_dir)

    if not records:
        print(f"No results.json files found in {run_dir}")
        sys.exit(1)

    self_role  = config.get("role", records[0].get("self_role", "seller"))
    our_metric = "seller_outcome" if self_role == "seller" else "buyer_outcome"
    opp_metric = "buyer_outcome"  if self_role == "seller" else "seller_outcome"

    modes    = sorted(set(r["mode"]    for r in records if "mode"    in r))
    personas = [p for p in PERSONA_ORDER
                if any(r.get("persona") == p for r in records)]
    # also include any personas not in the canonical order
    extras = [r["persona"] for r in records
              if r.get("persona") and r["persona"] not in PERSONA_ORDER]
    for p in extras:
        if p not in personas:
            personas.append(p)

    has_baseline = "baseline" in modes
    has_profiler = "profiler" in modes

    # ── Header ───────────────────────────────────────────────────────
    section("EXPERIMENT ANALYSIS")
    print(f"  Run folder : {run_dir}")
    print(f"  Timestamp  : {config.get('timestamp', 'N/A')}")
    print(f"  Modes      : {', '.join(modes)}   |   Role: {self_role}")
    print(f"  Runs/persona: {config.get('num_runs', '?')}")
    print()
    if config.get("self_model"):
        print(f"  Baseline model:    {config['self_model']}")
    if config.get("negotiator_model"):
        print(f"  Negotiator model:  {config['negotiator_model']}")
    if config.get("profiler_model"):
        print(f"  Profiler model:    {config['profiler_model']}")
    if config.get("opponent_model"):
        print(f"  Opponent model:    {config['opponent_model']}")
    print()
    seller_cost = config.get("seller_cost", 40)
    buyer_wtp   = config.get("buyer_wtp", 60)
    print(f"  Seller cost: {seller_cost} ZUP   |   Buyer WTP: {buyer_wtp} ZUP   "
          f"|   ZOPA: [{seller_cost}, {buyer_wtp}] ZUP   "
          f"|   Total surplus if deal: {buyer_wtp - seller_cost} ZUP")
    print(f"  Games loaded: {len(records)}")

    # ── Per-persona table ─────────────────────────────────────────────
    section(f"PER-PERSONA COMPARISON  (our role: {self_role}, primary metric: {our_metric})")

    # Column widths
    col_p = 12   # persona
    col_n = 4    # n
    col_out = 12 # outcome mean±std
    col_deal = 10 # deal rate
    col_price = 7 # avg deal price
    col_turns = 6 # avg turns
    col_delta = 9 # delta

    def header_row():
        b = "BASELINE" if has_baseline else ""
        p = "PROFILER" if has_profiler else ""
        d = "DELTA" if (has_baseline and has_profiler) else ""
        print(f"{'Persona':<{col_p}}  {'n':>{col_n}}  "
              f"{'── ' + b + ' ──────────────────':<{col_out+col_deal+col_price+col_turns+6}}  "
              f"{'── ' + p + ' ──────────────────':<{col_out+col_deal+col_price+col_turns+6}}  "
              f"{d}")

    def col_headers():
        seg = (f"{'outcome':>{col_out}}  {'deal%':<{col_deal}}  "
               f"{'price':>{col_price}}  {'turns':>{col_turns}}")
        print(f"{'':>{col_p}}  {'':>{col_n}}  {seg}  {seg}  "
              f"{'Δoutcome':>{col_delta}}")

    header_row()
    col_headers()
    divider()

    all_b_outcomes = []
    all_p_outcomes = []
    all_b_deals    = []
    all_p_deals    = []

    notes = []  # collect interesting observations

    for persona in personas:
        b = compute_persona_stats(records, "baseline", persona, self_role) if has_baseline else None
        p = compute_persona_stats(records, "profiler",  persona, self_role) if has_profiler else None

        n = (b["n"] if b else 0) + (p["n"] if p else 0)

        def fmt_seg(stats):
            if stats is None:
                return f"{'N/A':>{col_out}}  {'':>{col_deal}}  {'':>{col_price}}  {'':>{col_turns}}"
            out_str   = fmt_mean_std(stats["our_outcomes_all"])
            deal_str  = fmt_pct(stats["deal_count"], stats["n"])
            price_str = f"{mean(stats['deal_prices']):>5.1f}" if stats["deal_prices"] else "  N/A"
            turns_str = f"{mean(stats['turns']):>5.1f}"       if stats["turns"]       else "  N/A"
            return (f"{out_str:>{col_out}}  {deal_str:<{col_deal}}  "
                    f"{price_str:>{col_price}}  {turns_str:>{col_turns}}")

        b_seg = fmt_seg(b)
        p_seg = fmt_seg(p)

        delta_str = ""
        if b and p and b["our_outcomes_all"] and p["our_outcomes_all"]:
            delta = mean(p["our_outcomes_all"]) - mean(b["our_outcomes_all"])
            sign  = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.1f}"
            all_b_outcomes.extend(b["our_outcomes_all"])
            all_p_outcomes.extend(p["our_outcomes_all"])

        if b:
            all_b_deals.extend([1 if r.get("deal_reached") else 0
                                 for r in records if r.get("mode") == "baseline"
                                 and r.get("persona") == persona])
        if p:
            all_p_deals.extend([1 if r.get("deal_reached") else 0
                                 for r in records if r.get("mode") == "profiler"
                                 and r.get("persona") == persona])

        # Flag interesting cases
        if b and p:
            b_deal_rate = b["deal_count"] / b["n"] if b["n"] else 0
            p_deal_rate = p["deal_count"] / p["n"] if p["n"] else 0
            if b_deal_rate != p_deal_rate:
                notes.append(f"  {persona}: deal rate differs "
                              f"(baseline {fmt_pct(b['deal_count'], b['n'])}, "
                              f"profiler {fmt_pct(p['deal_count'], p['n'])})")
            # Flag if profiler extracted surplus > total available
            if p["opp_outcomes_deals"]:
                neg_opp = [x for x in p["opp_outcomes_deals"] if x < 0]
                if neg_opp:
                    notes.append(f"  {persona}: profiler pushed opponent below reservation "
                                 f"(opponent outcome {min(neg_opp):.0f} < 0) in {len(neg_opp)} game(s)")
            if b["opp_outcomes_deals"]:
                neg_opp = [x for x in b["opp_outcomes_deals"] if x < 0]
                if neg_opp:
                    notes.append(f"  {persona}: baseline opponent also below reservation "
                                 f"(outcome {min(neg_opp):.0f}) in {len(neg_opp)} game(s)")

        print(f"{persona:<{col_p}}  {n:>{col_n}}  {b_seg}  {p_seg}  {delta_str:>{col_delta}}")

    divider()

    # Overall row
    def overall_seg(outcomes, deals_list):
        if not outcomes:
            return f"{'N/A':>{col_out}}  {'':>{col_deal}}  {'':>{col_price}}  {'':>{col_turns}}"
        deal_count = sum(deals_list)
        total      = len(deals_list)
        out_str    = fmt_mean_std(outcomes)
        deal_str   = fmt_pct(deal_count, total)
        return (f"{out_str:>{col_out}}  {deal_str:<{col_deal}}  "
                f"{'----':>{col_price}}  {'----':>{col_turns}}")

    b_seg_all = overall_seg(all_b_outcomes, all_b_deals)
    p_seg_all = overall_seg(all_p_outcomes, all_p_deals)
    delta_all = ""
    if all_b_outcomes and all_p_outcomes:
        d = mean(all_p_outcomes) - mean(all_b_outcomes)
        delta_all = f"{'+'if d>=0 else ''}{d:.1f}"
    print(f"{'OVERALL':<{col_p}}  {'':>{col_n}}  {b_seg_all}  {p_seg_all}  {delta_all:>{col_delta}}")

    if notes:
        print()
        print("  Notes:")
        for note in notes:
            print(note)

    # ── Win/loss breakdown ────────────────────────────────────────────
    if has_baseline and has_profiler:
        section("HEAD-TO-HEAD: PROFILER vs BASELINE (per persona)")
        print(f"  Metric: {our_metric}")
        print()
        print(f"  {'Persona':<12}  {'Baseline':>9}  {'Profiler':>9}  {'Delta':>7}  Verdict")
        print(f"  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*30}")

        profiler_wins = 0
        ties          = 0
        profiler_loss = 0

        for persona in personas:
            b = compute_persona_stats(records, "baseline", persona, self_role)
            p = compute_persona_stats(records, "profiler",  persona, self_role)
            if not b or not p:
                continue
            bm = mean(b["our_outcomes_all"])
            pm = mean(p["our_outcomes_all"])
            if bm is None or pm is None:
                continue
            delta = pm - bm
            if delta > 0:
                verdict = f"✓ Profiler better by {delta:+.1f}"
                profiler_wins += 1
            elif delta < 0:
                verdict = f"✗ Profiler worse by {delta:+.1f}"
                profiler_loss += 1
            else:
                verdict = "= Tie"
                ties += 1
            print(f"  {persona:<12}  {bm:>+9.1f}  {pm:>+9.1f}  {delta:>+7.1f}  {verdict}")

        print(f"  {'─'*80}")
        total_compared = profiler_wins + ties + profiler_loss
        print(f"  Profiler wins: {profiler_wins}/{total_compared}  |  "
              f"Ties: {ties}/{total_compared}  |  "
              f"Losses: {profiler_loss}/{total_compared}")

    # ── Deal price distribution ───────────────────────────────────────
    section("DEAL PRICE ANALYSIS  (ZOPA midpoint = "
            f"{(seller_cost + buyer_wtp) // 2} ZUP, "
            f"seller favored > midpoint)")
    print(f"  {'Persona':<12}  {'Baseline price':>14}  {'Profiler price':>14}  "
          f"{'Δ price':>8}  {'Who benefits'}  ")
    print(f"  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*30}")

    midpoint = (seller_cost + buyer_wtp) / 2

    for persona in personas:
        b = compute_persona_stats(records, "baseline", persona, self_role) if has_baseline else None
        p = compute_persona_stats(records, "profiler",  persona, self_role) if has_profiler else None

        def price_str(stats):
            if stats is None or not stats["deal_prices"]:
                return f"{'no deal':>14}"
            m = mean(stats["deal_prices"])
            return f"{m:>14.1f}"

        b_price = mean(b["deal_prices"]) if b and b["deal_prices"] else None
        p_price = mean(p["deal_prices"]) if p and p["deal_prices"] else None

        delta_price = ""
        who_benefits = ""
        if b_price is not None and p_price is not None:
            dp = p_price - b_price
            delta_price = f"{dp:>+8.1f}"
            if self_role == "seller":
                who_benefits = "seller ↑" if dp > 0 else ("buyer ↓" if dp < 0 else "neutral")
            else:
                who_benefits = "buyer ↓"  if dp < 0 else ("seller ↑" if dp > 0 else "neutral")

        print(f"  {persona:<12}  {price_str(b)}  {price_str(p)}  "
              f"{delta_price}  {who_benefits}")

    # ── Turns analysis ────────────────────────────────────────────────
    section("NEGOTIATION EFFICIENCY  (fewer turns = faster convergence)")
    print(f"  {'Persona':<12}  {'Baseline turns':>14}  {'Profiler turns':>14}  {'Δ turns':>8}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*8}")

    for persona in personas:
        b = compute_persona_stats(records, "baseline", persona, self_role) if has_baseline else None
        p = compute_persona_stats(records, "profiler",  persona, self_role) if has_profiler else None

        def turns_str(stats):
            if stats is None or not stats["turns"]:
                return f"{'N/A':>14}"
            return f"{mean(stats['turns']):>14.1f}"

        b_t = mean(b["turns"]) if b and b["turns"] else None
        p_t = mean(p["turns"]) if p and p["turns"] else None
        delta_t = f"{p_t - b_t:>+8.1f}" if b_t is not None and p_t is not None else ""

        print(f"  {persona:<12}  {turns_str(b)}  {turns_str(p)}  {delta_t}")

    # ── Raw results table ─────────────────────────────────────────────
    section("RAW RESULTS  (every game)")
    print(f"  {'Mode':<10}  {'Persona':<12}  {'Run':>3}  "
          f"{'Result':<8}  {'Seller':>6}  {'Buyer':>6}  {'Turns':>5}  {'Price':>6}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*3}  "
          f"{'─'*8}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}")

    sorted_records = sorted(
        records,
        key=lambda r: (
            r.get("persona", ""),
            r.get("mode", ""),
            r.get("run", 0),
        )
    )
    for r in sorted_records:
        price = f"{r['deal_price']:>6}" if r.get("deal_price") is not None else f"{'--':>6}"
        s_out = f"{r['seller_outcome']:>+6}" if r.get("seller_outcome") is not None else f"{'N/A':>6}"
        b_out = f"{r['buyer_outcome']:>+6}" if r.get("buyer_outcome") is not None else f"{'N/A':>6}"
        print(f"  {r.get('mode','?'):<10}  {r.get('persona','?'):<12}  "
              f"{r.get('run', '?'):>3}  {r.get('final_response','?'):<8}  "
              f"{s_out}  {b_out}  {r.get('num_turns', '?'):>5}  {price}")

    # ── Verdict ───────────────────────────────────────────────────────
    if has_baseline and has_profiler and all_b_outcomes and all_p_outcomes:
        section("VERDICT")
        b_mean = mean(all_b_outcomes)
        p_mean = mean(all_p_outcomes)
        delta  = p_mean - b_mean
        b_deal_pct = 100 * sum(all_b_deals) / len(all_b_deals) if all_b_deals else 0
        p_deal_pct = 100 * sum(all_p_deals) / len(all_p_deals) if all_p_deals else 0

        print(f"  Baseline mean {our_metric}: {b_mean:+.1f}  (deal rate: {b_deal_pct:.0f}%)")
        print(f"  Profiler mean {our_metric}: {p_mean:+.1f}  (deal rate: {p_deal_pct:.0f}%)")
        print(f"  Overall delta:             {delta:+.1f}  ({'profiler better' if delta > 0 else 'baseline better' if delta < 0 else 'tied'})")
        print()
        print(f"  Profiler outperforms baseline in {profiler_wins}/{total_compared} personas.")
        if p_deal_pct < b_deal_pct:
            print(f"  Note: Profiler deal rate ({p_deal_pct:.0f}%) < baseline ({b_deal_pct:.0f}%) — "
                  "profiler walked away from some deals. Whether this is correct behavior "
                  "depends on whether those deals were below our reservation value.")

    print()

    # ── Optional CSV export ───────────────────────────────────────────
    if save_csv:
        csv_path = Path(run_dir) / "analysis.csv"
        with open(csv_path, "w") as f:
            f.write("mode,persona,run,self_role,final_response,"
                    "seller_outcome,buyer_outcome,num_turns,deal_reached,deal_price\n")
            for r in sorted_records:
                f.write(
                    f"{r.get('mode','')},{r.get('persona','')},{r.get('run','')},"
                    f"{r.get('self_role','')},{r.get('final_response','')},"
                    f"{r.get('seller_outcome','')},"
                    f"{r.get('buyer_outcome','')},"
                    f"{r.get('num_turns','')},"
                    f"{r.get('deal_reached','')},"
                    f"{r.get('deal_price','')}\n"
                )
        print(f"  CSV saved: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline vs profiler experiment results."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to run_TIMESTAMP/ folder (default: latest run)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also save an analysis.csv to the run folder",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            print(f"No experiment runs found in {BASE_RESULTS_DIR}")
            sys.exit(1)
        print(f"Using latest run: {run_dir}")

    if not run_dir.exists():
        print(f"Directory not found: {run_dir}")
        sys.exit(1)

    analyze(run_dir, save_csv=args.csv)


if __name__ == "__main__":
    main()
