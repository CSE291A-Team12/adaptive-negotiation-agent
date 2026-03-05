"""Analyze baseline vs profiler vs compare experiment results.

Loads all results.json files from a run directory and prints:
  1. Cross-scenario summary  (deal rate + % of ZOPA captured, per scenario × mode)
  2. Per-persona breakdown   (aggregated across all scenarios)
  3. Head-to-head delta      (profiler vs baseline, compare vs baseline, in surplus%)
  4. Raw results table

Usage:
    python scripts/analyze_experiment.py                   # latest run
    python scripts/analyze_experiment.py path/to/run_XYZ/
    python scripts/analyze_experiment.py --csv
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

BASE_RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiments"

PERSONA_ORDER = ["neutral", "hardball", "friendly", "sycophant", "stalling"]
MODE_ORDER    = ["baseline", "profiler", "compare"]


# ── Math helpers ──────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else None

def std(values):
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def fmt_val(values, suffix="", plus=True):
    """Format mean ± std, or N/A."""
    if not values:
        return "N/A"
    m = mean(values)
    s = std(values)
    sign = "+" if plus and m >= 0 else ""
    if s is not None:
        return f"{sign}{m:.1f}±{s:.1f}{suffix}"
    return f"{sign}{m:.1f}{suffix}"

def fmt_pct(count, total):
    if total == 0:
        return "N/A"
    return f"{count}/{total}({100*count//total}%)"


# ── Data loading ──────────────────────────────────────────────────────

def find_latest_run():
    if not BASE_RESULTS_DIR.exists():
        return None
    runs = sorted(BASE_RESULTS_DIR.glob("run_*/"), reverse=True)
    return runs[0] if runs else None

def load_results(run_dir):
    records = []
    for path in Path(run_dir).rglob("results.json"):
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
    path = Path(run_dir) / "config.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Stats ─────────────────────────────────────────────────────────────

def surplus_pct(outcome, seller_cost, buyer_wtp):
    """Our outcome as % of available ZOPA. None if ZOPA ≤ 0 or outcome is None."""
    zopa = buyer_wtp - seller_cost
    if zopa <= 0 or outcome is None:
        return None
    return outcome / zopa * 100

def compute_stats(records, mode, self_role, scenario=None, persona=None):
    """Compute stats for a filtered slice of records."""
    subset = [r for r in records if r.get("mode") == mode]
    if scenario is not None:
        subset = [r for r in subset if r.get("scenario") == scenario]
    if persona is not None:
        subset = [r for r in subset if r.get("persona") == persona]
    if not subset:
        return None

    our_outcomes, surplus_pcts, turns_list = [], [], []
    deal_count = 0

    for r in subset:
        s_out = r.get("seller_outcome")
        b_out = r.get("buyer_outcome")
        sc    = r.get("seller_cost", 40)
        bw    = r.get("buyer_wtp",   60)

        our_out = s_out if self_role == "seller" else b_out
        if our_out is not None:
            our_outcomes.append(our_out)

        pct = surplus_pct(our_out, sc, bw)
        if pct is not None:
            surplus_pcts.append(pct)

        if r.get("deal_reached"):
            deal_count += 1

        t = r.get("num_turns")
        if t is not None:
            turns_list.append(t)

    return {
        "n":            len(subset),
        "deal_count":   deal_count,
        "our_outcomes": our_outcomes,
        "surplus_pcts": surplus_pcts,
        "turns":        turns_list,
    }


# ── Display helpers ───────────────────────────────────────────────────

W = 130

def section(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

def divider(w=None):
    print("─" * (w or W))


# ── Main analysis ─────────────────────────────────────────────────────

def analyze(run_dir, save_csv=False):
    run_dir = Path(run_dir)
    config  = load_config(run_dir)
    records = load_results(run_dir)

    if not records:
        print(f"No results.json files found in {run_dir}")
        sys.exit(1)

    self_role = config.get("role", records[0].get("self_role", "seller"))
    modes     = [m for m in MODE_ORDER if any(r.get("mode") == m for r in records)]

    unique_scenarios = list(dict.fromkeys(
        r.get("scenario") for r in records if r.get("scenario")
    ))
    personas = [p for p in PERSONA_ORDER if any(r.get("persona") == p for r in records)]
    for p in set(r.get("persona") for r in records if r.get("persona")):
        if p not in personas:
            personas.append(p)

    # ZOPA per scenario
    sc_meta = {}  # scenario -> (seller_cost, buyer_wtp, zopa)
    for r in records:
        sc = r.get("scenario")
        if sc and sc not in sc_meta:
            cost = r.get("seller_cost", 40)
            wtp  = r.get("buyer_wtp",   60)
            sc_meta[sc] = (cost, wtp, wtp - cost)

    # ── Header ───────────────────────────────────────────────────────
    section("EXPERIMENT ANALYSIS")
    print(f"  Run folder    : {run_dir}")
    print(f"  Timestamp     : {config.get('timestamp', 'N/A')}")
    print(f"  Role          : {self_role}   |   Modes: {', '.join(modes)}")
    print(f"  Scenarios     : {len(unique_scenarios)}   |   Personas: {len(personas)}   |   Records: {len(records)}")
    print()
    for label, key in [("Baseline model  ", "self_model"),
                        ("Compare model   ", "compare_model"),
                        ("Negotiator model", "negotiator_model"),
                        ("Profiler model  ", "profiler_model"),
                        ("Opponent model  ", "opponent_model")]:
        if config.get(key):
            print(f"  {label} : {config[key]}")

    # ── Cross-scenario summary ────────────────────────────────────────
    section("CROSS-SCENARIO SUMMARY  "
            "(surplus% = our_outcome / ZOPA × 100; higher = better for us)")

    # Columns: Scenario | cost | wtp | ZOPA | [per mode: deal%  surplus%  turns]
    COL_SC   = 13
    COL_NUM  = 5
    COL_MODE = 28   # "deal%  surplus%  turns" per mode

    def mode_cell(st):
        if not st:
            return "N/A"
        deal_s    = fmt_pct(st["deal_count"], st["n"])
        surplus_s = fmt_val(st["surplus_pcts"], suffix="%") if st["surplus_pcts"] else "N/A"
        turns_s   = f"{mean(st['turns']):.1f}t" if st["turns"] else "N/A"
        return f"{deal_s}  {surplus_s}  {turns_s}"

    hdr = (f"{'Scenario':<{COL_SC}}  {'cost':>{COL_NUM}}  {'wtp':>{COL_NUM}}  {'ZOPA':>{COL_NUM}}" +
           "".join(f"  {m:^{COL_MODE}}" for m in modes))
    print(hdr)
    divider(len(hdr) + 4)

    scenario_stats = {}
    for scen in unique_scenarios:
        cost, wtp, zopa = sc_meta.get(scen, (40, 60, 20))
        row = f"{scen:<{COL_SC}}  {cost:>{COL_NUM}}  {wtp:>{COL_NUM}}  {zopa:>+{COL_NUM}}"
        scenario_stats[scen] = {}
        for m in modes:
            st = compute_stats(records, m, self_role, scenario=scen)
            scenario_stats[scen][m] = st
            row += f"  {mode_cell(st):<{COL_MODE}}"
        print(row)

    divider(len(hdr) + 4)

    # ── Per-persona breakdown (across all scenarios) ──────────────────
    section("PER-PERSONA BREAKDOWN  "
            "(aggregated across all scenarios; surplus% normalized by per-game ZOPA)")

    COL_P = 12
    COL_M = 28

    hdr2 = f"{'Persona':<{COL_P}}" + "".join(f"  {m:^{COL_M}}" for m in modes)
    sub2 = f"{'':>{COL_P}}" + "".join(f"  {'deal%  surplus%  turns':^{COL_M}}" for _ in modes)
    print(hdr2)
    print(sub2)
    divider()

    persona_stats = {}
    for persona in personas:
        row = f"{persona:<{COL_P}}"
        persona_stats[persona] = {}
        for m in modes:
            st = compute_stats(records, m, self_role, persona=persona)
            persona_stats[persona][m] = st
            row += f"  {mode_cell(st):<{COL_M}}"
        print(row)

    divider()

    # ── Head-to-head delta ────────────────────────────────────────────
    comparators = [m for m in modes if m != "baseline"]
    if "baseline" in modes and comparators:
        section("HEAD-TO-HEAD  "
                "(Δ surplus% vs baseline per scenario; positive = better than baseline)")

        COL_SC2  = 13
        COL_COMP = 16

        hdr3 = (f"{'Scenario':<{COL_SC2}}  {'ZOPA':>5}  {'baseline':>12}" +
                "".join(f"  {'Δ ' + c:>{COL_COMP}}" for c in comparators))
        print(hdr3)
        divider()

        for scen in unique_scenarios:
            _, _, zopa = sc_meta.get(scen, (40, 60, 20))
            b_st   = scenario_stats[scen].get("baseline")
            b_mean = mean(b_st["surplus_pcts"]) if b_st and b_st["surplus_pcts"] else None
            b_str  = f"{b_mean:+.1f}%" if b_mean is not None else "N/A"

            row = f"{scen:<{COL_SC2}}  {zopa:>+5}  {b_str:>12}"
            for comp in comparators:
                c_st   = scenario_stats[scen].get(comp)
                c_mean = mean(c_st["surplus_pcts"]) if c_st and c_st["surplus_pcts"] else None
                if b_mean is not None and c_mean is not None:
                    d = c_mean - b_mean
                    delta_str = f"{'+'if d>=0 else ''}{d:.1f}%"
                else:
                    delta_str = "N/A"
                row += f"  {delta_str:>{COL_COMP}}"
            print(row)

        divider()
        print()
        for comp in comparators:
            all_b = [x for scen in unique_scenarios
                     for x in (scenario_stats[scen].get("baseline") or {}).get("surplus_pcts", [])]
            all_c = [x for scen in unique_scenarios
                     for x in (scenario_stats[scen].get(comp) or {}).get("surplus_pcts", [])]
            if all_b and all_c:
                d = mean(all_c) - mean(all_b)
                print(f"  Overall {comp} vs baseline: {'+'if d>=0 else ''}{d:.1f}%  "
                      f"({'better' if d > 0 else 'worse' if d < 0 else 'tied'})")

    # ── Raw results ───────────────────────────────────────────────────
    section("RAW RESULTS  (every game)")
    print(f"  {'Scenario':<13}  {'Mode':<10}  {'Persona':<12}  {'Run':>3}  "
          f"{'Result':<8}  {'Seller':>6}  {'Buyer':>6}  {'Surplus%':>9}  {'Turns':>5}")
    divider()

    sorted_records = sorted(records, key=lambda r: (
        r.get("scenario", ""),
        r.get("mode", ""),
        r.get("persona", ""),
        r.get("run", 0),
    ))

    for r in sorted_records:
        sc_val  = r.get("seller_cost")
        bw_val  = r.get("buyer_wtp")
        our_out = r.get("seller_outcome") if self_role == "seller" else r.get("buyer_outcome")
        pct     = surplus_pct(our_out, sc_val, bw_val) if sc_val is not None and bw_val is not None else None
        pct_str = f"{pct:+.0f}%" if pct is not None else "N/A"
        s_out   = f"{r['seller_outcome']:>+6}" if r.get("seller_outcome") is not None else f"{'N/A':>6}"
        b_out   = f"{r['buyer_outcome']:>+6}" if r.get("buyer_outcome") is not None else f"{'N/A':>6}"

        print(f"  {r.get('scenario','?'):<13}  {r.get('mode','?'):<10}  "
              f"{r.get('persona','?'):<12}  {r.get('run','?'):>3}  "
              f"{r.get('final_response','?'):<8}  {s_out}  {b_out}  "
              f"{pct_str:>9}  {r.get('num_turns','?'):>5}")

    # ── CSV export ────────────────────────────────────────────────────
    if save_csv:
        csv_path = Path(run_dir) / "analysis.csv"
        with open(csv_path, "w") as f:
            f.write("scenario,seller_cost,buyer_wtp,zopa,mode,persona,run,self_role,"
                    "final_response,seller_outcome,buyer_outcome,surplus_pct,"
                    "num_turns,deal_reached,deal_price\n")
            for r in sorted_records:
                sc_val  = r.get("seller_cost", "")
                bw_val  = r.get("buyer_wtp", "")
                our_out = r.get("seller_outcome") if self_role == "seller" else r.get("buyer_outcome")
                pct     = surplus_pct(our_out, sc_val, bw_val) if isinstance(sc_val, (int, float)) else None
                zopa    = bw_val - sc_val if isinstance(sc_val, (int, float)) and isinstance(bw_val, (int, float)) else ""
                f.write(
                    f"{r.get('scenario','')},{sc_val},{bw_val},{zopa},"
                    f"{r.get('mode','')},{r.get('persona','')},{r.get('run','')},"
                    f"{r.get('self_role','')},{r.get('final_response','')},"
                    f"{r.get('seller_outcome','')},{r.get('buyer_outcome','')},"
                    f"{f'{pct:.1f}' if pct is not None else ''},"
                    f"{r.get('num_turns','')},{r.get('deal_reached','')},"
                    f"{r.get('deal_price','')}\n"
                )
        print(f"\n  CSV saved: {csv_path}")

    print()


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze negotiation experiment results."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="Path to run_TIMESTAMP/ folder (default: latest run)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also save analysis.csv to the run folder",
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
