"""Analyze baseline vs profiler vs compare experiment results.

Loads all results.json files from a run directory and prints:
  0. Data quality / coverage report
  1. Experiment header
  2. ZOPA-band summary     (narrow / medium / wide ZOPA groups)
  3. Cross-scenario table  (deal%, surplus% mean±std, welfare%, turns)
  4. ZOPA-infeasible table (correct-rejection rate for no-deal scenarios)
  5. Per-persona breakdown
  6. Head-to-head delta    (paired within-game delta + sign-test p-value)
  7. Overall aggregate
  8. Raw results table

Metrics:
  surplus%    = our_outcome / ZOPA × 100  (can be < 0 or > 100 if deal outside ZOPA)
  welfare%    = (seller_outcome + buyer_outcome) / ZOPA × 100  (joint surplus captured)
  exp_surplus% = deal_rate × mean_conditional_surplus%  (expected value metric)
  Negative-ZOPA scenarios are analyzed separately (correct behavior = no deal).

Usage:
    python scripts/analyze_experiment.py                   # latest run
    python scripts/analyze_experiment.py path/to/run_XYZ/
    python scripts/analyze_experiment.py --csv
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE_RESULTS_DIR = Path(__file__).parent.parent / "results" / "experiments"

PERSONA_ORDER = ["neutral", "hardball", "friendly", "sycophant", "stalling"]
MODE_ORDER    = ["baseline", "profiler", "compare"]

# ZOPA bands (positive-ZOPA scenarios only)
ZOPA_BANDS = [
    ("narrow",  1,  10),
    ("medium", 11,  18),
    ("wide",   19, 999),
]


# ── Math helpers ──────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else None

def std(values):
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))

def bootstrap_ci(values, n=1000, alpha=0.05, seed=42):
    """Return (mean, ci_low, ci_high) via bootstrap. None if <2 values."""
    if not values:
        return None, None, None
    m = mean(values)
    if len(values) < 2:
        return m, m, m
    rng = random.Random(seed)
    k = len(values)
    boot_means = sorted(
        sum(rng.choice(values) for _ in range(k)) / k
        for _ in range(n)
    )
    lo = boot_means[int(alpha / 2 * n)]
    hi = boot_means[int((1 - alpha / 2) * n)]
    return m, lo, hi

def fmt_mean_std(values, suffix="", plus=False):
    """Format as mean±std or N/A."""
    if not values:
        return "N/A"
    m = mean(values)
    s = std(values)
    sign = "+" if plus and m is not None and m >= 0 else ""
    if s is not None:
        return f"{sign}{m:.1f}±{s:.1f}{suffix}"
    return f"{sign}{m:.1f}{suffix}"

def fmt_ci(values, suffix="", plus=False):
    """Format as mean [lo,hi] 95% CI."""
    if not values:
        return "N/A"
    m, lo, hi = bootstrap_ci(values)
    sign = "+" if plus and m is not None and m >= 0 else ""
    if lo is None or lo == hi:
        return f"{sign}{m:.1f}{suffix}"
    return f"{sign}{m:.1f}{suffix}[{lo:.1f},{hi:.1f}]"

def fmt_pct(count, total):
    if total == 0:
        return "N/A"
    return f"{count}/{total}({100*count//total}%)"

def sign_test_p(diffs):
    """Two-sided sign test p-value for H0: median=0."""
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return None
    k = len(nonzero)
    pos = sum(1 for d in nonzero if d > 0)

    def binom_coeff(n, r):
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result

    if k <= 30:
        extreme = max(pos, k - pos)
        tail = sum(binom_coeff(k, i) for i in range(extreme, k + 1))
        p = 2 * tail / (2 ** k)
        return min(p, 1.0)
    else:
        z = (pos - k / 2) / math.sqrt(k / 4)
        def norm_cdf(x):
            t = 1.0 / (1.0 + 0.2316419 * abs(x))
            poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
            pv = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
            return pv if x >= 0 else 1.0 - pv
        return min(2 * min(norm_cdf(z), 1 - norm_cdf(z)), 1.0)


# ── Data loading ──────────────────────────────────────────────────────

def find_latest_run():
    if not BASE_RESULTS_DIR.exists():
        return None
    runs = sorted(BASE_RESULTS_DIR.glob("run_*/"), reverse=True)
    return runs[0] if runs else None

def annotate_record(rec):
    """Add derived fields to a record in-place."""
    sc  = rec.get("seller_cost")
    bw  = rec.get("buyer_wtp")
    rec["_zopa_positive"] = (sc is not None and bw is not None and bw > sc)
    rec["_zopa"]          = (bw - sc) if (sc is not None and bw is not None) else None
    # Flag deals struck outside the ZOPA — kept in averages but surfaced in report
    sout = rec.get("seller_outcome")
    bout = rec.get("buyer_outcome")
    deal = rec.get("deal_reached", False)
    if deal and sc is not None and bw is not None and sout is not None and bout is not None:
        rec["_outside_zopa"] = (sout < 0 or bout < 0)
    else:
        rec["_outside_zopa"] = False

def load_results(run_dir):
    records = []
    for path in Path(run_dir).rglob("results.json"):
        if "framework_logs" in path.parts:
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
            rec["_path"] = str(path)
            annotate_record(rec)
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


# ── Core metrics ──────────────────────────────────────────────────────

def surplus_pct_val(our_outcome, seller_cost, buyer_wtp):
    """our_outcome / ZOPA × 100. None if ZOPA <= 0 or outcome is None."""
    zopa = buyer_wtp - seller_cost
    if zopa <= 0 or our_outcome is None:
        return None
    return our_outcome / zopa * 100

def welfare_pct_val(seller_outcome, buyer_outcome, seller_cost, buyer_wtp):
    """(seller + buyer) / ZOPA × 100. Should be 100% for any deal, 0% no-deal."""
    zopa = buyer_wtp - seller_cost
    if zopa <= 0 or seller_outcome is None or buyer_outcome is None:
        return None
    return (seller_outcome + buyer_outcome) / zopa * 100

def compute_stats(records, mode, self_role, scenario=None, persona=None,
                  positive_zopa_only=True):
    """
    Compute stats for a filtered slice.

    All deals included regardless of whether outcome is outside ZOPA.
    No-deals contribute outcome=0 → surplus_pct=0%.

    Returns dict with keys:
      n, deal_count, outside_zopa_count,
      surplus_pcts  (all positive-ZOPA games: deals + no-deals),
      conditional_pcts  (deals only, positive ZOPA),
      welfare_pcts  (deals only, positive ZOPA),
      turns, expected_surplus
    """
    subset = [r for r in records if r.get("mode") == mode]
    if scenario is not None:
        subset = [r for r in subset if r.get("scenario") == scenario]
    if persona is not None:
        subset = [r for r in subset if r.get("persona") == persona]
    if positive_zopa_only:
        subset = [r for r in subset if r.get("_zopa_positive")]
    if not subset:
        return None

    surplus_pcts     = []
    conditional_pcts = []
    welfare_pcts     = []
    turns_list       = []
    deal_count       = 0
    outside_zopa_count = 0

    for r in subset:
        sc   = r.get("seller_cost", 40)
        bw   = r.get("buyer_wtp",   60)
        sout = r.get("seller_outcome")
        bout = r.get("buyer_outcome")
        our  = sout if self_role == "seller" else bout

        if r.get("deal_reached"):
            deal_count += 1
            if r.get("_outside_zopa"):
                outside_zopa_count += 1
            pct = surplus_pct_val(our, sc, bw)
            if pct is not None:
                surplus_pcts.append(pct)
                conditional_pcts.append(pct)
            wp = welfare_pct_val(sout, bout, sc, bw)
            if wp is not None:
                welfare_pcts.append(wp)
        else:
            # No-deal: both outcomes are 0
            our_nodeal = our if our is not None else 0
            pct = surplus_pct_val(our_nodeal, sc, bw)
            surplus_pcts.append(pct if pct is not None else 0.0)

        t = r.get("num_turns")
        if t is not None:
            turns_list.append(t)

    n = len(subset)
    cond_mean = mean(conditional_pcts)
    deal_rate = deal_count / n if n > 0 else 0.0
    expected_surplus = (deal_rate * cond_mean) if cond_mean is not None else None

    return {
        "n":                  n,
        "deal_count":         deal_count,
        "outside_zopa_count": outside_zopa_count,
        "surplus_pcts":       surplus_pcts,
        "conditional_pcts":   conditional_pcts,
        "welfare_pcts":       welfare_pcts,
        "turns":              turns_list,
        "expected_surplus":   expected_surplus,
    }


# ── Display helpers ───────────────────────────────────────────────────

W = 150

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

    # Split positive / negative ZOPA
    sc_meta = {}
    for r in records:
        sc = r.get("scenario")
        if sc and sc not in sc_meta:
            cost = r.get("seller_cost", 40)
            wtp  = r.get("buyer_wtp",   60)
            sc_meta[sc] = (cost, wtp, wtp - cost)

    pos_scenarios = [s for s in unique_scenarios if sc_meta.get(s, (0,0,0))[2] > 0]
    neg_scenarios = [s for s in unique_scenarios if sc_meta.get(s, (0,0,0))[2] <= 0]

    # ── DATA QUALITY / COVERAGE REPORT ───────────────────────────────
    section("DATA QUALITY & COVERAGE REPORT")

    outside = [r for r in records if r.get("_outside_zopa")]
    no_deal_pos = [r for r in records if not r.get("deal_reached") and r.get("_zopa_positive")]

    print(f"  Total records              : {len(records)}")
    print(f"  Positive-ZOPA games        : {sum(1 for r in records if r.get('_zopa_positive'))}")
    print(f"  Negative-ZOPA games        : {sum(1 for r in records if not r.get('_zopa_positive'))}")
    print()
    print(f"  Deals struck outside ZOPA  : {len(outside)}  (surplus% outside [0,100]; included in all averages)")
    if outside:
        for r in outside:
            zv   = r.get("_zopa", "?")
            sout = r.get("seller_outcome")
            bout = r.get("buyer_outcome")
            our  = sout if self_role == "seller" else bout
            pct  = f"{our/zv*100:+.0f}%" if isinstance(zv, (int,float)) and zv != 0 and our is not None else "N/A"
            print(f"    {r.get('scenario','?'):<12} {r.get('mode','?'):<10} {r.get('persona','?'):<12} "
                  f"seller={str(sout):>5}  buyer={str(bout):>5}  price={str(r.get('deal_price','?')):>4}  "
                  f"ZOPA={zv:>+5}  our_surplus%={pct}" if isinstance(zv, (int,float)) else f"ZOPA={zv}  our_surplus%={pct}")
    print()
    print(f"  No-deals in positive-ZOPA  : {len(no_deal_pos)}  (missed opportunities; contribute 0% to surplus avg)")
    nd_by_mode = defaultdict(int)
    pos_total_by_mode = defaultdict(int)
    for r in records:
        if r.get("_zopa_positive"):
            pos_total_by_mode[r.get("mode","?")] += 1
            if not r.get("deal_reached"):
                nd_by_mode[r.get("mode","?")] += 1
    for m in modes:
        pt = pos_total_by_mode[m]
        nd = nd_by_mode[m]
        print(f"    {m:<12}: {nd} no-deals / {pt} games ({100*nd//pt if pt else 0}%)")

    # ── EXPERIMENT HEADER ─────────────────────────────────────────────
    section("EXPERIMENT SUMMARY")
    print(f"  Run folder    : {run_dir}")
    print(f"  Timestamp     : {config.get('timestamp', 'N/A')}")
    print(f"  Role          : {self_role}   |   Modes: {', '.join(modes)}")
    print(f"  Scenarios     : {len(unique_scenarios)} "
          f"({len(pos_scenarios)} positive-ZOPA, {len(neg_scenarios)} negative-ZOPA)"
          f"  |  Personas: {len(personas)}  |  Records: {len(records)}")
    print()
    for label, key in [("Baseline model  ", "self_model"),
                        ("Compare model   ", "compare_model"),
                        ("Negotiator model", "negotiator_model"),
                        ("Profiler model  ", "profiler_model"),
                        ("Opponent model  ", "opponent_model")]:
        if config.get(key):
            print(f"  {label} : {config[key]}")

    # ── ZOPA-BAND SUMMARY ─────────────────────────────────────────────
    section("ZOPA-BAND SUMMARY  (positive-ZOPA scenarios grouped by width)")
    print("  surplus%  = our_outcome / ZOPA × 100  (mean over ALL games; no-deals = 0%)")
    print("  cond%     = mean surplus% conditioned on deal being reached")
    print("  exp%      = deal_rate × cond%  (expected value; primary comparison metric)")
    print("  welfare%  = (seller+buyer)/ZOPA × 100  conditioned on deal (≈100% if correct)")
    print()

    COL_B  = 8
    COL_N  = 6
    COL_M  = 50

    hdr_b = (f"{'Band':<{COL_B}}  {'ZOPA rng':>9}  {'N/mode':>{COL_N}}" +
             "".join(f"  {m:^{COL_M}}" for m in modes))
    print(hdr_b)
    divider(len(hdr_b))

    for band_name, lo, hi in ZOPA_BANDS:
        band_scens = [s for s in pos_scenarios if lo <= sc_meta[s][2] <= hi]
        if not band_scens:
            continue
        zr = f"[{lo},{min(hi,99)}]"
        # approximate games per mode in this band
        n_per_mode = sum(
            sum(1 for r in records if r.get("scenario") == s and r.get("mode") == modes[0])
            for s in band_scens
        ) if modes else 0

        row = f"{band_name:<{COL_B}}  {zr:>9}  {n_per_mode:>{COL_N}}"
        for m in modes:
            sub = [r for r in records
                   if r.get("scenario") in band_scens
                   and r.get("mode") == m
                   and r.get("_zopa_positive")]
            if not sub:
                row += f"  {'N/A':<{COL_M}}"
                continue

            deals = [r for r in sub if r.get("deal_reached")]
            cond_pcts, welfare_p = [], []
            for r in deals:
                sc_v, bw_v = r.get("seller_cost",40), r.get("buyer_wtp",60)
                our = r.get("seller_outcome") if self_role=="seller" else r.get("buyer_outcome")
                sout, bout = r.get("seller_outcome"), r.get("buyer_outcome")
                p = surplus_pct_val(our, sc_v, bw_v)
                if p is not None: cond_pcts.append(p)
                wp = welfare_pct_val(sout, bout, sc_v, bw_v)
                if wp is not None: welfare_p.append(wp)

            # All games for surplus (no-deals = 0)
            all_pcts = []
            for r in sub:
                sc_v, bw_v = r.get("seller_cost",40), r.get("buyer_wtp",60)
                our = r.get("seller_outcome") if self_role=="seller" else r.get("buyer_outcome")
                if r.get("deal_reached"):
                    p = surplus_pct_val(our, sc_v, bw_v)
                    all_pcts.append(p if p is not None else 0.0)
                else:
                    all_pcts.append(0.0)

            n = len(sub)
            dr = len(deals)
            cond_m = mean(cond_pcts)
            exp_s = (dr/n * cond_m) if cond_m is not None and n > 0 else None
            oz    = sum(1 for r in deals if r.get("_outside_zopa"))
            turns = [r.get("num_turns") for r in sub if r.get("num_turns") is not None]

            dr_s   = fmt_pct(dr, n)
            all_s  = fmt_mean_std(all_pcts, suffix="%")
            cond_s = fmt_mean_std(cond_pcts, suffix="%") if cond_pcts else "N/A"
            exp_s2 = f"{exp_s:.1f}%" if exp_s is not None else "N/A"
            wf_s   = fmt_mean_std(welfare_p, suffix="%") if welfare_p else "N/A"
            t_s    = fmt_mean_std(turns, suffix="t") if turns else "N/A"
            oz_s   = f" oz={oz}" if oz else ""
            cell   = f"deal={dr_s} all={all_s} cond={cond_s} exp={exp_s2} wf={wf_s} itr={t_s}{oz_s}"
            row   += f"  {cell:<{COL_M}}"
        print(row)

    divider()

    # ── CROSS-SCENARIO TABLE ──────────────────────────────────────────
    section("CROSS-SCENARIO SUMMARY  (positive-ZOPA only)")
    print("  Columns per mode: deal% | all_surplus mean±std | cond_surplus mean±std | exp_surplus% | welfare% | turns")
    print("  'all' includes no-deals as 0%; 'cond' = deals only; 'oz' = deals outside ZOPA (still counted)")
    print()

    COL_SC  = 12
    COL_NUM = 5
    COL_M2  = 56

    hdr2 = (f"{'Scenario':<{COL_SC}}  {'cost':>{COL_NUM}}  {'wtp':>{COL_NUM}}  {'ZOPA':>{COL_NUM}}" +
            "".join(f"  {m:^{COL_M2}}" for m in modes))
    print(hdr2)
    divider(len(hdr2) + 4)

    scenario_stats = {}
    for scen in pos_scenarios:
        cost, wtp, zopa = sc_meta.get(scen, (40, 60, 20))
        row = f"{scen:<{COL_SC}}  {cost:>{COL_NUM}}  {wtp:>{COL_NUM}}  {zopa:>+{COL_NUM}}"
        scenario_stats[scen] = {}
        for m in modes:
            st = compute_stats(records, m, self_role, scenario=scen, positive_zopa_only=True)
            scenario_stats[scen][m] = st
            if not st:
                row += f"  {'N/A':<{COL_M2}}"
                continue
            dr_s   = fmt_pct(st["deal_count"], st["n"])
            all_s  = fmt_mean_std(st["surplus_pcts"], suffix="%") if st["surplus_pcts"] else "N/A"
            cond_s = fmt_mean_std(st["conditional_pcts"], suffix="%") if st["conditional_pcts"] else "N/A"
            exp_s  = f"{st['expected_surplus']:.1f}%" if st["expected_surplus"] is not None else "N/A"
            wf_s   = fmt_mean_std(st["welfare_pcts"], suffix="%") if st["welfare_pcts"] else "N/A"
            t_s    = fmt_mean_std(st["turns"], suffix="t") if st["turns"] else "N/A"
            oz_s   = f" oz={st['outside_zopa_count']}" if st["outside_zopa_count"] else ""
            cell   = f"{dr_s}  all={all_s}  cond={cond_s}  exp={exp_s}  wf={wf_s}  itr={t_s}{oz_s}"
            row   += f"  {cell:<{COL_M2}}"
        print(row)

    divider(len(hdr2) + 4)

    # ── ZOPA-INFEASIBLE ───────────────────────────────────────────────
    if neg_scenarios:
        section("ZOPA-INFEASIBLE SCENARIOS  (ZOPA ≤ 0; correct behavior = NO deal)")
        print("  Correct-rejection% = no-deal rate (higher = better for our agent)")
        print()
        COL_N2 = 12
        COL_M3 = 22
        hdr3 = f"{'Scenario':<{COL_N2}}  {'ZOPA':>5}" + "".join(f"  {m:^{COL_M3}}" for m in modes)
        print(hdr3)
        divider()
        for scen in neg_scenarios:
            _, _, zopa = sc_meta.get(scen, (55, 45, -10))
            row = f"{scen:<{COL_N2}}  {zopa:>+5}"
            for m in modes:
                sub = [r for r in records if r.get("scenario") == scen and r.get("mode") == m]
                if not sub:
                    row += f"  {'N/A':<{COL_M3}}"
                    continue
                correct = sum(1 for r in sub if not r.get("deal_reached"))
                deals   = sum(1 for r in sub if r.get("deal_reached"))
                deal_prices = [r.get("deal_price") for r in sub if r.get("deal_reached") and r.get("deal_price")]
                dp_str  = f" prices={deal_prices}" if deal_prices else ""
                row    += f"  {fmt_pct(correct, len(sub))} ok{dp_str:<{COL_M3-len(fmt_pct(correct,len(sub)))-4}}"
            print(row)
        divider()

    # ── PER-PERSONA BREAKDOWN ─────────────────────────────────────────
    section("PER-PERSONA BREAKDOWN  (positive-ZOPA only; across all scenarios)")
    print("  deal% | all_surplus mean±std | cond_surplus mean±std [95%CI] | exp% | welfare% | turns")
    print()

    COL_P  = 12
    COL_M4 = 58

    hdr4 = f"{'Persona':<{COL_P}}" + "".join(f"  {m:^{COL_M4}}" for m in modes)
    print(hdr4)
    divider()

    persona_stats = {}
    for persona in personas:
        row = f"{persona:<{COL_P}}"
        persona_stats[persona] = {}
        for m in modes:
            st = compute_stats(records, m, self_role, persona=persona, positive_zopa_only=True)
            persona_stats[persona][m] = st
            if not st:
                row += f"  {'N/A':<{COL_M4}}"
                continue
            dr_s   = fmt_pct(st["deal_count"], st["n"])
            all_s  = fmt_mean_std(st["surplus_pcts"], suffix="%") if st["surplus_pcts"] else "N/A"
            cond_s = fmt_ci(st["conditional_pcts"], suffix="%") if st["conditional_pcts"] else "N/A"
            exp_s  = f"{st['expected_surplus']:.1f}%" if st["expected_surplus"] is not None else "N/A"
            wf_s   = fmt_mean_std(st["welfare_pcts"], suffix="%") if st["welfare_pcts"] else "N/A"
            t_s    = fmt_mean_std(st["turns"], suffix="t") if st["turns"] else "N/A"
            oz_s   = f" oz={st['outside_zopa_count']}" if st["outside_zopa_count"] else ""
            cell   = f"{dr_s}  all={all_s}  cond={cond_s}  exp={exp_s}  wf={wf_s}  itr={t_s}{oz_s}"
            row   += f"  {cell:<{COL_M4}}"
        print(row)

    divider()

    # ── HEAD-TO-HEAD DELTA + PAIRED ───────────────────────────────────
    comparators = [m for m in modes if m != "baseline"]
    if "baseline" in modes and comparators:
        section("HEAD-TO-HEAD  (Δ vs baseline; all games, positive-ZOPA)")
        print("  Unpaired Δ: difference of overall means (exp_surplus%)")
        print("  Paired Δ:   within-game delta (same scenario+persona+run), surplus% all games")
        print("  p-value:    two-sided sign test on paired deltas")
        print()

        # Build surplus% lookup for all positive-ZOPA games
        lookup = {}
        for r in records:
            if not r.get("_zopa_positive"):
                continue
            sc_v = r.get("seller_cost", 40)
            bw_v = r.get("buyer_wtp", 60)
            our  = r.get("seller_outcome") if self_role == "seller" else r.get("buyer_outcome")
            if r.get("deal_reached"):
                p = surplus_pct_val(our, sc_v, bw_v)
                val = p if p is not None else 0.0
            else:
                val = 0.0
            key = (r.get("scenario"), r.get("persona"), r.get("run"), r.get("mode"))
            lookup[key] = val

        # Per-scenario unpaired delta
        COL_SC3 = 12
        COL_C   = 22
        hdr5 = (f"{'Scenario':<{COL_SC3}}  {'ZOPA':>5}  {'baseline exp%':>14}" +
                "".join(f"  {'Δ '+c+' exp%':^{COL_C}}" for c in comparators))
        print(hdr5)
        divider()

        for scen in pos_scenarios:
            _, _, zopa = sc_meta.get(scen, (40, 60, 20))
            b_st  = scenario_stats[scen].get("baseline")
            b_exp = b_st["expected_surplus"] if b_st else None
            b_str = f"{b_exp:+.1f}%" if b_exp is not None else "N/A"
            row = f"{scen:<{COL_SC3}}  {zopa:>+5}  {b_str:>14}"
            for comp in comparators:
                c_st  = scenario_stats[scen].get(comp)
                c_exp = c_st["expected_surplus"] if c_st else None
                if b_exp is not None and c_exp is not None:
                    d = c_exp - b_exp
                    delta_str = f"{'+'if d>=0 else ''}{d:.1f}%"
                else:
                    delta_str = "N/A"
                row += f"  {delta_str:^{COL_C}}"
            print(row)

        divider()
        print()

        # Paired analysis per comparator
        print("  PAIRED ANALYSIS (matched scenario+persona+run triples):")
        print()

        for comp in comparators:
            keys_b = {(s,pe,ru) for (s,pe,ru,mo) in lookup if mo == "baseline"}
            keys_c = {(s,pe,ru) for (s,pe,ru,mo) in lookup if mo == comp}
            matched = keys_b & keys_c

            diffs = []
            for s, pe, ru in matched:
                b_val = lookup.get((s, pe, ru, "baseline"))
                c_val = lookup.get((s, pe, ru, comp))
                if b_val is not None and c_val is not None:
                    diffs.append(c_val - b_val)

            m_d, lo, hi = bootstrap_ci(diffs)
            p = sign_test_p(diffs)
            pos_d = sum(1 for d in diffs if d > 0)
            neg_d = sum(1 for d in diffs if d < 0)
            ties  = sum(1 for d in diffs if d == 0)

            print(f"  {comp} vs baseline  (n_pairs={len(diffs)})")
            if diffs:
                m_str  = f"{m_d:+.1f}%" if m_d is not None else "N/A"
                ci_str = f"[{lo:.1f}%,{hi:.1f}%]" if lo is not None else ""
                p_str  = f"{p:.3f}" if p is not None else "N/A"
                sig    = "  *p<0.05" if (p is not None and p < 0.05) else ""
                var_d  = std(diffs)
                var_str = f"  std={var_d:.1f}%" if var_d is not None else ""
                print(f"    Mean Δ        : {m_str}  95%CI: {ci_str}{var_str}")
                print(f"    Direction     : {pos_d} games {comp}>baseline, "
                      f"{neg_d} {comp}<baseline, {ties} ties")
                print(f"    Sign-test p   : {p_str}{sig}")
            else:
                print("    No matched pairs found.")
            print()

    # ── OVERALL AGGREGATE ─────────────────────────────────────────────
    section("OVERALL AGGREGATE  (all positive-ZOPA, all personas)")
    print(f"  {'Mode':<12}  {'deal%':>12}  {'all surplus mean±std':>24}  "
          f"{'cond surplus [95%CI]':>28}  {'exp%':>8}  {'welfare mean±std':>20}  "
          f"{'turns mean±std':>16}  {'outside_ZOPA':>12}")
    divider()
    for m in modes:
        st = compute_stats(records, m, self_role, positive_zopa_only=True)
        if not st:
            continue
        dr_s  = fmt_pct(st["deal_count"], st["n"])
        all_s = fmt_mean_std(st["surplus_pcts"], suffix="%") if st["surplus_pcts"] else "N/A"
        ci_s  = fmt_ci(st["conditional_pcts"], suffix="%") if st["conditional_pcts"] else "N/A"
        exp_s = f"{st['expected_surplus']:.1f}%" if st["expected_surplus"] is not None else "N/A"
        wf_s  = fmt_mean_std(st["welfare_pcts"], suffix="%") if st["welfare_pcts"] else "N/A"
        t_s   = fmt_mean_std(st["turns"], suffix="t") if st["turns"] else "N/A"
        print(f"  {m:<12}  {dr_s:>12}  {all_s:>24}  {ci_s:>28}  "
              f"{exp_s:>8}  {wf_s:>20}  {t_s:>16}  {st['outside_zopa_count']:>12}")
    divider()

    # ── RAW RESULTS ───────────────────────────────────────────────────
    section("RAW RESULTS  (every game)")
    print(f"  {'Scenario':<12}  {'Mode':<10}  {'Persona':<12}  {'Run':>3}  "
          f"{'Result':<9}  {'Seller':>6}  {'Buyer':>6}  {'Surplus%':>9}  "
          f"{'Welfare%':>9}  {'Turns':>5}  {'Note'}")
    divider()

    sorted_records = sorted(records, key=lambda r: (
        r.get("scenario", ""),
        r.get("mode", ""),
        r.get("persona", ""),
        r.get("run", 0),
    ))

    for r in sorted_records:
        sc_v  = r.get("seller_cost")
        bw_v  = r.get("buyer_wtp")
        sout  = r.get("seller_outcome")
        bout  = r.get("buyer_outcome")
        our   = sout if self_role == "seller" else bout
        zopa  = r.get("_zopa")

        if r.get("_zopa_positive") and sc_v is not None and bw_v is not None:
            pct = surplus_pct_val(our if our is not None else 0, sc_v, bw_v)
            pct_str = f"{pct:+.0f}%" if pct is not None else "N/A"
            if r.get("deal_reached") and sout is not None and bout is not None:
                wp = welfare_pct_val(sout, bout, sc_v, bw_v)
                wf_str = f"{wp:+.0f}%" if wp is not None else "N/A"
            else:
                wf_str = "0%"
        else:
            pct_str = "N/A"
            wf_str  = "N/A"

        note = ""
        if r.get("_outside_zopa"):
            note = "outside-ZOPA"
        elif not r.get("_zopa_positive") and r.get("deal_reached"):
            note = "neg-ZOPA-deal"
        elif not r.get("_zopa_positive"):
            note = "neg-ZOPA"

        s_str = f"{sout:>+6}" if sout is not None else f"{'N/A':>6}"
        b_str = f"{bout:>+6}" if bout is not None else f"{'N/A':>6}"

        print(f"  {r.get('scenario','?'):<12}  {r.get('mode','?'):<10}  "
              f"{r.get('persona','?'):<12}  {r.get('run','?'):>3}  "
              f"{r.get('final_response','?'):<9}  {s_str}  {b_str}  "
              f"{pct_str:>9}  {wf_str:>9}  {r.get('num_turns','?'):>5}  {note}")

    # ── CSV export ────────────────────────────────────────────────────
    if save_csv:
        csv_path = Path(run_dir) / "analysis.csv"
        with open(csv_path, "w") as f:
            f.write("scenario,seller_cost,buyer_wtp,zopa,mode,persona,run,self_role,"
                    "final_response,seller_outcome,buyer_outcome,surplus_pct,"
                    "welfare_pct,num_turns,deal_reached,deal_price,"
                    "zopa_positive,outside_zopa\n")
            for r in sorted_records:
                sc_v  = r.get("seller_cost", "")
                bw_v  = r.get("buyer_wtp", "")
                sout  = r.get("seller_outcome")
                bout  = r.get("buyer_outcome")
                our   = sout if self_role == "seller" else bout
                zopa  = bw_v - sc_v if isinstance(sc_v, (int,float)) and isinstance(bw_v, (int,float)) else ""
                zpos  = r.get("_zopa_positive", "")

                if zpos and isinstance(sc_v, (int,float)):
                    our_val = our if our is not None else 0
                    pct = surplus_pct_val(our_val, sc_v, bw_v)
                    wp  = welfare_pct_val(sout, bout, sc_v, bw_v) if r.get("deal_reached") else 0.0
                    pct_str = f"{pct:.1f}" if pct is not None else ""
                    wp_str  = f"{wp:.1f}" if wp is not None else "0"
                else:
                    pct_str = ""
                    wp_str  = ""

                f.write(
                    f"{r.get('scenario','')},{sc_v},{bw_v},{zopa},"
                    f"{r.get('mode','')},{r.get('persona','')},{r.get('run','')},"
                    f"{r.get('self_role','')},{r.get('final_response','')},"
                    f"{sout if sout is not None else ''},{bout if bout is not None else ''},"
                    f"{pct_str},{wp_str},"
                    f"{r.get('num_turns','')},{r.get('deal_reached','')},"
                    f"{r.get('deal_price','') if r.get('deal_price') is not None else ''},"
                    f"{zpos},{r.get('_outside_zopa','')}\n"
                )
        print(f"\n  CSV saved: {csv_path}")

    print()


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze negotiation experiment results."
    )
    parser.add_argument(
        "run_dir", nargs="?",
        help="Path to run_TIMESTAMP/ folder (default: latest run)",
    )
    parser.add_argument(
        "--csv", action="store_true",
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
