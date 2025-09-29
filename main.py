# SEO Impact Forecaster – Streamlit App
# Author: Brandon Lazovic + ChatGPT (GPT-5 Thinking)

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List

st.set_page_config(page_title="SEO Impact Forecaster", layout="wide")
st.title("SEO Impact Forecaster")
st.caption("Forecast traffic → RTA submits → job closes → revenue from ranking improvements, rollouts, sitewide effects, and section-level RTA rates.")

# ---------- Global CSS: left-align all st.dataframe cells ----------
st.markdown("""
<style>
.stDataFrame table th, .stDataFrame table td { text-align: left !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Quick FAQ ----------
with st.expander("❓ Quick FAQ — What this is & how to use it"):
    st.markdown("""
**What it does**  
Estimates *traffic → RTA submits → job closes → revenue* from SEO ranking improvements, factoring in: rollout phases, a sitewide classifier milestone curve, a runway delay, and a post-runway realization ramp.

**How to use**  
1) Upload **Keyword CSV** with: **URL, SV, Current Rank, Page Template**.  
2) *(Optional)* Upload a **Master URL list (one URL per row)** to drive migration % off your true page inventory.  
3) Choose **grouping**: **Site Section** (derived from URL; single-segment = `misc`) or **Page Template**.  
4) Tune **RTA rates** (global + per-group), **close rate**, and **revenue bounds**.  
5) Define **rollout phases** (who ships when).  
6) Choose **runway** (no gains during 3/6 months) and **ramp** (Step/Linear/S-curve) for post-runway realization.  
7) Use **dampening** + **classifier caps (30/50/70/90%)** for conservative modeling.

**Runway vs Ramp**  
- *Runway* = hard delay (no incremental gains).  
- *Ramp* = shape of realization after runway ends (Step=instant, Linear=steady, S-curve=slow→fast→plateau).

**Classifier caps**  
Realized gains are capped by **overall % of site migrated**. As migration crosses 30/50/70/90%, the cap increases.

**Counts are integers**  
RTA submits & job closes are **floored monthly**; revenue uses those integers.

**CTR curve**  
Global SERP CTR curve for positions 1–20. Default sums to ~0.40 (~60% zero-click).
""")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_sample_ctr() -> pd.DataFrame:
    base = pd.DataFrame({
        "Position": list(range(1, 21)),
        "CTR": [0.28, 0.14, 0.09, 0.06, 0.04, 0.028, 0.022, 0.018, 0.015, 0.012,
                0.010, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.0035, 0.003, 0.0025]
    })
    total = base["CTR"].sum()
    base["CTR"] = base["CTR"] * (0.40 / total)
    return base

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    required = {"URL", "SV", "Current Rank", "Page Template"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["SV"] = pd.to_numeric(df["SV"], errors="coerce").fillna(0).astype(float)
    df["Current Rank"] = pd.to_numeric(df["Current Rank"], errors="coerce").fillna(100).astype(float)
    df["Page Template"] = df["Page Template"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_master_urls(file_bytes: bytes) -> pd.DataFrame:
    # support either a single 'URL'/'URLs' column or unnamed first column
    df = pd.read_csv(io.BytesIO(file_bytes))
    # pick the first non-empty column as URLs
    url_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("url", "urls"):
            url_col = c
            break
    if url_col is None:
        url_col = df.columns[0]
    out = pd.DataFrame({"URL": df[url_col].astype(str)})
    return out

# Rank scenario rules (mild)
def rank_delta_rules(rank: float, flavor: str) -> int:
    if flavor == "conservative":
        if 15 <= rank <= 20: return 1
        if 11 <= rank <= 14: return 1
        if 5  <= rank <= 9:  return 0
        return 0
    if flavor == "expected":
        if 15 <= rank <= 20: return 2
        if 11 <= rank <= 14: return 1
        if 5  <= rank <= 9:  return 1
        return 0
    if flavor == "aggressive":
        if 1  <= rank <= 4:  return 1
        if 15 <= rank <= 20: return 3
        if 11 <= rank <= 14: return 2
        if 5  <= rank <= 9:  return 1
        return 0
    return 0

# Grouping utils
def extract_site_section(url: str) -> str:
    try:
        u = str(url)
        if '//' in u: u = u.split('//', 1)[1]
        path = '/' + u.split('/', 1)[1] if '/' in u else '/'
        path = path.split('?', 1)[0].split('#', 1)[0]
        segs = path.strip('/').split('/') if path else []
        if len(segs) == 0 or segs[0] == '': return 'homepage'
        if len(segs) == 1:                  return 'misc'
        return segs[0].lower()
    except Exception:
        return 'unknown'

@st.cache_data(show_spinner=False)
def build_group_ctr(groups: List[str], default_ctr: pd.DataFrame) -> pd.DataFrame:
    pos = default_ctr[["Position"]].copy()
    data = {g: default_ctr["CTR"].values for g in groups}
    return pd.concat([pos, pd.DataFrame(data)], axis=1)

@st.cache_data(show_spinner=False)
def default_revenue_bounds(avg_revenue: float) -> Dict[str, float]:
    return {"min": avg_revenue * 0.8, "avg": avg_revenue, "max": avg_revenue * 1.2}

def expand_rollout_phases(phases: List[Dict], groups: List[str]) -> pd.DataFrame:
    months = list(range(1, 13))
    rows, current_month = [], 1
    live_month = {g: 13 for g in groups}
    for phase in phases:
        dur = max(1, int(phase.get("months", 1)))
        phase_groups = phase.get("groups", [])
        lm = min(12, current_month + dur - 1)
        for g in phase_groups:
            live_month[g] = min(live_month.get(g, 13), lm)
        current_month += dur
        if current_month > 12: break
    for m in months:
        for g in groups:
            rows.append({"Month": m, "Group": g, "IsLive": 1 if m >= live_month[g] else 0})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def default_classifier_multipliers() -> Dict[int, float]:
    return {30: 0.3, 50: 0.5, 70: 0.7, 90: 1.0}

def compute_overall_migrated_pct(rollout_df: pd.DataFrame, counts_by_group: pd.Series) -> pd.DataFrame:
    live_share = []
    total = max(1, counts_by_group.sum())
    for m, g in rollout_df.groupby("Month"):
        live_groups = g.loc[g["IsLive"] == 1, "Group"].tolist()
        share = counts_by_group.loc[counts_by_group.index.isin(live_groups)].sum() / total
        live_share.append({"Month": m, "OverallMigratedPct": float(share)})
    return rollout_df.merge(pd.DataFrame(live_share), on="Month", how="left")

def classifier_multiplier_for_pct(pct: float, milestones: Dict[int, float]) -> float:
    pct100 = pct * 100
    applicable = [k for k in milestones.keys() if k <= pct100]
    if not applicable: return 0.0
    return float(milestones[max(applicable)])

def realization_factor(month: int, runway: int, mode: str, ramp_months: int) -> float:
    if month <= runway: return 0.0
    t = max(0, month - runway)
    if mode == "Step":   return 1.0
    if mode == "Linear": return float(min(1.0, t / max(1, ramp_months)))
    x = float(min(1.0, t / max(1, ramp_months)))  # S-curve
    return 3 * x * x - 2 * x * x * x

def apply_rank_shift(current_rank: np.ndarray, scenario: str) -> np.ndarray:
    if scenario == "Baseline": return current_rank.copy()
    flavor = {"Conservative": "conservative", "Expected": "expected", "Aggressive": "aggressive"}.get(scenario, "expected")
    f = np.frompyfunc(lambda r: max(1.0, r - rank_delta_rules(float(r), flavor)), 1, 1)
    return f(current_rank).astype(float)

def ctr_from_curve(cur_rank: np.ndarray, ctr_col: pd.Series) -> np.ndarray:
    max_pos = int(ctr_col.index.max())
    pos = np.clip(np.rint(cur_rank).astype(int), 1, max_pos)
    return ctr_col.reindex(pos).to_numpy()

def project_clicks(sv: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    return sv * ctr

# NOTE: no persist=True → UI changes recompute
@st.cache_data(show_spinner=True)
def run_forecast(
    data: pd.DataFrame,
    ctr_table: pd.DataFrame,
    scenarios: List[str],
    rta_rate_default: float,
    close_rate: float,
    rev_bounds: Dict[str, float],
    rollout_phases: List[Dict],
    classifier_milestones: Dict[int, float],
    runway_months: int,
    group_col: str,
    rta_rates_map: Dict[str, float],
    ramp_mode: str,
    ramp_months: int,
    dampening: float,
    counts_by_group_override: pd.Series | None = None,
) -> Dict[str, pd.DataFrame]:
    df = data.copy()
    groups = sorted(df[group_col].unique().tolist())
    ctr_indexed = ctr_table.set_index("Position")

    # Rollout & migration share over time
    rollout = expand_rollout_phases(rollout_phases, groups)
    counts = counts_by_group_override if counts_by_group_override is not None else df[group_col].value_counts()
    counts = counts.reindex(groups).fillna(0)  # align
    rollout = compute_overall_migrated_pct(rollout, counts)
    live_map = {m: set(rollout[(rollout["Month"] == m) & (rollout["IsLive"] == 1)]["Group"]) for m in range(1, 13)}
    overall_pct = {m: float(rollout[rollout["Month"] == m]["OverallMigratedPct"].max()) for m in range(1, 13)}

    # Baseline clicks per row
    base_ctr_rows = []
    for gname, g in df.groupby(group_col):
        curve = ctr_indexed[gname]
        ctr_vals = ctr_from_curve(g["Current Rank"].to_numpy(), curve)
        base_ctr_rows.append(pd.DataFrame({"idx": g.index, "baseline_ctr": ctr_vals}))
    base_ctr_join = pd.concat(base_ctr_rows).set_index("idx").loc[df.index]
    baseline_clicks = project_clicks(df["SV"].to_numpy(), base_ctr_join["baseline_ctr"].to_numpy())

    per_row_rta_rate = df[group_col].map(rta_rates_map).fillna(rta_rate_default).to_numpy()

    # Baseline monthly (constant)
    base_rtas = baseline_clicks * per_row_rta_rate
    base_jobs = base_rtas * close_rate
    base_clicks_total = float(baseline_clicks.sum())
    base_rtas_total = int(math.floor(base_rtas.sum()))
    base_jobs_total = int(math.floor(base_jobs.sum()))
    base_rev_min = base_jobs_total * rev_bounds["min"]
    base_rev_avg = base_jobs_total * rev_bounds["avg"]
    base_rev_max = base_jobs_total * rev_bounds["max"]
    baseline_monthly_df = pd.DataFrame({
        "Month": list(range(1, 13)),
        "Clicks_baseline": [base_clicks_total]*12,
        "RTA_baseline": [base_rtas_total]*12,
        "Jobs_baseline": [base_jobs_total]*12,
        "RevMin_baseline": [base_rev_min]*12,
        "RevAvg_baseline": [base_rev_avg]*12,
        "RevMax_baseline": [base_rev_max]*12,
    }).set_index("Month")

    monthly_rows = []
    for scen in scenarios:
        shifted_ranks = apply_rank_shift(df["Current Rank"].to_numpy(), scen)
        scen_ctr_rows = []
        for gname, g in df.groupby(group_col):
            curve = ctr_indexed[gname]
            ctr_vals = ctr_from_curve(shifted_ranks[g.index], curve)
            scen_ctr_rows.append(pd.DataFrame({"idx": g.index, "scen_ctr": ctr_vals}))
        scen_ctr = pd.concat(scen_ctr_rows).set_index("idx").loc[df.index]
        potential_clicks = project_clicks(df["SV"].to_numpy(), scen_ctr["scen_ctr"].to_numpy())
        incr_potential_clicks = np.maximum(0.0, potential_clicks - baseline_clicks)

        for m in range(1, 13):
            ramp = realization_factor(m, runway_months, ramp_mode, ramp_months)
            sitewide_mult = classifier_multiplier_for_pct(overall_pct[m], classifier_milestones)
            is_live = df[group_col].isin(live_map[m]).to_numpy().astype(float)

            realized_clicks = baseline_clicks + incr_potential_clicks * is_live * sitewide_mult * ramp * dampening
            rtas = realized_clicks * per_row_rta_rate
            jobs = rtas * close_rate

            total_clicks = realized_clicks.sum()
            total_rtas   = math.floor(rtas.sum())
            total_jobs   = math.floor(jobs.sum())

            rev_min = total_jobs * rev_bounds["min"]
            rev_avg = total_jobs * rev_bounds["avg"]
            rev_max = total_jobs * rev_bounds["max"]

            monthly_rows.append({
                "Scenario": scen,
                "Month": m,
                # All-in (post)
                "Clicks_allin": float(total_clicks),
                "RTA_allin": int(total_rtas),
                "Jobs_allin": int(total_jobs),
                "RevenueMin_allin": float(rev_min),
                "RevenueAvg_allin": float(rev_avg),
                "RevenueMax_allin": float(rev_max),
                # Baseline
                "Clicks_baseline": baseline_monthly_df.loc[m, "Clicks_baseline"],
                "RTA_baseline": int(baseline_monthly_df.loc[m, "RTA_baseline"]),
                "Jobs_baseline": int(baseline_monthly_df.loc[m, "Jobs_baseline"]),
                "RevMin_baseline": float(baseline_monthly_df.loc[m, "RevMin_baseline"]),
                "RevAvg_baseline": float(baseline_monthly_df.loc[m, "RevAvg_baseline"]),
                "RevMax_baseline": float(baseline_monthly_df.loc[m, "RevMax_baseline"]),
                # Incremental (clipped for counts)
                "Clicks_incr": float(total_clicks - baseline_monthly_df.loc[m, "Clicks_baseline"]),
                "RTA_incr": int(max(0, total_rtas - baseline_monthly_df.loc[m, "RTA_baseline"])),
                "Jobs_incr": int(max(0, total_jobs - baseline_monthly_df.loc[m, "Jobs_baseline"])),
                "RevMin_incr": float(max(0.0, rev_min - baseline_monthly_df.loc[m, "RevMin_baseline"])),
                "RevAvg_incr": float(max(0.0, rev_avg - baseline_monthly_df.loc[m, "RevAvg_baseline"])),
                "RevMax_incr": float(max(0.0, rev_max - baseline_monthly_df.loc[m, "RevMax_baseline"])),
                # Context
                "Overall Migrated %": overall_pct[m],
                "Ramp": float(ramp),
                "Sitewide Mult": float(sitewide_mult),
            })

    monthly = pd.DataFrame(monthly_rows)

    # Per-group (All-in)
    tpl_rows = []
    for scen in scenarios:
        shifted_ranks = apply_rank_shift(df["Current Rank"].to_numpy(), scen)
        for gname, g in df.groupby(group_col):
            curve = ctr_indexed[gname]
            scen_ctr = ctr_from_curve(shifted_ranks[g.index], curve)
            base_ctr = ctr_from_curve(df.loc[g.index, "Current Rank"].to_numpy(), curve)
            potential_clicks = project_clicks(df.loc[g.index, "SV"].to_numpy(), scen_ctr)
            baseline_clicks_tpl = project_clicks(df.loc[g.index, "SV"].to_numpy(), base_ctr)
            incr_potential = np.maximum(0.0, potential_clicks - baseline_clicks_tpl)
            for m in range(1, 13):
                ramp = realization_factor(m, runway_months, ramp_mode, ramp_months)
                sitewide_mult = classifier_multiplier_for_pct(overall_pct[m], classifier_milestones)
                live = 1 if gname in live_map[m] else 0
                realized_clicks = baseline_clicks_tpl.sum() + incr_potential.sum() * live * sitewide_mult * ramp * dampening
                rta_rate_group = float(rta_rates_map.get(gname, rta_rate_default))
                rtas = realized_clicks * rta_rate_group
                jobs = rtas * close_rate
                tpl_rows.append({
                    "Scenario": scen,
                    "Month": m,
                    group_col: gname,
                    "Clicks": float(realized_clicks),
                    "RTA Submits": int(math.floor(rtas)),
                    "Job Closes": int(math.floor(jobs)),
                })
    per_group_monthly = pd.DataFrame(tpl_rows)

    return {"monthly": monthly, "per_group_monthly": per_group_monthly}

# -----------------------------
# Sidebar Controls
# -----------------------------
cols_upload = st.columns([1,1])
with cols_upload[0]:
    upload = st.file_uploader(
        "Keyword CSV (URL, SV, Current Rank, Page Template)",
        type=["csv"],
        help="Your keyword list with required columns.",
    )
with cols_upload[1]:
    master_upload = st.file_uploader(
        "Master URL list (optional)",
        type=["csv"],
        help="One URL per row; drives migration % off true page inventory.",
    )

with st.sidebar:
    st.header("Assumptions & Controls")
    st.markdown("**Conversions & Revenue**")
    rta_default = st.number_input(
        label="Global RTA submit rate per visit (fallback)",
        min_value=0.0, max_value=1.0, value=0.008, step=0.001, format="%f",
        help="Used when no per-section override is provided. Example: 0.008 = 0.8%."
    )
    close_rate = st.number_input(
        label="Close rate from RTA submit",
        min_value=0.0, max_value=1.0, value=0.19, step=0.01, format="%f",
        help="Fraction of RTA submits that become closed jobs."
    )
    avg_rev = st.number_input(
        label="Avg revenue per job ($)",
        min_value=0.0, max_value=10_000_000.0, value=23_000.0, step=500.0,
        help="Mean revenue per closed job."
    )
    default_bounds = default_revenue_bounds(avg_rev)
    rev_min = st.number_input(
        label="Min revenue per job ($)",
        min_value=0.0, max_value=avg_rev, value=float(default_bounds["min"]), step=500.0
    )
    rev_max = st.number_input(
        label="Max revenue per job ($)",
        min_value=avg_rev, max_value=50_000_000.0, value=float(default_bounds["max"]), step=500.0
    )

    st.markdown("---")
    st.markdown("**Runway & Realization**")
    runway_choice = st.radio(
        "Runway before benefits start",
        options=["3 months", "6 months"], index=0, horizontal=True,
        help="Model waits this long before any incremental gains can materialize."
    )
    runway_months = 3 if runway_choice.startswith("3") else 6
    ramp_mode = st.radio(
        "Post-runway ramp",
        options=["Step", "Linear", "S-curve"], index=1, horizontal=True,
        help="How quickly incremental gains realize after the runway ends."
    )
    ramp_months = st.number_input(
        label="Ramp duration (months)",
        min_value=1, max_value=12, value=3, step=1,
        help="Months to reach ~100% of incremental gains after the runway."
    )

    st.markdown("---")
    st.markdown("**Aggressiveness**")
    dampening = st.slider(
        "Scenario dampening (0.25 conservative → 1.0 aggressive)",
        min_value=0.25, max_value=1.0, value=0.6, step=0.05,
        help="Global multiplier on incremental gains after rollout, classifier, and ramp."
    )

    st.markdown("---")
    st.markdown("**Scenarios**")
    scen_opts = ["Baseline", "Conservative", "Expected", "Aggressive"]
    scenarios = st.multiselect(
        "Select scenarios",
        options=scen_opts, default=["Baseline", "Expected"],
        help="Baseline = current ranks; others use heuristics by position band."
    )

    st.markdown("---")
    st.markdown("**Sitewide Classifier Milestones**")
    ms = default_classifier_multipliers()
    c30 = st.slider("30% migrated multiplier", 0.0, 1.0, float(ms[30]), 0.05)
    c50 = st.slider("50% migrated multiplier", 0.0, 1.0, float(ms[50]), 0.05)
    c70 = st.slider("70% migrated multiplier", 0.0, 1.0, float(ms[70]), 0.05)
    c90 = st.slider("90% migrated multiplier", 0.0, 1.0, float(ms[90]), 0.05)
    milestones = {30: c30, 50: c50, 70: c70, 90: c90}

# -----------------------------
# Main Body
# -----------------------------
if upload is None:
    st.info("Upload your **Keyword CSV** to begin. A sample CTR curve appears below.")
    st.dataframe(load_sample_ctr(), width="stretch")
    st.stop()

try:
    df = load_csv(upload.getvalue())
except Exception as e:
    st.error(f"Error reading Keyword CSV: {e}")
    st.stop()

# Optional: master URL list for migration %
counts_override = None
if master_upload is not None:
    try:
        master_df = load_master_urls(master_upload.getvalue())
        master_df["Site Section"] = master_df["URL"].astype(str).apply(extract_site_section)
        counts_override = master_df["Site Section"].value_counts()
        st.success("Master URL list loaded – migration % will use this inventory.")
    except Exception as e:
        st.warning(f"Could not parse Master URL list; using KW list counts. Error: {e}")

# Derive site section for KW list
df["Site Section"] = df["URL"].astype(str).apply(extract_site_section)

st.subheader("Input Snapshot")
st.dataframe(df.head(20), width="stretch")

# Grouping selection
st.subheader("Forecast Grouping")
group_choice = st.radio(
    "Group by",
    options=["Site Section", "Page Template"], index=0, horizontal=True,
    help="The chosen dimension is used for CTR curves, rollouts, and RTA overrides."
)
group_col = "Site Section" if group_choice == "Site Section" else "Page Template"

# CTR curve
st.subheader("Global CTR Curve (Editable)")
st.caption("Curve should sum to ~0.40 across positions 1–20 (≈60% zero-click).")
curve_edit = st.data_editor(load_sample_ctr(), num_rows="dynamic", key="global_ctr_editor", width="stretch")
ctr_table = build_group_ctr(sorted(df[group_col].unique().tolist()), curve_edit)

# RTA overrides
st.subheader(f"RTA Submit Rate by {group_choice}")
st.caption("Enter as decimals (e.g., 0.008 = 0.8%).")
user_defaults = {
    "shop": 0.0030, "locations": 0.0280, "homepage": 0.0210,
    "ideas": 0.0037, "inspiration": 0.0011, "professionals": 0.0044, "performance": 0.0067
}
groups = sorted(df[group_col].unique().tolist())
rta_rows = [{group_choice: g, "RTA Rate": user_defaults.get(str(g).lower(), 0.008)} for g in groups]
rta_df = st.data_editor(pd.DataFrame(rta_rows), num_rows="dynamic", key="rta_editor", width="stretch")

def rta_rate_map_from_editor(df_rate: pd.DataFrame) -> Dict[str, float]:
    m = {}
    for _, r in df_rate.iterrows():
        try:
            m[str(r[group_choice])] = float(r["RTA Rate"]) if not pd.isna(r["RTA Rate"]) else np.nan
        except Exception:
            continue
    return m

rta_rates_map = rta_rate_map_from_editor(rta_df)

# Rollouts
st.subheader(f"Rollouts by {group_choice} (Ordered Phases)")
st.caption("Define phases & durations. Gains start only after runway, then follow the selected ramp. Migration % uses your master URL list when provided.")
phase_container = st.container()
max_phases = 6
phases: List[Dict] = []
for i in range(1, max_phases + 1):
    with phase_container.expander(f"Phase {i}", expanded=True if i == 1 else False):
        cols = st.columns([2, 1])
        selected = cols[0].multiselect(
            f"{group_choice}s in Phase {i}",
            options=groups, default=[] if i > 1 else groups[:2],
            key=f"phase_grp_{i}"
        )
        months = cols[1].number_input(
            "Duration (months)", min_value=0, max_value=12, value=3 if i <= 2 else 0, step=1, key=f"phase_months_{i}"
        )
        if months > 0 and selected:
            phases.append({"name": f"Phase {i}", "groups": selected, "months": int(months)})

if not phases:
    st.warning("Define at least one rollout phase with duration > 0.")
    st.stop()

# Run forecast
with st.spinner("Running forecast..."):
    results = run_forecast(
        data=df,
        ctr_table=ctr_table,
        scenarios=scenarios,
        rta_rate_default=rta_default,
        close_rate=close_rate,
        rev_bounds={"min": rev_min, "avg": avg_rev, "max": rev_max},
        rollout_phases=phases,
        classifier_milestones=milestones,
        runway_months=runway_months,
        group_col=group_col,
        rta_rates_map=rta_rates_map,
        ramp_mode=ramp_mode,
        ramp_months=int(ramp_months),
        dampening=float(dampening),
        counts_by_group_override=counts_override if group_col == "Site Section" else None,
    )

monthly = results["monthly"].copy()
per_group_monthly = results["per_group_monthly"].copy()

# ensure numeric for charts
for col in ["RevenueAvg_allin", "RevAvg_incr"]:
    if col in monthly.columns:
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce").fillna(0.0)

# Executive summaries
st.subheader("Executive Summary – All-in (12 months)")
summary_allin = monthly.groupby("Scenario").agg({
    "Clicks_allin": "sum", "RTA_allin": "sum", "Jobs_allin": "sum",
    "RevenueMin_allin": "sum", "RevenueAvg_allin": "sum", "RevenueMax_allin": "sum",
}).reset_index()
summary_allin["RTA_allin"] = summary_allin["RTA_allin"].astype(int)
summary_allin["Jobs_allin"] = summary_allin["Jobs_allin"].astype(int)
st.dataframe(summary_allin, width="stretch")

st.subheader("Executive Summary – Incremental vs Baseline (12 months)")
summary_incr = monthly.groupby("Scenario").agg({
    "Clicks_incr": "sum", "RTA_incr": "sum", "Jobs_incr": "sum",
    "RevMin_incr": "sum", "RevAvg_incr": "sum", "RevMax_incr": "sum",
}).reset_index()
summary_incr["RTA_incr"] = summary_incr["RTA_incr"].astype(int)
summary_incr["Jobs_incr"] = summary_incr["Jobs_incr"].astype(int)
st.dataframe(summary_incr, width="stretch")

# Month 12 snapshot
st.subheader("Month 12 Snapshot (All-in & Incremental)")
m12 = monthly[monthly["Month"] == 12]
cols = st.columns(len(scenarios))
for i, scen in enumerate(scenarios):
    m = m12[m12["Scenario"] == scen]
    if m.empty: continue
    with cols[i]:
        st.metric(f"{scen} – Clicks (M12, All-in)", f"{m['Clicks_allin'].iloc[0]:,.0f}")
        st.metric("RTA (M12, All-in)", f"{m['RTA_allin'].iloc[0]:,d}")
        st.metric("Jobs (M12, All-in)", f"{m['Jobs_allin'].iloc[0]:,d}")
        st.metric("Revenue Avg (M12, All-in)", f"${m['RevenueAvg_allin'].iloc[0]:,.0f}")
        st.metric("Revenue Avg (M12, Incremental)", f"${m['RevAvg_incr'].iloc[0]:,.0f}")

# Trends
st.subheader("Monthly Trends – Revenue Avg by Scenario (All-in)")
rev_trend_allin = monthly.pivot_table(index="Month", columns="Scenario", values="RevenueAvg_allin", aggfunc="sum").reset_index()
st.line_chart(rev_trend_allin.set_index("Month"))

st.subheader("Monthly Trends – Revenue Avg by Scenario (Incremental)")
rev_trend_incr = monthly.pivot_table(index="Month", columns="Scenario", values="RevAvg_incr", aggfunc="sum").reset_index()
st.line_chart(rev_trend_incr.set_index("Month"))

# Per-Group & Details
st.subheader(f"Per-{group_choice} Monthly Rollup (All-in)")
st.dataframe(per_group_monthly, width="stretch")

st.subheader("Detailed Monthly Table (All-in & Incremental)")
st.dataframe(monthly, width="stretch")

# Downloads
st.subheader("Downloads")
@st.cache_data
def to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download Monthly (CSV, all columns)",
        data=to_csv_bytes(monthly),
        file_name="monthly_forecast_all_columns.csv",
        mime="text/csv",
        help="Baseline, all-in, incremental for each month × scenario."
    )
with c2:
    st.download_button(
        f"Download Per-{group_choice} Monthly (CSV)",
        data=to_csv_bytes(per_group_monthly),
        file_name="per_group_monthly_allin.csv",
        mime="text/csv",
        help="Per-group all-in results by month and scenario."
    )
