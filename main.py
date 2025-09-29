# SEO Impact Forecaster – Streamlit App
# Author: Brandon Lazovic + ChatGPT (GPT-5 Thinking)
# Notes:
# - Upload a CSV with columns: URL, SV, Current Rank, Page Template
# - Choose grouping: L1 Site Section (from URL) or Page Template
# - Configure CTR curves per chosen group (positions 1–20). Assumes ~60% zero-click; curves should sum to ~0.40
# - Scenarios: Baseline, Conservative, Expected, Aggressive (rule-based deltas by current rank band)
# - Rollouts: define ordered phases with groups and phase durations (months)
# - Sitewide classifier milestones: 30/50/70/90% migration -> multipliers to realized gains
# - Runway toggle: 3 or 6 months before benefits can start
# - Forecast horizon fixed to 12 months
# - Conversions: Global defaults RTA 0.8%, close 19%, avg revenue $23,000, **plus per-group RTA overrides**
# - Integer constraints: RTA & Job closes are floored at monthly aggregates
# - Scales to ~25k rows via vectorized pandas

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List

st.set_page_config(page_title="SEO Impact Forecaster", layout="wide")
st.title("SEO Impact Forecaster")
st.caption("Forecast traffic → RTA submits → job closes → revenue from ranking improvements, rollouts, sitewide effects, and section-level RTA rates.")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_sample_ctr() -> pd.DataFrame:
    # Default curve (positions 1..20), normalized to sum ≈ 0.40 to respect ~60% zero-click
    base = pd.DataFrame({
        "Position": list(range(1, 21)),
        # Decaying curve (placeholder). You can overwrite in the UI
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

# Rank scenario rules
SCENARIO_RULES = {
    "Baseline": {"delta": 0, "apply_aggressive_top": False},
    "Conservative": {"delta": "rules_conservative", "apply_aggressive_top": False},
    "Expected": {"delta": "rules_expected", "apply_aggressive_top": False},
    "Aggressive": {"delta": "rules_aggressive", "apply_aggressive_top": True},
}

def rank_delta_rules(rank: float, flavor: str) -> int:
    # Base: 15–20 → +3; 11–14 → +2; 5–9 → +1
    # Aggressive: also +1 for 1–4 (capped at pos 1)
    if 15 <= rank <= 20:
        base = 3
    elif 11 <= rank <= 14:
        base = 2
    elif 5 <= rank <= 9:
        base = 1
    else:
        base = 0
    if flavor == "conservative":
        return max(0, base - 1)
    if flavor == "expected":
        return base
    if flavor == "aggressive":
        if 1 <= rank <= 4:
            return 1
        return base
    return 0

# Grouping utils

def extract_site_section(url: str) -> str:
    try:
        # Fast approximate L1 segment extraction
        u = str(url)
        if '//' in u:
            u = u.split('//', 1)[1]
        path = '/' + u.split('/', 1)[1] if '/' in u else '/'
        path = path.split('?', 1)[0].split('#', 1)[0]
        segs = path.strip('/').split('/') if path else []
        if len(segs) == 0 or segs[0] == '':
            return 'homepage'
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

# Rollouts & classifier

def expand_rollout_phases(phases: List[Dict], groups: List[str]) -> pd.DataFrame:
    months = list(range(1, 13))
    rows = []
    current_month = 1
    live_month = {g: 13 for g in groups}
    for phase in phases:
        dur = max(1, int(phase.get("months", 1)))
        phase_groups = phase.get("groups", [])
        lm = min(12, current_month + dur - 1)
        for g in phase_groups:
            live_month[g] = min(live_month.get(g, 13), lm)
        current_month += dur
        if current_month > 12:
            break
    for m in months:
        for g in groups:
            rows.append({"Month": m, "Group": g, "IsLive": 1 if m >= live_month[g] else 0})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def default_classifier_multipliers() -> Dict[int, float]:
    return {30: 0.3, 50: 0.5, 70: 0.7, 90: 1.0}

def compute_overall_migrated_pct(rollout_df: pd.DataFrame, counts_by_group: pd.Series) -> pd.DataFrame:
    live_share = []
    for m, g in rollout_df.groupby("Month"):
        live_groups = g.loc[g["IsLive"] == 1, "Group"].tolist()
        share = counts_by_group.loc[counts_by_group.index.isin(live_groups)].sum() / max(1, counts_by_group.sum())
        live_share.append({"Month": m, "OverallMigratedPct": float(share)})
    return rollout_df.merge(pd.DataFrame(live_share), on="Month", how="left")

def classifier_multiplier_for_pct(pct: float, milestones: Dict[int, float]) -> float:
    pct100 = pct * 100
    applicable = [k for k in milestones.keys() if k <= pct100]
    if not applicable:
        return 0.0
    return float(milestones[max(applicable)])

# Core forecast

def apply_rank_shift(current_rank: np.ndarray, scenario: str) -> np.ndarray:
    if scenario == "Baseline":
        return current_rank.copy()
    flavor = {"Conservative": "conservative", "Expected": "expected", "Aggressive": "aggressive"}.get(scenario, "expected")
    f = np.frompyfunc(lambda r: max(1.0, r - rank_delta_rules(float(r), flavor)), 1, 1)
    return f(current_rank).astype(float)

def ctr_from_curve(cur_rank: np.ndarray, ctr_col: pd.Series) -> np.ndarray:
    max_pos = int(ctr_col.index.max())
    pos = np.clip(np.rint(cur_rank).astype(int), 1, max_pos)
    return ctr_col.reindex(pos).to_numpy()

def project_clicks(sv: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    return sv * ctr

@st.cache_data(show_spinner=True, persist=True)
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
) -> Dict[str, pd.DataFrame]:
    df = data.copy()
    groups = sorted(df[group_col].unique().tolist())
    ctr_indexed = ctr_table.set_index("Position")

    # Rollout & migration share over time
    rollout = expand_rollout_phases(rollout_phases, groups)
    counts = df[group_col].value_counts()
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

    # Per-row RTA rate vector (fallback to default if missing)
    per_row_rta_rate = df[group_col].map(rta_rates_map).fillna(rta_rate_default).to_numpy()

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
            runway_mult = 0.0 if m <= runway_months else 1.0
            sitewide_mult = classifier_multiplier_for_pct(overall_pct[m], classifier_milestones)
            live_groups = live_map[m]
            is_live = df[group_col].isin(live_groups).to_numpy().astype(float)

            realized_clicks = baseline_clicks + incr_potential_clicks * is_live * sitewide_mult * runway_mult
            rtas = realized_clicks * per_row_rta_rate
            jobs = rtas * close_rate

            total_clicks = realized_clicks.sum()
            total_rtas = math.floor(rtas.sum())
            total_jobs = math.floor(jobs.sum())

            rev_min = total_jobs * rev_bounds["min"]
            rev_avg = total_jobs * rev_bounds["avg"]
            rev_max = total_jobs * rev_bounds["max"]

            monthly_rows.append({
                "Scenario": scen,
                "Month": m,
                "Total SV": float(df["SV"].sum()),
                "Clicks": float(total_clicks),
                "RTA Submits": int(total_rtas),
                "Job Closes": int(total_jobs),
                "Revenue Min": float(rev_min),
                "Revenue Avg": float(rev_avg),
                "Revenue Max": float(rev_max),
                "Overall Migrated %": overall_pct[m],
                "Runway Active": int(m <= runway_months),
                "Sitewide Mult": sitewide_mult,
            })

    monthly = pd.DataFrame(monthly_rows)

    # Per-group monthly rollup (using same gating and per-group RTA)
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
                runway_mult = 0.0 if m <= runway_months else 1.0
                sitewide_mult = classifier_multiplier_for_pct(overall_pct[m], classifier_milestones)
                live = 1 if gname in live_map[m] else 0
                realized_clicks = baseline_clicks_tpl.sum() + incr_potential.sum() * live * sitewide_mult * runway_mult
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
upload = st.file_uploader("Upload CSV (columns: URL, SV, Current Rank, Page Template)", type=["csv"])

with st.sidebar:
    st.header("Assumptions & Controls")
    st.markdown("**Conversions & Revenue**")
    rta_default = st.number_input("Global RTA submit rate per visit (fallback)", min_value=0.0, max_value=1.0, value=0.008, step=0.001, format="%f")
    close_rate = st.number_input("Close rate from RTA submit", min_value=0.0, max_value=1.0, value=0.19, step=0.01, format="%f")
    avg_rev = st.number_input("Avg revenue per job ($)", min_value=0.0, value=23000.0, step=500.0)
    rev_min = st.number_input("Min revenue per job ($)", min_value=0.0, value=float(default_revenue_bounds(avg_rev)["min"]), step=500.0)
    rev_max = st.number_input("Max revenue per job ($)", min_value=0.0, value=float(default_revenue_bounds(avg_rev)["max"]), step=500.0)

    st.markdown("---")
    st.markdown("**Runway & Horizon**")
    runway_choice = st.radio("Runway before benefits start", options=["3 months", "6 months"], index=0, horizontal=True)
    runway_months = 3 if runway_choice.startswith("3") else 6
    st.caption("Forecast horizon fixed to 12 months.")

    st.markdown("---")
    st.markdown("**Scenarios**")
    scen_opts = ["Baseline", "Conservative", "Expected", "Aggressive"]
    scenarios = st.multiselect("Select scenarios", options=scen_opts, default=["Baseline", "Expected", "Aggressive"])

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
    st.info("Upload your CSV to begin. A sample CTR curve will be shown below; customize per-group after upload.")
    st.dataframe(load_sample_ctr(), use_container_width=True)
    st.stop()

try:
    df = load_csv(upload.getvalue())
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

# Derive site section from URL if not present
df["Site Section"] = df["URL"].astype(str).apply(extract_site_section)

st.subheader("Input Snapshot")
st.dataframe(df.head(20), use_container_width=True)

# Grouping selection
st.subheader("Forecast Grouping")
group_choice = st.radio("Group by", options=["Site Section", "Page Template"], index=0, horizontal=True)
group_col = "Site Section" if group_choice == "Site Section" else "Page Template"

# CTR per-group table
groups = sorted(df[group_col].unique().tolist())
default_ctr = load_sample_ctr()
ctr_table = build_group_ctr(groups, default_ctr)

st.subheader(f"CTR Curves per {group_choice} (Editable)")
st.caption("Ensure each column sums to ~0.40 across positions 1–20 to reflect ~60% zero-click.")
ctr_edit = st.data_editor(ctr_table, num_rows="dynamic", use_container_width=True, key="ctr_editor")

# RTA rate overrides per group
st.subheader(f"RTA Submit Rate by {group_choice}")
st.caption("Enter as decimals (e.g., 0.008 = 0.8%). Rows not listed fall back to the global default above.")
# User-specified defaults for common sections
user_defaults = {
    "shop": 0.0030,           # 0.30%
    "locations": 0.0280,      # 2.8%
    "homepage": 0.0210,       # 2.1%
    "ideas": 0.0037,          # 0.37%
    "inspiration": 0.0011,    # 0.11%
    "professionals": 0.0044,  # 0.44%
    "performance": 0.0067,    # 0.67%
}
rta_rows = []
for g in groups:
    rta_rows.append({group_choice: g, "RTA Rate": user_defaults.get(str(g).lower(), rta_default)})
rta_df = st.data_editor(pd.DataFrame(rta_rows), num_rows="dynamic", use_container_width=True, key="rta_editor")

def rta_rate_map_from_editor(df_rate: pd.DataFrame) -> Dict[str, float]:
    colname = group_choice
    m = {}
    for _, r in df_rate.iterrows():
        try:
            m[str(r[colname])] = float(r["RTA Rate"]) if not pd.isna(r["RTA Rate"]) else np.nan
        except Exception:
            continue
    return m

rta_rates_map = rta_rate_map_from_editor(rta_df)

# Rollout Phases Builder
st.subheader(f"Rollouts by {group_choice} (Ordered Phases)")
st.caption("Define phases: which groups roll out together and how many months each phase takes. Benefits for a group start after its phase completes. Sitewide classifier caps realized gains by overall % migrated.")
phase_container = st.container()
max_phases = 6
phases: List[Dict] = []
for i in range(1, max_phases + 1):
    with phase_container.expander(f"Phase {i}", expanded=True if i == 1 else False):
        cols = st.columns([2, 1])
        selected = cols[0].multiselect(
            f"{group_choice}s in Phase {i}", options=groups, default=[] if i > 1 else groups[:2], key=f"phase_grp_{i}"
        )
        months = cols[1].number_input(f"Duration (months)", min_value=0, max_value=12, value=3 if i <= 2 else 0, step=1, key=f"phase_months_{i}")
        if months > 0 and selected:
            phases.append({"name": f"Phase {i}", "groups": selected, "months": int(months)})

if not phases:
    st.warning("Define at least one rollout phase with duration > 0.")
    st.stop()

# Run Forecast
with st.spinner("Running forecast..."):
    results = run_forecast(
        data=df,
        ctr_table=ctr_edit,
        scenarios=scenarios,
        rta_rate_default=rta_default,
        close_rate=close_rate,
        rev_bounds={"min": rev_min, "avg": avg_rev, "max": rev_max},
        rollout_phases=phases,
        classifier_milestones=milestones,
        runway_months=runway_months,
        group_col=group_col,
        rta_rates_map=rta_rates_map,
    )

monthly = results["monthly"]
per_group_monthly = results["per_group_monthly"]

# Executive Summary
st.subheader("Executive Summary (12 months)")
summary = monthly.groupby("Scenario").agg({
    "Clicks": "sum",
    "RTA Submits": "sum",
    "Job Closes": "sum",
    "Revenue Min": "sum",
    "Revenue Avg": "sum",
    "Revenue Max": "sum",
}).reset_index()
summary["RTA Submits"] = summary["RTA Submits"].astype(int)
summary["Job Closes"] = summary["Job Closes"].astype(int)
st.dataframe(summary, use_container_width=True)

# Month 12 Snapshot
st.subheader("Month 12 Snapshot")
month12 = monthly[monthly["Month"] == 12].copy()
cols = st.columns(len(scenarios))
for i, scen in enumerate(scenarios):
    m = month12[month12["Scenario"] == scen]
    if m.empty:
        continue
    with cols[i]:
        st.metric(f"{scen} – Clicks (M12)", f"{m['Clicks'].iloc[0]:,.0f}")
        st.metric("RTA Submits (M12)", f"{m['RTA Submits'].iloc[0]:,d}")
        st.metric("Job Closes (M12)", f"{m['Job Closes'].iloc[0]:,d}")
        st.metric("Revenue Avg (M12)", f"${m['Revenue Avg'].iloc[0]:,.0f}")

# Trends
st.subheader("Monthly Trends – Revenue Avg by Scenario")
rev_trend = monthly.pivot_table(index="Month", columns="Scenario", values="Revenue Avg", aggfunc="sum").reset_index()
st.line_chart(rev_trend.set_index("Month"))

# Per-Group Rollup
st.subheader(f"Per-{group_choice} Monthly Rollup")
st.dataframe(per_group_monthly, use_container_width=True)

# Detailed Monthly Table
st.subheader("Detailed Monthly Table")
st.dataframe(monthly, use_container_width=True)

# Downloads
st.subheader("Downloads")
@st.cache_data
def to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Monthly Summary (CSV)", data=to_csv_bytes(monthly), file_name="monthly_forecast.csv", mime="text/csv")
with c2:
    st.download_button(f"Download Per-{group_choice} Monthly (CSV)", data=to_csv_bytes(per_group_monthly), file_name="per_group_monthly.csv", mime="text/csv")

# Notes
st.markdown("""
**Modeling notes**
- CTR curves should sum to ~0.40 across positions 1–20 (assumes ~60% zero-click). Provide per-group curves for realism.
- Scenario rank improvements use heuristics: 15–20 → +3; 11–14 → +2; 5–9 → +1; Aggressive also gives +1 for 1–4 (never above position 1.0).
- Benefits start only after the chosen runway (3 or 6 months). Before that, forecast returns baseline.
- Group gains only apply after their rollout phase completes. Sitewide classifier milestones cap realized gains based on overall % migrated.
- **Per-group RTA rates** override the global default for more accurate RTA → close → revenue modeling.
- RTA submits and Job Closes are floored to integers at the monthly aggregate level; revenue reported for min/avg/max based on those integers.
- 12-month horizon fixed by design. Performance: vectorized pandas to support up to ~25k rows.
""")
