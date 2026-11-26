"""
app.py - Keechery Health Block: Epidemiological Surveillance, 3-Year Review & Early-Warning Dashboard (Phase 4)

Tabs:  1) 3-Year MO Review (Time-Place-Person)   2) Predictive Early Warning   3) IHIP Data Upload & Cleaner
Uses : data_cleaner.clean_ihip_exports(), forecaster.get_climate() / run_pipeline()
Secrets (all optional - the app runs on bundled sample data without any):
       KEECHERY_SHEET_CSV (monthly_panel tab URL), KEECHERY_MASTER_CSV, KEECHERY_LINELIST_CSV,
       CARTO_API_KEY, GROQ_API_KEY
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from data_cleaner import AGE_COLS, DISEASES, clean_ihip_exports
from forecaster import BLOCK_LAT, BLOCK_LON, DEFAULT_FLOORS, build_voronoi, get_climate, run_pipeline

st.set_page_config(page_title="Keechery Block - Surveillance Dashboard", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
SAMPLE_FILES = {"master": "keechery_sample_master_units.csv",
                "panel": "keechery_sample_monthly_panel.csv",
                "ll": "keechery_sample_ward_line_list.csv"}
PANEL_REQ = ["year", "month", "month_start", "panchayat", "disease", "p_cases", "l_cases", "total_cases",
             "deaths", *AGE_COLS, "male", "female"]
LL_REQ = ["case_id", "onset_date", "panchayat", "ward_no", "disease", "form_type", "age", "sex", "outcome"]
MASTER_REQ = ["panchayat", "population", "lat", "lon"]
RISK_ICON = {"High": "🔴 High", "Moderate": "🟡 Moderate", "Low": "🟢 Low"}
RISK_COLOR = {"High": "#d73027", "Moderate": "#f5a623", "Low": "#1a9850"}
AGE_LABELS = dict(zip(AGE_COLS, ["<5", "5-14", "15-44", "45-59", "60+"]))
CARTO_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
              '&copy; <a href="https://carto.com/attributions">CARTO</a>')


# =========================================================================== #
# Secrets (crash-proof) & data loading
# =========================================================================== #
def secret(name: str, default: str = "") -> str:
    """st.secrets.get() that can never raise (also safe when no secrets.toml exists, e.g. local runs)."""
    try:
        v = st.secrets.get(name, default)
        return str(v).strip() if v is not None else default
    except Exception:
        return default


@st.cache_data(ttl=600, show_spinner="Loading data from Google Sheet…")
def read_remote_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)


def read_sample(key: str) -> pd.DataFrame:
    for p in (BASE / SAMPLE_FILES[key], BASE / "data" / SAMPLE_FILES[key]):
        if p.exists():
            return pd.read_csv(p)
    st.error(f"Sample file {SAMPLE_FILES[key]} not found next to app.py. Add it to the repository.")
    st.stop()


def prep_panel(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in PANEL_REQ if c not in df.columns]
    if miss:
        raise ValueError(f"monthly_panel is missing columns: {miss}")
    p = df.copy()
    p["month_start"] = pd.to_datetime(p["month_start"])
    for c in ["p_cases", "l_cases", "total_cases", "deaths", *AGE_COLS, "male", "female"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0).astype(int)
    p["year"], p["month"] = p["month_start"].dt.year, p["month_start"].dt.month
    if "remarks" not in p.columns:
        p["remarks"] = ""
    p["remarks"] = p["remarks"].fillna("")
    for c in ("panchayat", "disease"):
        p[c] = p[c].astype(str).str.strip()
    return p.sort_values(["month_start", "panchayat", "disease"]).reset_index(drop=True)


def prep_ll(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in LL_REQ if c not in df.columns]
    if miss:
        raise ValueError(f"ward_line_list is missing columns: {miss}")
    d = df.copy()
    d["onset_date"] = pd.to_datetime(d["onset_date"], errors="coerce")
    d["year"], d["month"] = d["onset_date"].dt.year, d["onset_date"].dt.month
    d["ward_no"] = pd.to_numeric(d["ward_no"], errors="coerce")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    return d.dropna(subset=["onset_date"])


def prep_master(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in MASTER_REQ if c not in df.columns]
    if miss:
        raise ValueError(f"master_units is missing columns: {miss}")
    m = df.copy()
    m["panchayat"] = m["panchayat"].astype(str).str.strip()
    for c in ("population", "lat", "lon"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    return m.dropna(subset=["population", "lat", "lon"]).reset_index(drop=True)


def load_table(key: str, secret_name: str, prep):
    """Remote (published Google Sheet CSV) if configured and valid, else bundled sample. Returns (df, source, msg)."""
    url = secret(secret_name)
    if url:
        try:
            return prep(read_remote_csv(url)), "Google Sheet", ""
        except Exception as exc:
            return prep(read_sample(key)), "Local sample", f"{secret_name} failed ({type(exc).__name__}: {exc}); using sample."
    return prep(read_sample(key)), "Local sample", ""


def load_all():
    master, s_m, e_m = load_table("master", "KEECHERY_MASTER_CSV", prep_master)
    panel, s_p, e_p = load_table("panel", "KEECHERY_SHEET_CSV", prep_panel)
    ll, s_l, e_l = load_table("ll", "KEECHERY_LINELIST_CSV", prep_ll)
    sources = {"master_units": s_m, "monthly_panel": s_p, "ward_line_list": s_l}
    errs = [e for e in (e_m, e_p, e_l) if e]
    return master, panel, ll, sources, errs


def panel_for_merge(p: pd.DataFrame) -> pd.DataFrame:
    out = p.copy()
    out["month_start"] = out["month_start"].dt.strftime("%Y-%m-%d")
    return out


def ll_for_merge(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["onset_date"] = out["onset_date"].dt.strftime("%Y-%m-%d")
    return out


# =========================================================================== #
# Cached heavy work
# =========================================================================== #
@st.cache_data(ttl=3600, show_spinner="Fetching climate from Open-Meteo…")
def _cached_climate(start: str, end: str):
    df, msg = get_climate(start, end)
    if df is None:
        raise RuntimeError(msg)            # exceptions are NOT cached, so a network blip doesn't stick for an hour
    return df, msg


def safe_climate(start: str, end: str):
    try:
        return _cached_climate(start, end)
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Training Random Forest models and back-testing (first run takes a minute)…")
def cached_pipeline(panel: pd.DataFrame, master: pd.DataFrame, climate, floors: dict, n_test_months: int):
    return run_pipeline(panel, master, climate, floors=floors, n_test_months=n_test_months)


@st.cache_resource(show_spinner=False)
def cached_voronoi(master: pd.DataFrame):
    return build_voronoi(master)


# =========================================================================== #
# Map helpers
# =========================================================================== #
def base_map(center, dark: bool) -> folium.Map:
    key = secret("CARTO_API_KEY")
    if key:
        style = "dark_all" if dark else "light_all"
        url = f"https://basemaps.cartocdn.com/{style}/{{z}}/{{x}}/{{y}}.png?key={key}"
        return folium.Map(location=center, zoom_start=12, tiles=url, attr=CARTO_ATTR, control_scale=True)
    return folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)


def fit_and_label(m: folium.Map, gdf, master: pd.DataFrame):
    minx, miny, maxx, maxy = gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    for _, r in master.iterrows():
        folium.Marker([r["lat"], r["lon"]], icon=folium.DivIcon(
            html=f'<div style="font:600 11px sans-serif;color:#111;text-shadow:0 0 3px #fff,0 0 3px #fff;'
                 f'white-space:nowrap;transform:translate(-50%,-50%)">{r["panchayat"]}</div>')).add_to(m)


def incidence_map(vor, master, inc: pd.Series, cases: pd.Series, title: str, dark: bool) -> folium.Map:
    g = vor[["panchayat", "geometry"]].copy()
    g["inc"] = g["panchayat"].map(inc).fillna(0.0).round(2)
    g["cases"] = g["panchayat"].map(cases).fillna(0).astype(int)
    vmax = max(float(g["inc"].max()), 0.01)
    cmap = LinearColormap(["#ffffb2", "#fd8d3c", "#bd0026"], vmin=0, vmax=vmax, caption=title)
    m = base_map([master["lat"].mean(), master["lon"].mean()], dark)
    folium.GeoJson(
        g, style_function=lambda f: {"fillColor": cmap(f["properties"]["inc"]), "color": "#555",
                                     "weight": 1.2, "fillOpacity": 0.65},
        tooltip=folium.GeoJsonTooltip(fields=["panchayat", "cases", "inc"],
                                      aliases=["Unit", "Cases", "Per 10,000"])).add_to(m)
    cmap.add_to(m)
    fit_and_label(m, g, master)
    return m


def forecast_map(vor, master, fc: pd.DataFrame, colour_by: str, dark: bool) -> folium.Map:
    g = vor[["panchayat", "geometry"]].merge(fc, on="panchayat", how="left")
    g["pred_txt"] = g.apply(lambda r: f'{r["predicted_cases"]:.1f} ({r["pred_low"]:.1f}-{r["pred_high"]:.1f})', axis=1)
    g["prob_txt"] = g["outbreak_prob_pct"].map(lambda v: f"{v:.0f}%")
    g["last_txt"] = g["last_month_cases"].map(lambda v: f"{v:.0f}")
    keep = ["panchayat", "geometry", "pred_txt", "prob_txt", "risk_level", "last_txt", "predicted_per_10k"]
    g = g[keep]
    m = base_map([master["lat"].mean(), master["lon"].mean()], dark)
    if colour_by == "Outbreak risk (traffic light)":
        style = lambda f: {"fillColor": RISK_COLOR.get(f["properties"]["risk_level"], "#999"),
                           "color": "#444", "weight": 1.2, "fillOpacity": 0.55}
    else:
        vmax = max(float(g["predicted_per_10k"].max()), 0.01)
        cmap = LinearColormap(["#ffffb2", "#fd8d3c", "#bd0026"], vmin=0, vmax=vmax,
                              caption="Predicted cases per 10,000 (next month)")
        style = lambda f: {"fillColor": cmap(float(f["properties"]["predicted_per_10k"] or 0)),
                           "color": "#444", "weight": 1.2, "fillOpacity": 0.65}
        cmap.add_to(m)
    folium.GeoJson(g, style_function=style, tooltip=folium.GeoJsonTooltip(
        fields=["panchayat", "last_txt", "pred_txt", "prob_txt", "risk_level"],
        aliases=["Unit", "Last month", "Predicted (10-90%)", "Outbreak probability", "Risk"])).add_to(m)
    fit_and_label(m, g, master)
    return m


# =========================================================================== #
# AI / rule-based field directive
# =========================================================================== #
SYSTEM_PROMPT = (
    "You are an epidemiology decision-support assistant for a Kerala health block (IDSP / NHM). "
    "Use ONLY the figures provided; never invent numbers, outbreaks or locations. Be concise and practical. "
    "Follow IDSP/NCDC and Kerala DHS guidelines in spirit, do not state drug doses, and remind that the Medical "
    "Officer must verify with field investigation before declaring an outbreak.")

ACTIONS = {
    "Leptospirosis": {
        "MO": "Alert all facilities to flag fever with myalgia/jaundice/red eyes and recent water or soil exposure; "
              "review stock and protocol for doxycycline prophylaxis and treatment; confirm referral path for severe cases.",
        "HI": "Counsel and offer chemoprophylaxis (per MO order) to high-risk exposed groups (drain/canal cleaners, "
              "paddy/farm and employment-guarantee workers); rodent control; advise gloves and boots; "
              "follow up every case with a household/exposure survey."},
    "Dengue": {
        "MO": "Activate fever surveillance and daily line-listing; ensure NS1/IgM testing and platelet monitoring; "
              "plan fogging only around confirmed case clusters as per guideline.",
        "HI": "Targeted source reduction in the unit: containers, tyres, arecanut spathes, coconut shells, "
              "roof gutters; larval survey around each case; weekly dry-day drive and house-to-house awareness."},
    "H1N1 Influenza": {
        "MO": "Apply triage categories A/B/C at OPD; keep oseltamivir available for category B/C per guideline; "
              "isolate and notify suspect cases promptly.",
        "HI": "Cough-etiquette and mask advice in crowded settings, line-list contacts, check high-risk groups "
              "(pregnant women, elderly, comorbid), inform schools/workplaces of the cluster."},
    "Hepatitis A": {
        "MO": "Review recent jaundice/ALT-positive cases for a common source; advise isolation and rest; "
              "consider contact advice as per state policy.",
        "HI": "Super-chlorinate public and household wells in the affected wards; inspect water supply, food outlets, "
              "ice and juice vendors; promote handwashing and boiled water; recheck residual chlorine."},
    "Acute Diarrheal Disease": {
        "MO": "Keep ORS-zinc stock and ORS corners active; monitor for clustered cases (same ward / same water source); "
              "send stool/water samples when a cluster is suspected.",
        "HI": "Super-chlorination of wells and tanks, water-quality testing (H2S/FTK), food-safety checks, "
              "health education on boiled water; verify sanitation near the water source."},
}


def rule_based_directive(rows: pd.DataFrame, month: str) -> str:
    if rows.empty:
        return "No unit-disease pair is at Moderate or High risk for the forecast month. Continue routine surveillance."
    out = [f"**Forecast month: {month}** (rule-based, decision support only)"]
    for d, g in rows.groupby("disease", sort=False):
        a = ACTIONS.get(d)
        units = ", ".join(f'{r.panchayat} ({r.risk_level}, {r.outbreak_prob_pct:.0f}%, ~{r.predicted_cases:.0f} cases)'
                          for r in g.itertuples())
        out.append(f"\n**{d}** - priority units: {units}")
        if a:
            out.append(f"- *Medical Officer:* {a['MO']}\n- *Health Inspectors:* {a['HI']}")
    out.append("\n*All forecasts are statistical estimates. Verify with field investigation before declaring an outbreak.*")
    return "\n".join(out)


@st.cache_data(ttl=3600, show_spinner=False)
def groq_call(prompt: str, _key: str) -> str:
    import requests
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {_key}", "Content-Type": "application/json"},
                      json={"model": "llama-3.3-70b-versatile", "temperature": 0.2, "max_tokens": 800,
                            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                         {"role": "user", "content": prompt}]}, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def build_prompt(rows: pd.DataFrame, month: str, climate_note: str) -> str:
    t = rows[["panchayat", "disease", "last_month_cases", "predicted_cases", "pred_low", "pred_high",
              "threshold", "outbreak_prob_pct", "risk_level"]].round(1).to_csv(index=False)
    return (f"Forecast month: {month}, Keechery Health Block, Ernakulam, Kerala.\n{climate_note}\n"
            f"Priority unit-disease forecasts (CSV; threshold = outbreak threshold in cases):\n{t}\n"
            "Write: (1) a 2-line situation summary; (2) 'For the Medical Officer' - up to 4 bullets; "
            "(3) 'For Health Inspectors' - up to 5 bullets naming the units; (4) one line on uncertainty. "
            "Maximum 250 words.")


# =========================================================================== #
# Load data (+ session overrides from the cleaner tab)
# =========================================================================== #
master, panel, ll, sources, load_errors = load_all()
if st.session_state.get("ov_panel") is not None:
    panel = prep_panel(st.session_state["ov_panel"])
    sources["monthly_panel"] = "IHIP upload (this session)"
if st.session_state.get("ov_ll") is not None:
    ll = prep_ll(st.session_state["ov_ll"])
    sources["ward_line_list"] = "IHIP upload (this session)"

years = sorted(panel["year"].unique())
lm = panel["month_start"].max()                     # latest month in data
partial_year = int(lm.year) if lm.month < 12 else None
ymap = {y: (f"{y} YTD (Jan-{lm:%b})" if y == partial_year else str(y)) for y in years}
pop = master.set_index("panchayat")["population"]

# =========================================================================== #
# Sidebar
# =========================================================================== #
with st.sidebar:
    st.title("🩺 Keechery Block")
    st.caption("Surveillance • 3-year review • early warning")
    for k, v in sources.items():
        st.write(f"{'🟢' if v != 'Local sample' else '🟠'} **{k}**: {v}")
    st.caption(f"Data through **{lm:%B %Y}**")
    if st.button("🔄 Reload sheet data"):
        read_remote_csv.clear()
        st.rerun()
    st.divider()
    dark = st.radio("Map basemap", ["Positron (light)", "Dark Matter"], horizontal=False) == "Dark Matter"
    if not secret("CARTO_API_KEY"):
        st.caption("CARTO key not set → OpenStreetMap basemap.")
    st.divider()
    st.subheader("Filters (Tab 1)")
    sel_disease = st.selectbox("Disease", DISEASES)
    sel_units = st.multiselect("Units", master["panchayat"].tolist(), default=master["panchayat"].tolist())
    if not sel_units:
        sel_units = master["panchayat"].tolist()
    st.divider()
    with st.expander("Outbreak threshold floors (Tab 2)"):
        st.caption("Minimum monthly cases that can ever count as an outbreak. Match your IDSP/district definition.")
        floors = {d: int(st.number_input(d, min_value=1, max_value=50, value=int(DEFAULT_FLOORS[d]), step=1))
                  for d in DISEASES}
    st.caption(f"Groq AI: {'configured' if secret('GROQ_API_KEY') else 'not set (rule-based directive)'}")

st.title("Keechery Health Block - Communicable Disease Surveillance")
if "Local sample" in sources.values():
    st.warning("**Showing SAMPLE (synthetic) data** for at least one table. Set the Google-Sheet secrets "
               "(Phase 5) to see real data. Do not use these numbers for decisions.")
for e in load_errors:
    st.error(e)
st.caption("Decision-support tool. Figures come from IHIP P/L-form reports; forecasts are statistical estimates "
           "and must be verified by field investigation.")

tab1, tab2, tab3 = st.tabs(["📊 3-Year MO Review (TPP)", "⚠️ Predictive Early Warning", "📥 IHIP Upload & Cleaner"])

# =========================================================================== #
# TAB 1 - 3-Year review
# =========================================================================== #
with tab1:
    period_opts = {ymap[y]: (lambda d, y=y: d["year"] == y) for y in years}
    period_opts["Last 12 months"] = lambda d: d["month_start"] > lm - pd.DateOffset(months=12)
    period_opts["All years"] = lambda d: pd.Series(True, index=d.index)
    period_label = st.selectbox("Period for place / person analysis", list(period_opts), index=len(years) - 1)
    pmask = period_opts[period_label]

    # ---------------- TIME ----------------
    st.header("⏱ Time")
    rows = []
    for d in DISEASES:
        r = {"Disease": d}
        for y in years:
            r[ymap[y]] = int(panel[(panel.disease == d) & (panel.year == y)].total_cases.sum())
        if partial_year and (partial_year - 1) in years:
            prev = int(panel[(panel.disease == d) & (panel.year == partial_year - 1) & (panel.month <= lm.month)].total_cases.sum())
            cur = r[ymap[partial_year]]
            r[f"{partial_year - 1} Jan-{lm:%b}"] = prev
            r["Change vs same period"] = round(100 * (cur - prev) / prev, 1) if prev else np.nan
        rows.append(r)
    yoy = pd.DataFrame(rows)
    st.subheader("Year-on-year comparison (total cases)")
    st.dataframe(yoy, hide_index=True, column_config={
        "Change vs same period": st.column_config.NumberColumn(format="%+.0f%%")} if "Change vs same period" in yoy else None)
    long = yoy.melt(id_vars="Disease", value_vars=[ymap[y] for y in years], var_name="Year", value_name="Cases")
    fig = px.bar(long, x="Year", y="Cases", color="Year", facet_col="Disease", facet_col_wrap=3,
                 category_orders={"Year": [ymap[y] for y in years], "Disease": DISEASES}, height=520)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(title=None, showticklabels=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig)
    st.caption("The latest year is year-to-date; compare it with the 'same period' column, not with full years.")

    sub = panel[(panel.disease == sel_disease) & (panel.panchayat.isin(sel_units))]
    st.subheader(f"Monthly seasonal curve - {sel_disease}")
    seas = sub.groupby(["year", "month"], as_index=False)["total_cases"].sum()
    seas["Year"] = seas["year"].map(ymap)
    figs = px.line(seas, x="month", y="total_cases", color="Year", markers=True,
                   labels={"month": "Month", "total_cases": "Cases"})
    figs.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                      ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    st.plotly_chart(figs)

    st.subheader(f"P-form (probable) vs L-form (lab-confirmed) and mortality - {sel_disease}")
    pl = sub.groupby("year", as_index=False)[["p_cases", "l_cases", "total_cases", "deaths"]].sum()
    pl["Year"] = pl["year"].map(ymap)
    pl["% lab-confirmed"] = (100 * pl["l_cases"] / pl["total_cases"].replace(0, np.nan)).round(1)
    pl["Case fatality %"] = (100 * pl["deaths"] / pl["total_cases"].replace(0, np.nan)).round(2)
    c1, c2 = st.columns([3, 2])
    with c1:
        figpl = px.bar(pl.melt(id_vars="Year", value_vars=["p_cases", "l_cases"], var_name="Form", value_name="Cases")
                       .replace({"p_cases": "P-form (probable)", "l_cases": "L-form (lab-confirmed)"}),
                       x="Year", y="Cases", color="Form", barmode="stack",
                       category_orders={"Year": [ymap[y] for y in years]})
        st.plotly_chart(figpl)
    with c2:
        st.dataframe(pl[["Year", "total_cases", "p_cases", "l_cases", "deaths", "% lab-confirmed", "Case fatality %"]]
                     .rename(columns={"total_cases": "Total", "p_cases": "P", "l_cases": "L", "deaths": "Deaths"}),
                     hide_index=True)

    # ---------------- PLACE ----------------
    st.header("📍 Place")
    pp = panel[pmask(panel)]
    cases_ud = (pp.groupby(["panchayat", "disease"])["total_cases"].sum().unstack()
                .reindex(index=master["panchayat"], columns=DISEASES).fillna(0))
    inc = cases_ud.div(pop.reindex(cases_ud.index), axis=0) * 10000
    st.subheader(f"Incidence per 10,000 population - {period_label}")
    figh = px.imshow(inc.round(1).values, x=DISEASES, y=inc.index.tolist(), text_auto=".1f", aspect="auto",
                     color_continuous_scale="YlOrRd", labels={"color": "per 10,000"})
    st.plotly_chart(figh)
    st.caption("Uses the population in master_units (a single figure per unit). For partial periods the rate is not annualised.")
    st.subheader(f"Map - {sel_disease}, incidence per 10,000 ({period_label})")
    try:
        vor = cached_voronoi(master)
        m1 = incidence_map(vor, master, inc[sel_disease], cases_ud[sel_disease],
                           f"{sel_disease} per 10,000 ({period_label})", dark)
        st_folium(m1, height=480, use_container_width=True, returned_objects=[], key="map_inc")
        st.caption("Polygons are Voronoi catchments around each unit's coordinates - approximate, not official boundaries.")
    except Exception as exc:
        st.error(f"Map could not be drawn: {exc}")

    # ---------------- PERSON ----------------
    st.header("👥 Person")
    ppu = pp[pp.panchayat.isin(sel_units)]
    age = ppu.groupby("disease")[AGE_COLS].sum().reindex(DISEASES).fillna(0)
    age_pct = (age.div(age.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1)
    a1, a2 = st.columns(2)
    with a1:
        st.subheader(f"Age distribution (% of cases) - {period_label}")
        al = age_pct.reset_index().melt(id_vars="disease", var_name="Age", value_name="%")
        al["Age"] = al["Age"].map(AGE_LABELS)
        st.plotly_chart(px.bar(al, x="Age", y="%", color="disease", barmode="group",
                               category_orders={"Age": list(AGE_LABELS.values())}))
    with a2:
        st.subheader("Sex distribution")
        sx = ppu.groupby("disease")[["male", "female"]].sum().reindex(DISEASES).fillna(0)
        sxp = (sx.div(sx.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1).reset_index() \
            .melt(id_vars="disease", var_name="Sex", value_name="%")
        st.plotly_chart(px.bar(sxp, x="disease", y="%", color="Sex", barmode="stack"))
    st.caption("Age and sex come from the monthly_panel columns. Rows with missing age/sex in IHIP are not counted here.")

    st.subheader(f"Repeat hotspot wards - {sel_disease}")
    topn = st.slider("A ward is a 'hotspot' in a year if it is in the block's top N wards by cases", 3, 10, 5)
    lw = ll[(ll.disease == sel_disease) & (ll.panchayat.isin(sel_units))].dropna(subset=["ward_no"])
    if lw.empty:
        st.info("No ward-level cases for this selection.")
    else:
        wc = lw.groupby(["year", "panchayat", "ward_no"]).size().rename("cases").reset_index()
        wc["rank"] = wc.groupby("year")["cases"].rank(ascending=False, method="min")
        top = wc[(wc["rank"] <= topn) & (wc["cases"] >= 2)]
        rep = (top.groupby(["panchayat", "ward_no"])
               .agg(years_in_top=("year", "nunique"), years=("year", lambda s: ", ".join(str(int(v)) for v in sorted(s))))
               .reset_index())
        tot = wc.groupby(["panchayat", "ward_no"])["cases"].sum().rename("total_cases").reset_index()
        rep = rep.merge(tot, on=["panchayat", "ward_no"])
        rep = rep[rep["years_in_top"] >= 2].sort_values(["years_in_top", "total_cases"], ascending=False)
        rep["ward_no"] = rep["ward_no"].astype(int)
        if rep.empty:
            st.success("No ward appeared in the top list in two or more years.")
        else:
            st.dataframe(rep.rename(columns={"panchayat": "Unit", "ward_no": "Ward", "years_in_top": "Years in top list",
                                             "years": "Which years", "total_cases": "Total cases"}), hide_index=True)
        st.caption("Ward numbers are only meaningful within their unit. The current year is partial.")

# =========================================================================== #
# TAB 2 - Predictive early warning
# =========================================================================== #
with tab2:
    st.header("⚠️ Next-month early warning")
    use_clim = st.checkbox("Use Open-Meteo climate (rainfall, max temperature, humidity)", value=True)
    n_test = st.slider("Back-test months", 3, 12, 6)
    if st.button("▶ Generate / refresh forecast", type="primary", key="gen_fc"):
        st.session_state["fc_on"] = True
    if not st.session_state.get("fc_on"):
        st.info("Click **Generate / refresh forecast** to train the models. The first run takes about a minute; "
                "later changes are cached.")
    else:
        climate, cmsg = (None, "Climate switched off.")
        if use_clim:
            climate, cmsg = safe_climate(f"{panel.month_start.min():%Y-%m-%d}", f"{(lm + pd.offsets.MonthEnd(0)):%Y-%m-%d}")
        (st.success if climate is not None else st.warning)(cmsg)
        try:
            res = cached_pipeline(panel, master, climate, floors, n_test)
        except Exception as exc:
            st.error(f"Forecast failed: {type(exc).__name__}: {exc}")
            st.stop()
        fc = res.forecast.copy()
        fmonth = fc["forecast_month"].iloc[0]
        st.subheader(f"Forecast for {pd.Period(fmonth).strftime("%B %Y")}")

        tot = fc.groupby("disease")[["predicted_cases", "last_month_cases"]].sum().reindex(DISEASES)
        cols = st.columns(len(DISEASES))
        for c, d in zip(cols, DISEASES):
            c.metric(d, f"{tot.loc[d, 'predicted_cases']:.0f} cases",
                     f"{tot.loc[d, 'predicted_cases'] - tot.loc[d, 'last_month_cases']:+.0f} vs last month",
                     delta_color="inverse")
        st.caption("Block totals of the unit-level forecasts.")

        # map
        mc1, mc2 = st.columns([1, 1])
        map_d = mc1.selectbox("Map disease", DISEASES, index=DISEASES.index(sel_disease), key="map_d")
        colour_by = mc2.radio("Colour by", ["Outbreak risk (traffic light)", "Predicted incidence per 10,000"],
                              horizontal=True)
        fm = forecast_map(res.voronoi, master, fc[fc.disease == map_d], colour_by, dark)
        st_folium(fm, height=480, use_container_width=True, returned_objects=[], key="map_fc")
        st.caption("Voronoi catchments are approximate. Traffic light: 🔴 ≥30% • 🟡 10-30% • 🟢 <10% outbreak probability.")

        # table
        st.subheader("Unit-level forecast")
        show = fc.copy()
        show["Risk"] = show["risk_level"].map(RISK_ICON)
        show["Forecast (10-90% range)"] = show.apply(
            lambda r: f"{r.predicted_cases:.1f}  ({r.pred_low:.1f}-{r.pred_high:.1f})", axis=1)
        dsel = st.multiselect("Show diseases", DISEASES, default=DISEASES, key="tbl_d")
        show = show[show.disease.isin(dsel)]
        st.dataframe(
            show[["panchayat", "disease", "last_month_cases", "Forecast (10-90% range)", "threshold",
                  "outbreak_prob_pct", "Risk"]].rename(columns={
                      "panchayat": "Unit", "disease": "Disease", "last_month_cases": "Last month",
                      "threshold": "Outbreak threshold", "outbreak_prob_pct": "Outbreak probability"}),
            hide_index=True,
            column_config={"Outbreak probability": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100)})
        st.download_button("⬇ Download forecast CSV", fc.to_csv(index=False).encode("utf-8"),
                           f"forecast_{fmonth}.csv", "text/csv")

        # validation
        st.subheader("How reliable is this? (time-series back-test)")
        M = res.metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("MAE - model", f"{M['mae_model']:.2f}")
        k2.metric("MAE - 'same as last month'", f"{M['mae_last_month']:.2f}",
                  f"{M['mae_model_vs_last_month_pct']:+.0f}% model advantage", delta_color="off")
        k3.metric("MAE - same month last year", f"{M['mae_same_month_last_year']:.2f}")
        k4.metric("Outbreak events in test", f"{M['outbreak_events']}")
        k5, k6, _ = st.columns(3)
        k5.metric("Brier: model vs base rate", f"{M['brier_model']:.3f}", f"base {M['brier_base_rate']:.3f}", delta_color="off")
        k6.metric("AUC", "n/a" if pd.isna(M["auc"]) else f"{M['auc']:.2f}")
        st.caption(f"Test period: {M['test_period']} (expanding window; no random split). MAE = mean absolute error in cases per unit-disease-month.")
        if M["mae_model"] >= M["mae_last_month"]:
            st.warning("The model is NOT better than simply repeating last month's count in this back-test - treat case-count forecasts with caution.")
        if M["brier_model"] >= M["brier_base_rate"] or pd.isna(M["auc"]) or M["auc"] < 0.65:
            st.warning("Outbreak probabilities are not yet clearly better than the base rate (few past outbreak-months). "
                       "Use them as a screening aid alongside field reports, not as a trigger.")
        with st.expander("Error by disease"):
            st.dataframe(M["mae_by_disease"].rename(columns={"model": "MAE model", "last_month": "MAE last-month",
                                                             "mean_cases": "Mean cases"}))
        with st.expander("What drives the forecast? (feature importance)"):
            imp = res.importance.head(12).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h"))
        for n in res.notes:
            st.caption("ℹ️ " + n)

        # AI directive
        st.subheader("🧭 Field Action Directive")
        prio = fc[fc.risk_level.isin(["High", "Moderate"])].head(10)
        if prio.empty:
            prio = fc.head(5)
            st.info("No Moderate/High risks; showing the 5 highest-probability pairs.")
        climate_note = ""
        if climate is not None and len(climate):
            last = climate.iloc[-1]
            same = climate[climate.month_start.dt.month == last.month_start.month]["rain_mm"].mean()
            climate_note = (f"Latest complete month ({last.month_start:%b %Y}): rainfall {last.rain_mm:.0f} mm "
                            f"(same-month average {same:.0f} mm), mean max temperature {last.tmax_c:.1f} C.")
        if st.button("Generate directive", key="gen_dir"):
            key = secret("GROQ_API_KEY")
            text, src = None, "Rule-based fallback"
            if key:
                try:
                    with st.spinner("Asking Groq (llama-3.3-70b-versatile)…"):
                        text = groq_call(build_prompt(prio, fmonth, climate_note), key)
                    src = "Groq llama-3.3-70b-versatile"
                except Exception as exc:
                    st.warning(f"Groq unavailable ({type(exc).__name__}); showing rule-based directive instead.")
            if text is None:
                text = rule_based_directive(prio, fmonth)
            st.session_state["directive"] = (text, src, fmonth)
        d = st.session_state.get("directive")
        if d and d[2] == fmonth:
            st.markdown(d[0])
            st.caption(f"Source: {d[1]}. Only aggregated forecast figures are sent to the AI - no patient data. "
                       "Not a substitute for the Medical Officer's judgement.")

# =========================================================================== #
# TAB 3 - IHIP upload & cleaner
# =========================================================================== #
with tab3:
    st.header("📥 IHIP Data Upload & Cleaner")
    st.write("Drop raw **P-form** and **L-form** exports (Excel or CSV). Patient names/IDs are used only to detect "
             "duplicates and are never kept.")
    files = st.file_uploader("IHIP exports", type=["xlsx", "xls", "csv"], accept_multiple_files=True)
    hints = {}
    for f in files or []:
        hints[f.name] = st.radio(f"Form type for **{f.name}**", ["Auto (from file)", "P-form", "L-form"],
                                 horizontal=True, key=f"h_{f.name}")
    merge = st.checkbox("Merge into the data currently loaded (replaces only the months present in the upload)", value=True)
    alias_txt = st.text_area("Extra location spellings (one per line: `raw spelling = Panchayat`)",
                             placeholder="Keechery PHC = Amballoor", height=70)
    aliases = {}
    for line in alias_txt.splitlines():
        if "=" in line:
            a, b = (s.strip() for s in line.split("=", 1))
            if a and b:
                aliases[a] = b
    pad = st.checkbox("Pad months with zero cases up to the current month", value=False)

    if st.button("Run cleaner", type="primary", disabled=not files):
        try:
            batch = []
            for f in files:
                f.seek(0)
                h = hints[f.name]
                batch.append((f, None if h.startswith("Auto") else h[0]))
            with st.spinner("Cleaning…"):
                r = clean_ihip_exports(
                    batch, master,
                    existing_panel=panel_for_merge(panel) if merge else None,
                    existing_line_list=ll_for_merge(ll) if merge else None,
                    end=pd.Timestamp.today().strftime("%Y-%m-01") if pad else None,
                    unit_aliases=aliases or None)
            st.session_state["cleaned"] = r
        except Exception as exc:
            st.session_state["cleaned"] = None
            st.error(f"Cleaning failed: {exc}")

    r = st.session_state.get("cleaned")
    if r is not None:
        rep = r.report
        c1, c2, c3 = st.columns(3)
        c1.metric("Cases in upload (after de-duplication)", rep["cases_in_line_list"])
        c2.metric("Duplicates removed (P+L / repeats)", rep["duplicates_removed"])
        c3.metric("Rows rejected", len(r.rejected))
        st.caption(f"Months covered by upload: {rep['months_covered']}")
        for w in r.warnings:
            st.warning(w)
        for fi in rep["files"]:
            st.caption(f"📄 {fi['file']}: form {','.join(fi['form']) or '-'} • read {fi['rows_read']} • kept {fi['rows_kept']} • rejected {fi['rows_rejected']}")
        if len(r.rejected):
            with st.expander(f"Rejected rows ({len(r.rejected)}) - check before trusting totals", expanded=True):
                st.dataframe(r.rejected, hide_index=True)
                top_loc = r.rejected[r.rejected["reject_reason"].str.contains("location")]["raw_location"].value_counts().head(5)
                if len(top_loc):
                    st.info("Unmatched locations - add them in the spelling box above and re-run: " +
                            ", ".join(f"'{k}' ({v})" for k, v in top_loc.items()))
        st.subheader("Updated monthly_panel (preview, newest months)")
        st.dataframe(r.panel.sort_values("month_start", ascending=False).head(30), hide_index=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ monthly_panel.csv", r.panel.to_csv(index=False).encode("utf-8"), "monthly_panel.csv", "text/csv")
        d2.download_button("⬇ ward_line_list.csv", r.line_list.to_csv(index=False).encode("utf-8"), "ward_line_list.csv", "text/csv")
        d3.download_button("⬇ rejected_rows.csv", r.rejected.to_csv(index=False).encode("utf-8"), "rejected_rows.csv", "text/csv")
        b1, b2 = st.columns(2)
        if b1.button("✅ Use this data in the dashboard (this session only)"):
            st.session_state["ov_panel"], st.session_state["ov_ll"] = r.panel, r.line_list
            st.session_state["fc_on"] = False
            st.rerun()
        if st.session_state.get("ov_panel") is not None and b2.button("↩ Revert to loaded data"):
            st.session_state["ov_panel"] = st.session_state["ov_ll"] = None
            st.rerun()
        st.info("To make the update permanent, paste/import the downloaded CSVs into the `monthly_panel` and "
                "`ward_line_list` tabs of your Google Sheet (Replace current sheet). Streamlit Cloud cannot save files itself.")
