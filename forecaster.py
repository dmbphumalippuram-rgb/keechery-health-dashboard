"""
forecaster.py  -  Keechery Health Block surveillance dashboard (Phase 3)

  1. Open-Meteo climate  : monthly rainfall / max temperature / relative humidity (free, keyless)
  2. Spatial structure   : Voronoi polygons around each unit + touching-neighbour list (GeoPandas/Shapely)
  3. Feature engineering : month sin/cos, lag-1/lag-2, 3-month rolling mean, neighbour spill-over,
                           lagged climate, unit & disease indicators
  4. Models              : RandomForestRegressor  -> next-month case count (+ 10-90% tree range)
                           RandomForestClassifier -> probability that next month exceeds the outbreak threshold
  5. Honest validation   : expanding-window time-series back-test (NO random split) with MAE compared
                           against two naive baselines (last month, same month last year)

Everything is a plain function / small class so app.py (Phase 4) only has to call run_pipeline().
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

DISEASES = ["Leptospirosis", "Dengue", "H1N1 Influenza", "Hepatitis A", "Acute Diarrheal Disease"]
BLOCK_LAT, BLOCK_LON = 9.8432, 76.4321

# Outbreak threshold = max(floor, mean + K_SD * SD) of the same unit/disease in the same season window.
# ADJUST THE FLOORS to your IDSP / district definition (they are the minimum count that can ever be "an outbreak").
DEFAULT_FLOORS = {"Leptospirosis": 2, "Dengue": 3, "H1N1 Influenza": 2, "Hepatitis A": 2,
                  "Acute Diarrheal Disease": 5}
K_SD = 2.0
RISK_HIGH, RISK_MODERATE = 0.30, 0.10          # probability cut-offs for the traffic-light label

BASE_FEATURES = ["m_sin", "m_cos", "lag1", "lag2", "roll3", "nb_lag1", "pop10k"]
CLIMATE_FEATURES = ["rain_lag1", "rain_lag2", "tmax_lag1", "tmax_lag2", "rh_lag1", "rh_lag2"]


# =========================================================================== #
# 1. Climate (Open-Meteo, no API key)
# =========================================================================== #
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Daily rain/tmax + hourly humidity JSON -> one row per day."""
    d = pd.DataFrame({
        "date": pd.to_datetime(payload["daily"]["time"]),
        "rain_mm": payload["daily"]["precipitation_sum"],
        "tmax_c": payload["daily"]["temperature_2m_max"],
    })
    h = pd.DataFrame({"t": pd.to_datetime(payload["hourly"]["time"]),
                      "rh": payload["hourly"]["relative_humidity_2m"]})
    rh = h.groupby(h["t"].dt.normalize())["rh"].mean().rename("rh_pct")
    return d.merge(rh, left_on="date", right_index=True, how="left")


def fetch_climate_monthly(start, end, lat=BLOCK_LAT, lon=BLOCK_LON, timeout=40, session=None) -> pd.DataFrame:
    """
    Monthly climate for [start, end]. Only COMPLETE months are returned (the archive lags ~5 days).
    Columns: month_start, rain_mm (sum), tmax_c (mean of daily max), rh_pct (mean).
    Raises on any network/API problem - use get_climate() for the safe wrapper.
    """
    import requests
    get = (session or requests).get
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    last_ok = pd.Timestamp.today().normalize() - pd.Timedelta(days=6)
    end = min(end, last_ok)
    frames = []
    for year in range(start.year, end.year + 1):                       # one request per year
        s, e = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
        r = get(ARCHIVE_URL, timeout=timeout, params={
            "latitude": lat, "longitude": lon,
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,precipitation_sum",
            "hourly": "relative_humidity_2m",
            "timezone": "Asia/Kolkata",
        })
        r.raise_for_status()
        frames.append(_parse_open_meteo(r.json()))
    d = pd.concat(frames, ignore_index=True)
    d["month_start"] = d["date"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby("month_start")
    m = pd.DataFrame({"rain_mm": g["rain_mm"].sum(min_count=1), "tmax_c": g["tmax_c"].mean(),
                      "rh_pct": g["rh_pct"].mean(), "days": g["date"].count()}).reset_index()
    m["dim"] = m["month_start"].dt.days_in_month
    m = m[m["days"] >= m["dim"]]                                       # complete months only
    return m[["month_start", "rain_mm", "tmax_c", "rh_pct"]].reset_index(drop=True)


def get_climate(start, end, lat=BLOCK_LAT, lon=BLOCK_LON) -> tuple[pd.DataFrame | None, str]:
    """Safe wrapper: never raises. Returns (df or None, human-readable status)."""
    try:
        df = fetch_climate_monthly(start, end, lat, lon)
        if df.empty:
            return None, "Open-Meteo returned no complete months."
        return df, f"Open-Meteo OK ({len(df)} months, {df['month_start'].min():%b %Y}-{df['month_start'].max():%b %Y})."
    except Exception as exc:                                           # network down, rate limit, etc.
        return None, f"Climate unavailable ({type(exc).__name__}); model runs WITHOUT climate features."


# =========================================================================== #
# 2. Spatial: Voronoi polygons + neighbours
# =========================================================================== #
def build_voronoi(master: pd.DataFrame, buffer_km: float = 3.0):
    """
    GeoDataFrame of one Voronoi polygon per unit (EPSG:4326), clipped to the units' convex hull
    buffered by `buffer_km`. These are *approximate catchments*, not official panchayat boundaries.
    Needs >= 3 units with lat/lon.
    """
    import geopandas as gpd
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram

    if len(master) < 3:
        raise ValueError("Need at least 3 units with lat/lon to build Voronoi polygons.")
    g = gpd.GeoDataFrame(master.copy(), geometry=gpd.points_from_xy(master["lon"], master["lat"]),
                         crs="EPSG:4326").to_crs("EPSG:32643")          # UTM 43N (Kerala) -> metres
    pts = MultiPoint(list(g.geometry))
    hull = pts.convex_hull.buffer(buffer_km * 1000)
    cells = list(voronoi_diagram(pts, envelope=hull).geoms)
    polys = []
    for pt in g.geometry:
        cell = next(c for c in cells if c.contains(pt) or c.distance(pt) < 1e-6)
        polys.append(cell.intersection(hull))
    g["geometry"] = polys
    return g.to_crs("EPSG:4326")


def voronoi_neighbours(vor) -> dict[str, list[str]]:
    """Units whose polygons touch (1 m tolerance, computed in a metric CRS)."""
    m = vor.to_crs("EPSG:32643")
    names, geoms = m["panchayat"].tolist(), list(m.geometry)
    nb = {n: [] for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if geoms[i].buffer(1.0).intersects(geoms[j]):
                nb[names[i]].append(names[j])
                nb[names[j]].append(names[i])
    return nb


# =========================================================================== #
# 3. Feature engineering
# =========================================================================== #
def _prep_panel(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["month_start"] = pd.to_datetime(p["month_start"])
    return p[["month_start", "panchayat", "disease", "total_cases"]]


def build_design(panel: pd.DataFrame, master: pd.DataFrame, neighbours: dict,
                 climate: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    One row per (target month T, unit, disease), INCLUDING one future row (last month + 1) with y = NaN.
    All features use information from months < T only.
    """
    p = _prep_panel(panel)
    units = master["panchayat"].tolist()
    months = pd.date_range(p["month_start"].min(), p["month_start"].max() + pd.offsets.MonthBegin(1), freq="MS")
    grid = pd.MultiIndex.from_product([months, units, DISEASES], names=["month_start", "panchayat", "disease"])
    df = (p.set_index(["month_start", "panchayat", "disease"])["total_cases"]
            .reindex(grid).rename("y").reset_index())
    df = df.sort_values(["panchayat", "disease", "month_start"]).reset_index(drop=True)
    key = df.groupby(["panchayat", "disease"], sort=False)["y"]
    df["lag1"], df["lag2"], df["lag12"] = key.shift(1), key.shift(2), key.shift(12)
    df["roll3"] = key.transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())

    # neighbour spill-over: mean of touching units' cases in month T-1, same disease
    W = np.zeros((len(units), len(units)))
    for i, u in enumerate(units):
        for n in neighbours.get(u, []):
            if n in units:
                W[i, units.index(n)] = 1.0
    W = W / np.where(W.sum(1, keepdims=True) == 0, 1, W.sum(1, keepdims=True))
    parts = []
    for d in DISEASES:
        C = df[df["disease"] == d].pivot(index="month_start", columns="panchayat", values="y")[units]
        nb = pd.DataFrame(C.values @ W.T, index=C.index, columns=units).shift(1)
        parts.append(nb.stack().rename("nb_lag1").reset_index().assign(disease=d))
    nbdf = pd.concat(parts).rename(columns={"level_1": "panchayat"})
    nbdf.columns = ["month_start", "panchayat", "nb_lag1", "disease"]
    df = df.merge(nbdf, on=["month_start", "panchayat", "disease"], how="left")

    df["pop10k"] = df["panchayat"].map(master.set_index("panchayat")["population"] / 10000.0)
    df["m_sin"] = np.sin(2 * np.pi * df["month_start"].dt.month / 12)
    df["m_cos"] = np.cos(2 * np.pi * df["month_start"].dt.month / 12)

    feats = list(BASE_FEATURES)
    if climate is not None and len(climate):
        c = climate.set_index("month_start").reindex(months)
        for col, short in (("rain_mm", "rain"), ("tmax_c", "tmax"), ("rh_pct", "rh")):
            s = c[col]
            s = s.fillna(s.groupby(s.index.month).transform("mean")).fillna(s.mean())   # light imputation
            c[f"{short}_lag1"], c[f"{short}_lag2"] = s.shift(1), s.shift(2)
        c = c[[f for f in CLIMATE_FEATURES]].reset_index().rename(columns={"index": "month_start"})
        df = df.merge(c, on="month_start", how="left")
        for f in CLIMATE_FEATURES:
            df[f] = df[f].fillna(df[f].mean())
        feats += CLIMATE_FEATURES

    df["disease"] = pd.Categorical(df["disease"], DISEASES)
    df["panchayat"] = pd.Categorical(df["panchayat"], units)
    dummies = pd.get_dummies(df[["disease", "panchayat"]], prefix=["dis", "unit"], dtype=float)
    df = pd.concat([df, dummies], axis=1)
    feats += list(dummies.columns)
    return df, feats


# =========================================================================== #
# 4. Outbreak thresholds + models
# =========================================================================== #
def compute_thresholds(train: pd.DataFrame, floors: dict | None = None, k_sd: float = K_SD) -> pd.DataFrame:
    """
    Per unit x disease x calendar month: max(floor, mean + k*SD) over the SAME season window (month-1..month+1)
    of the TRAINING period only. Falls back to all months if the window has < 6 points.
    """
    floors = floors or DEFAULT_FLOORS
    rows = []
    for (u, d), g in train.groupby(["panchayat", "disease"], observed=True):
        cm = g["month_start"].dt.month.values
        y = g["y"].values.astype(float)
        for m in range(1, 13):
            win = {(m - 2) % 12 + 1, m, m % 12 + 1}
            v = y[np.isin(cm, list(win))]
            if len(v) < 6:
                v = y
            thr = max(float(floors.get(d, 2)), float(v.mean() + k_sd * v.std(ddof=0)))
            rows.append((u, d, m, thr))
    return pd.DataFrame(rows, columns=["panchayat", "disease", "cal_month", "threshold"])


def attach_thresholds(df: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["cal_month"] = d["month_start"].dt.month
    d["panchayat"] = d["panchayat"].astype(str)
    d["disease"] = d["disease"].astype(str)
    d = d.merge(thr, on=["panchayat", "disease", "cal_month"], how="left")
    d["outbreak"] = (d["y"] > d["threshold"]).astype(int)
    d.loc[d["y"].isna(), "outbreak"] = np.nan
    return d


def _fit(train: pd.DataFrame, feats: list[str], n_estimators: int, seed: int):
    reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, max_features=0.6,
                                random_state=seed, n_jobs=-1).fit(train[feats], train["y"])
    pos = int(train["outbreak"].sum())
    clf = None
    if pos >= 5 and pos < len(train):                                  # need some events to learn from
        clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3, max_features=0.6,
                                     random_state=seed, n_jobs=-1).fit(train[feats], train["outbreak"].astype(int))
    return reg, clf, pos


def _prob(clf, reg, X: pd.DataFrame, thr: pd.Series) -> np.ndarray:
    """Outbreak probability: classifier if we have one, else share of trees predicting > threshold."""
    if clf is not None:
        return clf.predict_proba(X)[:, 1]
    per_tree = np.column_stack([t.predict(X.values) for t in reg.estimators_])
    return (per_tree > thr.values[:, None]).mean(axis=1)


# =========================================================================== #
# 5. Back-test (time-series hold-out) and final forecast
# =========================================================================== #
def backtest(df: pd.DataFrame, feats: list[str], n_test_months: int = 6, floors=None,
             n_estimators: int = 200, seed: int = 42) -> dict:
    """Expanding window: for each of the last n months, train on earlier months only, predict that month."""
    known = df[df["y"].notna()]
    test_months = sorted(known["month_start"].unique())[-n_test_months:]
    out = []
    for T in test_months:
        tr = known[(known["month_start"] < T)].dropna(subset=["lag2", "roll3"])
        te = known[known["month_start"] == T]
        thr = compute_thresholds(tr, floors)
        tr, te = attach_thresholds(tr, thr), attach_thresholds(te, thr)
        reg, clf, _ = _fit(tr, feats, n_estimators, seed)
        te = te.assign(pred=reg.predict(te[feats]),
                       prob=_prob(clf, reg, te[feats], te["threshold"]),
                       base_rate=tr["outbreak"].mean())
        out.append(te)
    bt = pd.concat(out, ignore_index=True)
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    res = {"n_test_months": len(test_months), "n_rows": len(bt),
           "test_period": f"{pd.Timestamp(test_months[0]):%b %Y} - {pd.Timestamp(test_months[-1]):%b %Y}",
           "mae_model": mae(bt["pred"], bt["y"]), "mae_last_month": mae(bt["lag1"], bt["y"])}
    seas = bt.dropna(subset=["lag12"])
    res["mae_same_month_last_year"] = mae(seas["lag12"], seas["y"]) if len(seas) else np.nan
    res["mae_model_vs_last_month_pct"] = 100 * (1 - res["mae_model"] / res["mae_last_month"]) if res["mae_last_month"] else np.nan
    res["mae_by_disease"] = (bt.groupby("disease", observed=True)
                               .apply(lambda g: pd.Series({"model": mae(g["pred"], g["y"]),
                                                           "last_month": mae(g["lag1"], g["y"]),
                                                           "mean_cases": g["y"].mean()}), include_groups=False)
                               .round(3))
    res["outbreak_events"] = int(bt["outbreak"].sum())
    res["brier_model"] = float(brier_score_loss(bt["outbreak"], bt["prob"]))
    res["brier_base_rate"] = float(brier_score_loss(bt["outbreak"], bt["base_rate"]))
    res["auc"] = float(roc_auc_score(bt["outbreak"], bt["prob"])) if 0 < bt["outbreak"].sum() < len(bt) else np.nan
    return res


def forecast_next_month(df: pd.DataFrame, feats: list[str], floors=None, n_estimators: int = 400, seed: int = 42):
    """Fit on all known months, predict the single future month (the NaN-target rows)."""
    known = df[df["y"].notna()].dropna(subset=["lag2", "roll3"])
    fut = df[df["y"].isna()].copy()
    thr = compute_thresholds(df[df["y"].notna()], floors)
    known, fut = attach_thresholds(known, thr), attach_thresholds(fut, thr)
    reg, clf, pos = _fit(known, feats, n_estimators, seed)
    X = fut[feats]
    per_tree = np.column_stack([t.predict(X.values) for t in reg.estimators_])
    fut["predicted_cases"] = per_tree.mean(1)
    fut["pred_low"], fut["pred_high"] = np.percentile(per_tree, [10, 90], axis=1)
    fut["outbreak_prob_pct"] = 100 * _prob(clf, reg, X, fut["threshold"])
    fut["risk_level"] = np.where(fut["outbreak_prob_pct"] >= 100 * RISK_HIGH, "High",
                         np.where(fut["outbreak_prob_pct"] >= 100 * RISK_MODERATE, "Moderate", "Low"))
    imp = (pd.Series(reg.feature_importances_, index=feats).sort_values(ascending=False)
             .rename("importance").reset_index().rename(columns={"index": "feature"}))
    info = {"train_rows": len(known), "train_outbreak_events": pos,
            "probability_method": "RandomForestClassifier" if clf is not None else "share of trees above threshold (too few past outbreaks to train a classifier)"}
    return fut, imp, thr, info


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict
    importance: pd.DataFrame
    thresholds: pd.DataFrame
    voronoi: object
    neighbours: dict
    notes: list = field(default_factory=list)


def run_pipeline(panel: pd.DataFrame, master: pd.DataFrame, climate: pd.DataFrame | None = None,
                 floors: dict | None = None, n_test_months: int = 6) -> ForecastResult:
    """The one call app.py will make."""
    notes = []
    vor = build_voronoi(master)
    nb = voronoi_neighbours(vor)
    df, feats = build_design(panel, master, nb, climate)
    if climate is None:
        notes.append("No climate data: forecast uses case history, season and neighbour spill-over only.")
    metrics = backtest(df, feats, n_test_months, floors)
    fut, imp, thr, info = forecast_next_month(df, feats, floors)
    notes.append(f"Outbreak probability via {info['probability_method']}; {info['train_outbreak_events']} past outbreak-months in training.")
    pm = fut["month_start"].iloc[0]
    last = df[df["month_start"] == df["month_start"].max() - pd.offsets.MonthBegin(1)].set_index(["panchayat", "disease"])["y"]
    fut["last_month_cases"] = [last.get((u, d), np.nan) for u, d in zip(fut["panchayat"], fut["disease"])]
    fut["forecast_month"] = pm.strftime("%Y-%m")
    fut["predicted_per_10k"] = fut["predicted_cases"] / fut["pop10k"]
    cols = ["forecast_month", "panchayat", "disease", "last_month_cases", "predicted_cases", "pred_low",
            "pred_high", "predicted_per_10k", "threshold", "outbreak_prob_pct", "risk_level"]
    out = fut[cols].copy()
    for c in ["predicted_cases", "pred_low", "pred_high", "predicted_per_10k", "threshold", "outbreak_prob_pct"]:
        out[c] = out[c].round(2)
    risk_rank = {"High": 0, "Moderate": 1, "Low": 2}
    out = out.sort_values(["risk_level", "outbreak_prob_pct"], key=lambda s: s.map(risk_rank) if s.name == "risk_level" else -s)
    return ForecastResult(out.reset_index(drop=True), metrics, imp, thr, vor, nb, notes)


# =========================================================================== #
# CLI:  python forecaster.py --panel monthly_panel.csv --master master_units.csv [--no-climate]
# =========================================================================== #
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--climate-csv", help="optional saved climate csv (month_start,rain_mm,tmax_c,rh_pct)")
    ap.add_argument("--no-climate", action="store_true")
    ap.add_argument("--out", default="forecast_out.csv")
    a = ap.parse_args()
    panel, master = pd.read_csv(a.panel), pd.read_csv(a.master)
    climate = None
    if a.climate_csv:
        climate = pd.read_csv(a.climate_csv, parse_dates=["month_start"])
    elif not a.no_climate:
        s = pd.to_datetime(panel["month_start"]).min()
        e = pd.to_datetime(panel["month_start"]).max() + pd.offsets.MonthEnd(0)
        climate, msg = get_climate(s, e)
        print(msg)
        if climate is not None:
            climate.to_csv("climate_monthly.csv", index=False)
    r = run_pipeline(panel, master, climate)
    pd.set_option("display.width", 200)
    print("\nNEIGHBOURS:", r.neighbours)
    print("\nBACK-TEST", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.metrics.items() if k != "mae_by_disease"})
    print(r.metrics["mae_by_disease"])
    print("\nTOP FEATURES\n", r.importance.head(10).to_string(index=False))
    print("\nFORECAST (top 15)\n", r.forecast.head(15).to_string(index=False))
    for n in r.notes:
        print("NOTE:", n)
    r.forecast.to_csv(a.out, index=False)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    _cli()
