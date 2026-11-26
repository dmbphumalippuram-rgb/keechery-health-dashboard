"""
data_cleaner.py  -  Keechery Health Block surveillance dashboard (Phase 2)

Turns raw IHIP P-form / L-form line-list exports (Excel or CSV) into:
  1. ward_line_list : de-identified case list (no names, hashed anonymous IDs)
  2. monthly_panel  : zero-filled month x panchayat x disease table (Phase 1 schema)

Design rules
  * Headers are matched by alias, so small IHIP export changes do not break it.
  * Title rows above the header are skipped automatically.
  * Patient names are used ONLY in memory to detect duplicates, never written out.
  * A case present in both P and L forms is counted once, as L.
  * total_cases = p_cases + l_cases  (p = probable, NOT lab-confirmed).
  * Rows that cannot be used are returned in `rejected` with a reason (never silently lost).
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DISEASES = [
    "Leptospirosis",
    "Dengue",
    "H1N1 Influenza",
    "Hepatitis A",
    "Acute Diarrheal Disease",
]
DISEASE_PATTERNS = [
    ("Leptospirosis", r"lepto"),
    ("Dengue", r"dengue|\bdhf\b|\bdss\b"),
    ("H1N1 Influenza", r"h1\s*n1|swine"),
    ("Hepatitis A", r"hep(atitis)?[\s\-_]*a\b|\bhav\b"),
    ("Acute Diarrheal Disease", r"diarrh|\badd\b|gastro"),
]
AGE_COLS = ["age_u5", "age_5_14", "age_15_44", "age_45_59", "age_60plus"]
AGE_BINS = [-1, 5, 15, 45, 60, 200]  # right-open: [0,5) [5,15) [15,45) [45,60) [60,+)

PANEL_COLUMNS = [
    "year", "month", "month_start", "panchayat", "disease",
    "p_cases", "l_cases", "total_cases", "deaths",
    *AGE_COLS, "male", "female", "remarks",
]
LINE_LIST_COLUMNS = [
    "case_id", "onset_date", "year", "month", "panchayat", "ward_no",
    "disease", "form_type", "age", "sex", "outcome",
]

# Header aliases (compared after normalising: lowercase, punctuation -> space)
FIELD_ALIASES = {
    "case_uid": ["case id", "patient id", "ihip id", "ihip case id", "unique id", "uid", "case no"],
    "name": ["patient name", "name of patient", "name", "patient s name"],
    "age": ["age", "age in years", "age years", "age yrs", "patient age"],
    "age_unit": ["age unit", "age type", "age in"],
    "sex": ["sex", "gender"],
    "disease": ["disease", "disease syndrome", "disease name", "illness", "syndrome",
                "provisional diagnosis", "diagnosis", "disease condition", "disease illness"],
    "onset": ["date of onset", "onset date", "onset", "symptom onset date", "date of onset of illness",
              "date of symptom onset", "dateofonset"],
    "report_date": ["date of reporting", "reporting date", "report date", "date of report",
                    "date of entry", "reported on", "reported date", "entry date"],
    "location": ["panchayat", "local body", "lsg", "reporting unit", "reporting location",
                 "health facility", "facility", "facility name", "phc", "location", "village",
                 "area", "place", "address panchayat", "residence"],
    "ward": ["ward", "ward no", "ward number", "ward no ", "wardno"],
    "outcome": ["outcome", "final outcome", "patient outcome", "status", "case outcome"],
    "form": ["form", "form type", "form_type", "type of form"],
}

# Optional manual spellings -> canonical panchayat. Extend freely.
DEFAULT_UNIT_ALIASES = {
    "keechery": "Amballoor",
    "amballur": "Amballoor",
    "amballoor chc": "Amballoor",
    "mulanthuruty": "Mulanthuruthy",
    "mulamthuruthy": "Mulanthuruthy",
    "edakattuvayal": "Edakkattuvayal",
    "edakkattuvayal": "Edakkattuvayal",
    "chottanikara": "Chottanikkara",
    "maneed": "Maneed",
    "poothrikka": "Poothrikka",
    "puthrikka": "Poothrikka",
    "thiruvaniyur": "Thiruvaniyoor",
}
_STOPWORDS = {"chc", "phc", "fhc", "uphc", "panchayat", "grama", "gram", "block", "health",
              "centre", "center", "hospital", "family", "primary", "community", "gp", "ward"}


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# --------------------------------------------------------------------------- #
# Reading raw files
# --------------------------------------------------------------------------- #
def _load_table(file) -> pd.DataFrame:
    """Read an uploaded/local csv/xlsx into a header-less raw DataFrame."""
    name = str(getattr(file, "name", file)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file, header=None, dtype=object)
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    width = max(len(r) for r in rows)
    return pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object)


def _all_aliases() -> set:
    return {a for v in FIELD_ALIASES.values() for a in map(_norm, v)}


def _promote_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Find the header row (skipping title rows) and return a proper DataFrame."""
    known = _all_aliases()
    best_i, best_score = None, 0
    for i in range(min(25, len(raw))):
        score = sum(_norm(c) in known for c in raw.iloc[i].tolist() if c is not None and str(c).strip())
        if score > best_score:
            best_i, best_score = i, score
    if best_i is None or best_score < 3:
        raise ValueError(
            "Could not find a header row. Expected columns like Disease, Date of Onset, "
            "Age, Sex, Panchayat/Location. Check that this is an IHIP line-list export."
        )
    df = raw.iloc[best_i + 1:].copy()
    df.columns = [str(c).strip() if c is not None else f"col_{j}" for j, c in enumerate(raw.iloc[best_i])]
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _map_columns(df: pd.DataFrame) -> dict:
    """field -> actual column name. Exact alias match first, then substring fallback."""
    norm_cols = {c: _norm(c) for c in df.columns}
    mapping, used = {}, set()
    for fld, aliases in FIELD_ALIASES.items():          # pass 1: exact
        al = [_norm(a) for a in aliases]
        for a in al:                                    # alias order = priority
            hit = next((c for c, n in norm_cols.items() if n == a and c not in used), None)
            if hit:
                mapping[fld] = hit
                used.add(hit)
                break
    for fld, aliases in FIELD_ALIASES.items():          # pass 2: substring
        if fld in mapping:
            continue
        for a in (_norm(x) for x in aliases):
            if len(a) < 4:
                continue
            hit = next((c for c, n in norm_cols.items() if a in n and c not in used), None)
            if hit:
                mapping[fld] = hit
                used.add(hit)
                break
    return mapping


# --------------------------------------------------------------------------- #
# Field cleaners
# --------------------------------------------------------------------------- #
def normalise_disease(text) -> str | None:
    t = _norm(text)
    for canon, pat in DISEASE_PATTERNS:
        if re.search(pat, t):
            return canon
    return None


class UnitMatcher:
    """Maps free-text location strings to the canonical panchayat list (master_units)."""

    def __init__(self, panchayats, aliases: dict | None = None):
        self.canon = list(panchayats)
        self.keys = {self._strip(p): p for p in self.canon}
        self.aliases = {self._strip(k): v for k, v in {**DEFAULT_UNIT_ALIASES, **(aliases or {})}.items()
                        if v in self.canon}
        self._cache: dict = {}

    @staticmethod
    def _strip(s) -> str:
        return " ".join(w for w in _norm(s).split() if w not in _STOPWORDS)

    def match(self, text) -> str | None:
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return None
        if text in self._cache:
            return self._cache[text]
        s = self._strip(text)
        out = None
        if s in self.keys:
            out = self.keys[s]
        elif s in self.aliases:
            out = self.aliases[s]
        else:
            for k, v in {**self.keys, **self.aliases}.items():      # substring either way
                if k and (k in s or (len(s) >= 5 and s in k)):
                    out = v
                    break
            if out is None:
                close = difflib.get_close_matches(s, list(self.keys) + list(self.aliases), n=1, cutoff=0.8)
                if close:
                    out = self.keys.get(close[0]) or self.aliases.get(close[0])
        self._cache[text] = out
        return out


def parse_age_years(val, unit=None) -> float:
    """'34', '34 Y', '8 M', '10 days', 5.0 -> age in (fractional) years."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]*)", str(val).strip().lower())
    if not m:
        return np.nan
    num, u = float(m.group(1)), (m.group(2) or _norm(unit or ""))[:1]
    if u == "m":
        return num / 12
    if u == "d":
        return num / 365
    return num


def clean_sex(val) -> str:
    s = _norm(val)[:1].upper() if val is not None else ""
    return s if s in ("M", "F") else "U"


def clean_outcome(val) -> str:
    s = _norm(val)
    if re.search(r"died|dead|death|expired|deceased", s):
        return "Died"
    if re.search(r"recover|discharg|cured|alive|improved", s):
        return "Recovered"
    return "Unknown"


def parse_ward(val):
    m = re.search(r"\d+", str(val)) if val is not None else None
    return int(m.group()) if m else np.nan


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, dayfirst=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):                       # older pandas
        return pd.to_datetime(series, dayfirst=True, errors="coerce")


def infer_form_from_name(filename: str) -> str | None:
    n = _norm(filename)
    if re.search(r"\bl\s*form|lform|\blab|laborator", n):
        return "L"
    if re.search(r"\bp\s*form|pform|probable|presumptive", n):
        return "P"
    return None


# --------------------------------------------------------------------------- #
# Core: raw export -> clean line list
# --------------------------------------------------------------------------- #
@dataclass
class CleanResult:
    panel: pd.DataFrame
    line_list: pd.DataFrame
    rejected: pd.DataFrame
    report: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def clean_export(file, matcher: UnitMatcher, form_hint: str | None = None, today: datetime | None = None):
    """Clean ONE raw export. Returns (line_list_with_key, rejected, info)."""
    today = today or datetime.now()
    fname = str(getattr(file, "name", file))
    df = _promote_header(_load_table(file))
    cm = _map_columns(df)
    for need in ("disease", "location"):
        if need not in cm:
            raise ValueError(f"{fname}: no '{need}' column found. Columns seen: {list(df.columns)}")
    if "onset" not in cm and "report_date" not in cm:
        raise ValueError(f"{fname}: need 'Date of Onset' or 'Date of Reporting'. Columns seen: {list(df.columns)}")

    get = lambda f: df[cm[f]] if f in cm else pd.Series([None] * len(df), index=df.index)
    out = pd.DataFrame(index=df.index)
    out["_row"] = df.index + 2                              # approximate source row for the report
    out["_source"] = fname

    # form type: per-row column > explicit hint > filename
    if "form" in cm:
        out["form_type"] = get("form").map(lambda v: "L" if _norm(v)[:1] in ("l",) else "P")
    else:
        form = form_hint or infer_form_from_name(fname)
        if form not in ("P", "L"):
            raise ValueError(f"{fname}: cannot tell if this is a P-form or L-form export. Tell me which.")
        out["form_type"] = form

    out["disease"] = get("disease").map(normalise_disease)
    out["panchayat"] = get("location").map(matcher.match)
    if "ward" not in cm:                                    # sometimes ward lives in a combined field
        out["ward_no"] = np.nan
    else:
        out["ward_no"] = get("ward").map(parse_ward)
    onset = _parse_dates(get("onset")) if "onset" in cm else pd.Series(pd.NaT, index=df.index)
    if "report_date" in cm:
        onset = onset.fillna(_parse_dates(get("report_date")))
    out["onset_date"] = onset
    out["age_f"] = [parse_age_years(a, u) for a, u in zip(get("age"), get("age_unit"))]
    out["sex"] = get("sex").map(clean_sex)
    out["outcome"] = get("outcome").map(clean_outcome)

    # de-duplication key (names/IDs are hashed, never exported)
    def key(i):
        uid = get("case_uid").iloc[i] if "case_uid" in cm else None
        if uid is not None and not (isinstance(uid, float) and np.isnan(uid)) and str(uid).strip():
            raw = f"uid|{_norm(uid)}"
        else:
            nm = _norm(get("name").iloc[i]) if "name" in cm else ""
            raw = "|".join([nm, str(out["age_f"].iloc[i]), out["sex"].iloc[i], str(out["disease"].iloc[i]),
                            str(out["panchayat"].iloc[i]), str(out["onset_date"].iloc[i])[:10]])
        return hashlib.sha1(raw.encode()).hexdigest()
    out["key"] = [key(i) for i in range(len(df))]

    # reject unusable rows (with reasons)
    reason = pd.Series("", index=out.index)
    reason[out["disease"].isna()] = "disease not in tracked list"
    reason[(reason == "") & out["panchayat"].isna()] = "location not matched to a panchayat"
    reason[(reason == "") & out["onset_date"].isna()] = "date missing/unreadable"
    bad_range = (out["onset_date"] < "2015-01-01") | (out["onset_date"] > pd.Timestamp(today) + timedelta(days=1))
    reason[(reason == "") & bad_range] = "date out of range"
    rejected = out[reason != ""].copy()
    rejected["reject_reason"] = reason[reason != ""]
    rejected["raw_disease"] = get("disease")[reason != ""]
    rejected["raw_location"] = get("location")[reason != ""]
    good = out[reason == ""].copy()
    info = {"file": fname, "form": sorted(good["form_type"].unique().tolist()), "rows_read": len(df),
            "rows_kept": len(good), "rows_rejected": len(rejected), "column_map": cm}
    return good, rejected, info


def finalise_line_list(parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Combine P and L parts, drop duplicates (L wins over P), anonymise."""
    allr = pd.concat(parts, ignore_index=True)
    before = len(allr)
    allr["_rank"] = (allr["form_type"] != "L").astype(int)          # L first
    allr = allr.sort_values(["_rank", "onset_date"]).drop_duplicates("key", keep="first")
    dups = before - len(allr)
    ll = pd.DataFrame({
        "case_id": "S" + allr["key"].str[:10].str.upper(),           # anonymous, stable across re-uploads
        "onset_date": allr["onset_date"].dt.strftime("%Y-%m-%d"),
        "year": allr["onset_date"].dt.year,
        "month": allr["onset_date"].dt.month,
        "panchayat": allr["panchayat"],
        "ward_no": allr["ward_no"].astype("Int64"),
        "disease": allr["disease"],
        "form_type": allr["form_type"],
        "age": np.floor(allr["age_f"]).astype("Int64"),
        "sex": allr["sex"],
        "outcome": allr["outcome"],
    })
    return ll.sort_values(["onset_date", "panchayat", "disease"]).reset_index(drop=True)[LINE_LIST_COLUMNS], dups


# --------------------------------------------------------------------------- #
# Line list -> monthly panel
# --------------------------------------------------------------------------- #
def build_panel(ll: pd.DataFrame, panchayats: list[str], start=None, end=None) -> pd.DataFrame:
    if ll.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    d = ll.copy()
    d["month_start"] = pd.to_datetime(d["onset_date"]).dt.to_period("M").dt.to_timestamp()
    d["p_cases"] = (d["form_type"] == "P").astype(int)
    d["l_cases"] = (d["form_type"] == "L").astype(int)
    d["deaths"] = (d["outcome"] == "Died").astype(int)
    band = pd.cut(d["age"].astype(float), bins=AGE_BINS, labels=AGE_COLS, right=False)
    for c in AGE_COLS:
        d[c] = (band == c).astype(int)
    d["male"] = (d["sex"] == "M").astype(int)
    d["female"] = (d["sex"] == "F").astype(int)
    val_cols = ["p_cases", "l_cases", "deaths", *AGE_COLS, "male", "female"]
    agg = d.groupby(["month_start", "panchayat", "disease"])[val_cols].sum()

    start = pd.Timestamp(start) if start is not None else d["month_start"].min()
    end = pd.Timestamp(end) if end is not None else d["month_start"].max()
    months = pd.date_range(start, end, freq="MS")
    grid = pd.MultiIndex.from_product([months, panchayats, DISEASES],
                                      names=["month_start", "panchayat", "disease"])
    p = agg.reindex(grid, fill_value=0).reset_index()
    p["total_cases"] = p["p_cases"] + p["l_cases"]
    p["year"], p["month"] = p["month_start"].dt.year, p["month_start"].dt.month
    p["month_start"] = p["month_start"].dt.strftime("%Y-%m-%d")
    p["remarks"] = ""
    return p[PANEL_COLUMNS].astype({c: int for c in PANEL_COLUMNS if c not in ("month_start", "panchayat", "disease", "remarks")})


def validate_panel(p: pd.DataFrame, panchayats: list[str]) -> list[str]:
    w = []
    if p.empty:
        return ["Panel is empty."]
    if (p["total_cases"] != p["p_cases"] + p["l_cases"]).any():
        w.append("total_cases != p_cases + l_cases in some rows.")
    n_age = (p[AGE_COLS].sum(axis=1) != p["total_cases"]).sum()
    if n_age:
        w.append(f"{n_age} rows: age groups do not sum to total (missing/blank ages in the export).")
    n_sex = (p["male"] + p["female"] != p["total_cases"]).sum()
    if n_sex:
        w.append(f"{n_sex} rows: male+female != total (missing/unknown sex in the export).")
    if (p["deaths"] > p["total_cases"]).any():
        w.append("deaths exceed total cases in some rows.")
    if p.duplicated(["year", "month", "panchayat", "disease"]).any():
        w.append("Duplicate month/panchayat/disease keys.")
    if not set(p["panchayat"]) <= set(panchayats):
        w.append("Unknown panchayat names present.")
    if not set(p["disease"]) <= set(DISEASES):
        w.append("Unknown disease names present.")
    return w


# --------------------------------------------------------------------------- #
# Monthly update helpers (merge a new upload into the existing sheet data)
# --------------------------------------------------------------------------- #
def merge_panel(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Months present in `new` replace the same months in `existing`."""
    if existing is None or existing.empty:
        return new
    months = set(new["month_start"])
    keep = existing[~existing["month_start"].astype(str).isin(months)]
    return pd.concat([keep[PANEL_COLUMNS], new[PANEL_COLUMNS]], ignore_index=True) \
             .sort_values(["month_start", "panchayat", "disease"]).reset_index(drop=True)


def merge_line_list(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new
    lo, hi = new["onset_date"].min()[:7], new["onset_date"].max()[:7]
    ex_m = existing["onset_date"].astype(str).str[:7]
    keep = existing[(ex_m < lo) | (ex_m > hi)]
    return pd.concat([keep[LINE_LIST_COLUMNS], new[LINE_LIST_COLUMNS]], ignore_index=True) \
             .sort_values(["onset_date", "panchayat", "disease"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# One-call entry point (this is what app.py will call)
# --------------------------------------------------------------------------- #
def clean_ihip_exports(files, master_units: pd.DataFrame, existing_panel=None, existing_line_list=None,
                       end=None, unit_aliases=None, today=None) -> CleanResult:
    """
    files: list of (file_or_path_or_UploadedFile, form_hint) ; form_hint in {'P','L',None}
    master_units: DataFrame with at least a 'panchayat' column (Phase 1 tab master_units)
    """
    panchayats = master_units["panchayat"].tolist()
    matcher = UnitMatcher(panchayats, unit_aliases)
    parts, rejs, infos = [], [], []
    for f, hint in files:
        good, rej, info = clean_export(f, matcher, hint, today)
        parts.append(good)
        rejs.append(rej)
        infos.append(info)
    ll_new, dups = finalise_line_list(parts)
    panel_new = build_panel(ll_new, panchayats, end=end)
    rejected = pd.concat(rejs, ignore_index=True) if rejs else pd.DataFrame()
    rej_cols = ["_source", "_row", "form_type", "reject_reason", "raw_disease", "raw_location"]
    rejected = rejected[[c for c in rej_cols if c in rejected.columns]]
    warns = validate_panel(panel_new, panchayats)
    if len(rejected):
        warns.append(f"{len(rejected)} rows were rejected - review the rejected table "
                     f"(most common: {rejected['reject_reason'].value_counts().idxmax()}).")
    report = {"files": infos, "duplicates_removed": dups, "cases_in_line_list": len(ll_new),
              "months_covered": f"{ll_new['onset_date'].min()[:7]} to {ll_new['onset_date'].max()[:7]}" if len(ll_new) else "-"}
    return CleanResult(
        panel=merge_panel(existing_panel, panel_new),
        line_list=merge_line_list(existing_line_list, ll_new),
        rejected=rejected, report=report, warnings=warns,
    )


# --------------------------------------------------------------------------- #
# Command line:  python data_cleaner.py --p P.xlsx --l L.csv --master master_units.csv --out out
# --------------------------------------------------------------------------- #
def _cli():
    ap = argparse.ArgumentParser(description="Clean IHIP P/L-form exports.")
    ap.add_argument("--p", nargs="*", default=[], help="P-form export file(s)")
    ap.add_argument("--l", nargs="*", default=[], help="L-form export file(s)")
    ap.add_argument("--master", required=True, help="master_units.csv")
    ap.add_argument("--panel", help="existing monthly_panel.csv to merge into")
    ap.add_argument("--linelist", help="existing ward_line_list.csv to merge into")
    ap.add_argument("--end", help="last month to include, e.g. 2026-11-01")
    ap.add_argument("--out", default="clean_out")
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    res = clean_ihip_exports(
        [(f, "P") for f in a.p] + [(f, "L") for f in a.l],
        pd.read_csv(a.master),
        pd.read_csv(a.panel) if a.panel else None,
        pd.read_csv(a.linelist) if a.linelist else None,
        end=a.end,
    )
    res.panel.to_csv(f"{a.out}/monthly_panel.csv", index=False)
    res.line_list.to_csv(f"{a.out}/ward_line_list.csv", index=False)
    res.rejected.to_csv(f"{a.out}/rejected_rows.csv", index=False)
    print(res.report)
    for w in res.warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    _cli()
