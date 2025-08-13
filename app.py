# app.py
import streamlit as st
import pandas as pd
import re
import unicodedata
from io import BytesIO
from urllib.parse import urlencode, quote
from collections import defaultdict
import requests

# ===================== App setup =====================
st.set_page_config(page_title="Daily Consumption – Editor & Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption (Inline Google Sheets Editor + One-click Check)")

# Your Google Sheet ID (fixed)
SHEET_ID = "11Ln80T1iUp8kAPoNhdBjS1Xi5dsxSSANhGoYPa08GoA"

# —— Use only GIDs to fetch CSV ——
SHEET_GIDS = {
    "raw unit calculation": "1286746668",
    "raw to semi":          "584726276",
    "semi to semi":         "975746819",
    "Semi to Product":      "111768073",
    "Raw to Product":       "38216750",
    "AM_Opening_Raw":       "1832884740",
    "Purchases_Raw":        "1288563451",
    "PM_Ending_Raw":        "11457195",
    "AM_Opening_semi":      "772369686",
    "PM_Ending_semi":       "880869121",
    "Dish_Production":      "878974890",
}

# Required tabs and columns (strict match)
SHEETS = {
    "raw_unit":       ("raw unit calculation", ["Name", "Unit calculation", "Type"]),
    "raw_to_semi":    ("raw to semi",         ["Semi/100g", "Made From", "Quantity", "Unit"]),
    "semi_to_semi":   ("semi to semi",        ["Semi/Unit", "Made From", "Quantity", "Unit"]),
    "semi_to_prod":   ("Semi to Product",     ["Product/Bowl", "Made From", "Quantity", "Unit"]),
    "raw_to_prod":    ("Raw to Product",      ["Product", "Made From", "Quantity", "Unit"]),
    "am_raw":         ("AM_Opening_Raw",      ["Ingredient", "Quantity", "Unit"]),
    "am_semi":        ("AM_Opening_semi",     ["Semi", "Quantity", "Unit"]),
    "purch_raw":      ("Purchases_Raw",       ["Ingredient", "Quantity", "Unit"]),
    "pm_raw":         ("PM_Ending_Raw",       ["Ingredient", "Quantity", "Unit"]),
    "pm_semi":        ("PM_Ending_semi",      ["Semi", "Quantity", "Unit"]),
    "prod_qty":       ("Dish_Production",     ["Product", "Quantity"]),
}

UNIT_SYNONYMS = {
    "pcs":"piece","pc":"piece","pieces":"piece",
    "bag":"bag","bags":"bag","box":"box","boxes":"box",
    "btl":"bottle","bottle":"bottle","bottles":"bottle",
    "can":"can","cans":"can",
}
BASE_UNIT_SYNONYMS = {"pieces":"piece","pcs":"piece","pc":"piece"}

RE_UNITCALC = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*(g|ml|piece)s?\s*/\s*([a-zA-Z]+)\s*$', re.IGNORECASE)

# ===================== Helpers =====================
def _norm(s):
    if pd.isna(s): return ""
    return str(s).strip()

def _num(x):
    try:
        if pd.isna(x) or str(x).strip()=="":
            return 0.0
        return float(str(x).strip())
    except:
        return 0.0

def _clean_header(name: str) -> str:
    if name is None: return ""
    s = unicodedata.normalize("NFKC", str(name))
    return " ".join(s.strip().split())

def normalize_headers_and_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_header(c) for c in df.columns]
    alias_map = {
        "ingredient": "Ingredient", "ingredients": "Ingredient",
        "qty": "Quantity", "amount": "Quantity",
        "product/bowl": "Product/Bowl",
        "semi/100 g": "Semi/100g", "semi/unit": "Semi/Unit",
        "product ": "Product", " semi": "Semi",
    }
    new_cols = []
    for c in df.columns:
        mapped = alias_map.get(c.lower())
        new_cols.append(mapped if mapped else c)
    df.columns = [_clean_header(c) for c in new_cols]
    return df

def normalize_and_validate(df: pd.DataFrame, required_cols: list, sheet_label: str) -> pd.DataFrame:
    df = normalize_headers_and_aliases(df)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"[{sheet_label}] is missing required columns: {missing}. Current columns: {list(df.columns)}")
        for m in missing: df[m] = None
    for c in df.columns:
        if c.lower() == "quantity":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    ordered = [*required_cols, *[c for c in df.columns if c not in required_cols]]
    return df[ordered]

def norm_unit(u: str) -> str:
    u = _norm(u).lower()
    u = UNIT_SYNONYMS.get(u, u)
    u = BASE_UNIT_SYNONYMS.get(u, u)
    return u

# ===================== Core logic =====================
def build_pack_map(dfs):
    pack_map = {}
    df = dfs["raw_unit"]
    for _, r in df.iterrows():
        name = _norm(r.get("Name")); rule = _norm(r.get("Unit calculation"))
        if not name or not rule: continue
        m = RE_UNITCALC.match(rule)
        if not m: continue
        qty, base_u, pack_u = m.groups()
        pack_map[name.lower()] = {
            "base_qty": float(qty),
            "base_unit": norm_unit(base_u),
            "pack_unit": norm_unit(pack_u)
        }
    return pack_map

def convert_to_base(name, qty, unit, pack_map):
    if qty == 0: return 0.0
    u = norm_unit(unit); key = _norm(name).lower()
    if u in ["g","ml","piece"]: return qty
    rule = pack_map.get(key)
    if rule and u == rule["pack_unit"]: return qty * rule["base_qty"]
    return qty

def build_bom_maps(dfs):
    semi_raw  = defaultdict(lambda: defaultdict(float))
    semi_semi = defaultdict(lambda: defaultdict(float))
    prod_semi = defaultdict(lambda: defaultdict(float))
    prod_raw  = defaultdict(lambda: defaultdict(float))
    for _, r in dfs["raw_to_semi"].iterrows():
        semi_raw[_norm(r["Semi/100g"])][_norm(r["Made From"])] += _num(r["Quantity"])
    for _, r in dfs["semi_to_semi"].iterrows():
        semi_semi[_norm(r["Semi/Unit"])][_norm(r["Made From"])] += _num(r["Quantity"])
    for _, r in dfs["semi_to_prod"].iterrows():
        prod_semi[_norm(r["Product/Bowl"])][_norm(r["Made From"])] += _num(r["Quantity"])
    for _, r in dfs["raw_to_prod"].iterrows():
        prod_raw[_norm(r["Product"])][_norm(r["Made From"])] += _num(r["Quantity"])
    return semi_raw, semi_semi, prod_semi, prod_raw

def read_production(dfs):
    prod_qty = defaultdict(float)
    for _, r in dfs["prod_qty"].iterrows():
        prod_qty[_norm(r["Product"])] += _num(r["Quantity"])
    return prod_qty

def expand_semi_demand(prod_qty, prod_semi, semi_semi):
    total = defaultdict(float)
    for prod, qty in prod_qty.items():
        for semi, per_unit in prod_semi.get(prod, {}).items():
            total[semi] += qty * per_unit
    queue = list(total.items())
    while queue:
        semi, need = queue.pop()
        for child, per_unit in semi_semi.get(semi, {}).items():
            add = need * per_unit
            total[child] += add
            queue.append((child, add))
    return total

def calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw):
    raw_need = defaultdict(float)
    for prod, qty in prod_qty.items():
        for raw, per_unit in prod_raw.get(prod, {}).items():
            raw_need[raw] += qty * per_unit
    for semi, sneed in total_semi_need.items():
        for raw, per_unit in semi_raw.get(semi, {}).items():
            raw_need[raw] += sneed * per_unit
    return raw_need

def compare_and_report(theoretical_map, actual_map, label, pct_tol):
    items = []
    for name in sorted(set(theoretical_map) | set(actual_map)):
        theo = theoretical_map.get(name, 0.0); act = actual_map.get(name, 0.0)
        diff = act - theo
        if abs(theo) < 1e-9:
            pct = 0.0 if abs(act) < 1e-9 else (1.0 if diff > 0 else -1.0)
        else:
            pct = diff / theo
        color = "black"
        if pct > pct_tol: color = "red"
        elif pct < -pct_tol: color = "green"
        if color != "black":
            items.append((abs(diff), name, theo, act, diff, pct, color))
    if not items:
        return f"<h3>{label} Pass ✅</h3>", pd.DataFrame()
    items.sort(reverse=True)
    rows = [
        f"<tr><td>{n}</td><td>{t:.2f}</td><td>{a:.2f}</td>"
        f"<td style='color:{c}'>{d:.2f} ({p:+.0%})</td></tr>"
        for _, n, t, a, d, p, c in items
    ]
    df_out = pd.DataFrame(
        [{"Name": n, "Theoretical": t, "Actual": a, "Diff": d, "Diff%": p, "Type": label}
         for _, n, t, a, d, p, c in items]
    )
    html = (
        f"<h3>{label} Issues</h3>"
        f"<table border=1><tr><th>Name</th><th>Theoretical</th><th>Actual</th><th>Diff</th></tr>"
        f"{''.join(rows)}</table>"
    )
    return html, df_out

# ===================== Google Sheets fetching (GID only) =====================
def gs_export_csv_url_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def fetch_csv_df(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 403: raise RuntimeError("403 Forbidden: the sheet must allow 'Anyone with the link – Viewer' for CSV export.")
        if r.status_code == 404: raise RuntimeError("404 Not Found: invalid sheet_id or gid.")
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"CSV fetch failed: {e}")

@st.cache_data(show_spinner=False, ttl=60)
def load_from_gs_using_gids(sheet_id: str):
    dfs, errors, debug = {}, [], []
    for key, (tab, cols) in SHEETS.items():
        gid = SHEET_GIDS.get(tab, "")
        if not gid:
            errors.append(f"[{tab}] missing gid (fill it in SHEET_GIDS in code).")
            dfs[key] = pd.DataFrame(columns=cols)
            continue
        url = gs_export_csv_url_by_gid(sheet_id, gid)
        try:
            df_raw = fetch_csv_df(url)
            df = normalize_and_validate(df_raw, cols, tab)
            debug.append((tab, f"gid={gid}", list(df_raw.columns)[:6]))
            dfs[key] = df
        except Exception as e:
            errors.append(f"Read {tab} failed (gid={gid}): {e}")
            dfs[key] = pd.DataFrame(columns=cols)
    return dfs, errors, debug

def export_workbook(dfs):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as w:
        for key, (sheet, cols) in SHEETS.items():
            df = dfs[key].copy()
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            df.to_excel(w, sheet_name=sheet, index=False)
    output.seek(0)
    return output

# ===================== UI: two tabs =====================
tab_edit, tab_check = st.tabs(["📝 Inline Edit (Google Sheets)", "✅ One-click Check"])

# ---- Tab 1: inline editor (iframe) ----
with tab_edit:
    st.subheader("Edit your Google Sheet right in the page")
    st.info("To edit inside the iframe, the Google Sheet must be shared as: Anyone with the link – **Editor**.")
    height = st.slider("Iframe height (px)", 600, 1400, 900, 20, key="iframe_height")
    base_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
    st.link_button("Open in new tab (edit)", base_url, use_container_width=True)
    st.components.v1.iframe(src=base_url, height=height, scrolling=True)

# ---- Tab 2: check (fixed GIDs) ----
with tab_check:
    st.subheader("Validation Results")
    pct = st.slider("Tolerance (±%)", 5, 50, 15, step=1) / 100
    run = st.button("🚀 Run Check", use_container_width=True)

    if run:
        with st.spinner("Fetching from Google Sheets and validating…"):
            dfs, errs, debug = load_from_gs_using_gids(SHEET_ID)
            for tab, src, cols in debug:
                st.caption(f"✔️ Loaded `{tab}` via {src} → columns preview: {cols}")
            for msg in errs:
                st.warning(msg)

            # Compute
            try:
                pack_map = build_pack_map(dfs)
                semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
                prod_qty = read_production(dfs)
                total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
                theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
                theo_semi = total_semi_need
            except Exception as e:
                st.error(f"Failed to build theoretical consumption: {e}")
                st.stop()

            # Actuals
            try:
                am_raw = defaultdict(float); purch = defaultdict(float); pm_raw = defaultdict(float)
                for _, r in dfs["am_raw"].iterrows():
                    am_raw[_norm(r["Ingredient"])] += convert_to_base(r["Ingredient"], _num(r["Quantity"]), r.get("Unit",""), pack_map)
                for _, r in dfs["purch_raw"].iterrows():
                    purch[_norm(r["Ingredient"])]   += convert_to_base(r["Ingredient"], _num(r["Quantity"]), r.get("Unit",""), pack_map)
                for _, r in dfs["pm_raw"].iterrows():
                    pm_raw[_norm(r["Ingredient"])]  += convert_to_base(r["Ingredient"], _num(r["Quantity"]), r.get("Unit",""), pack_map)
                actual_raw = defaultdict(float)
                for name in set(am_raw) | set(purch) | set(pm_raw):
                    actual_raw[name] = am_raw.get(name,0.0) + purch.get(name,0.0) - pm_raw.get(name,0.0)

                am_semi = defaultdict(float); pm_semi = defaultdict(float)
                for _, r in dfs["am_semi"].iterrows():
                    am_semi[_norm(r["Semi"])] += _num(r["Quantity"])
                for _, r in dfs["pm_semi"].iterrows():
                    pm_semi[_norm(r["Semi"])] += _num(r["Quantity"])
                actual_semi = defaultdict(float)
                for name in set(am_semi) | set(pm_semi):
                    actual_semi[name] = am_semi.get(name,0.0) - pm_semi.get(name,0.0)
            except Exception as e:
                st.error(f"Failed to aggregate actuals: {e}")
                st.stop()

            # Report
            raw_html,  raw_df  = compare_and_report(theo_raw,  actual_raw,  "RAW",  pct)
            semi_html, semi_df = compare_and_report(theo_semi, actual_semi, "SEMI", pct)

        st.markdown(raw_html,  unsafe_allow_html=True)
        st.markdown(semi_html, unsafe_allow_html=True)

        if not raw_df.empty or not semi_df.empty:
            out = pd.concat([raw_df, semi_df], ignore_index=True)
            st.download_button(
                "⬇️ Download Issues (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="issues.csv",
                mime="text/csv"
            )

# ===================== Help =====================
with st.expander("📘 Sheet spec"):
    st.markdown("""
- **Required sheets & columns (exact match)**  
  - raw unit calculation: `Name`, `Unit calculation` (e.g. `100g/can`), `Type`  
  - raw to semi: `Semi/100g`, `Made From`, `Quantity`, `Unit`  
  - semi to semi: `Semi/Unit`, `Made From`, `Quantity`, `Unit`  
  - Semi to Product: `Product/Bowl`, `Made From`, `Quantity`, `Unit`  
  - Raw to Product: `Product`, `Made From`, `Quantity`, `Unit`  
  - AM_Opening_Raw / Purchases_Raw / PM_Ending_Raw: `Ingredient`, `Quantity`, `Unit`  
  - AM_Opening_semi / PM_Ending_semi: `Semi`, `Quantity`, `Unit`  
  - Dish_Production: `Product`, `Quantity`

- **Permissions**  
  - Inline editing (iframe): set Google Sheet to “Anyone with the link – **Editor**”.  
  - CSV export (for validation): at least “Anyone with the link – Viewer”.
""")
