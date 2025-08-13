# app.py
import streamlit as st
import pandas as pd
import re
import unicodedata
from io import BytesIO
from urllib.parse import urlencode, quote
from collections import defaultdict
import requests

# ===================== 基本设置 =====================
st.set_page_config(page_title="Daily Consumption – Editor & Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption（在线编辑 + 一键校验）")

# 你的 Google Sheet ID（固定）
SHEET_ID = "11Ln80T1iUp8kAPoNhdBjS1Xi5dsxSSANhGoYPa08GoA"

# —— 只用 gid 抓取（你给的截图里提取到的所有 gid）——
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

# 需要的标签及标准列（严格匹配）
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

# ===================== 小工具 =====================
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
        st.error(f"【{sheet_label}】缺少必需列：{missing}。当前列：{list(df.columns)}")
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

# ===================== 业务逻辑 =====================
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

# ===================== Google Sheets 抓取（只用 gid） =====================
def gs_export_csv_url_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def fetch_csv_df(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 403: raise RuntimeError("403 Forbidden：Sheet 需设为 'Anyone with the link – Viewer'（抓取 CSV）。")
        if r.status_code == 404: raise RuntimeError("404 Not Found：sheet_id / gid 可能不对。")
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"拉取 CSV 失败：{e}")

@st.cache_data(show_spinner=False, ttl=60)
def load_from_gs_using_gids(sheet_id: str):
    dfs, errors, debug = {}, [], []
    for key, (tab, cols) in SHEETS.items():
        gid = SHEET_GIDS.get(tab, "")
        if not gid:
            errors.append(f"【{tab}】缺少 gid（请在代码的 SHEET_GIDS 中补齐）。")
            dfs[key] = pd.DataFrame(columns=cols)
            continue
        url = gs_export_csv_url_by_gid(sheet_id, gid)
        try:
            df_raw = fetch_csv_df(url)
            df = normalize_and_validate(df_raw, cols, tab)
            debug.append((tab, f"gid={gid}", list(df_raw.columns)[:6]))
            dfs[key] = df
        except Exception as e:
            errors.append(f"读取 {tab} 失败（gid={gid}）：{e}")
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

# ===================== UI：两个 Tab =====================
tab_edit, tab_check = st.tabs(["📝 在线编辑（原生 Google Sheets）", "✅ 一键校验"])

# ---- Tab 1：在线编辑（iframe） ----
with tab_edit:
    st.subheader("直接在页面里编辑 Google Sheet")
    st.info("⚠️ 在 iframe 内可编辑：把 Google Sheet 权限设为 “Anyone with the link – **Editor**”。")
    height = st.slider("嵌入高度（px）", 600, 1400, 900, 20, key="iframe_height")
    base_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
    st.link_button("在新标签页打开（编辑）", base_url, use_container_width=True)
    st.components.v1.iframe(src=base_url, height=height, scrolling=True)

# ---- Tab 2：一键校验（固定 gid 抓取） ----
with tab_check:
    st.subheader("校验结果")
    pct = st.slider("容差（±%）", 5, 50, 15, step=1) / 100
    run = st.button("🚀 运行校验", use_container_width=True)

    if run:
        with st.spinner("正在从 Google Sheets 抓取并校验…"):
            dfs, errs, debug = load_from_gs_using_gids(SHEET_ID)
            for tab, src, cols in debug:
                st.caption(f"✔️ 抓取 `{tab}` via {src} → 原始列预览：{cols}")
            for msg in errs:
                st.warning(msg)

            # 计算
            try:
                pack_map = build_pack_map(dfs)
                semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
                prod_qty = read_production(dfs)
                total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
                theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
                theo_semi = total_semi_need
            except Exception as e:
                st.error(f"构建理论用量失败：{e}")
                st.stop()

            # 实际
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
                st.error(f"汇总实际用量失败：{e}")
                st.stop()

            # 报告
            raw_html,  raw_df  = compare_and_report(theo_raw,  actual_raw,  "RAW",  pct)
            semi_html, semi_df = compare_and_report(theo_semi, actual_semi, "SEMI", pct)

        st.markdown(raw_html,  unsafe_allow_html=True)
        st.markdown(semi_html, unsafe_allow_html=True)

        if not raw_df.empty or not semi_df.empty:
            out = pd.concat([raw_df, semi_df], ignore_index=True)
            st.download_button(
                "⬇️ 下载 Issues (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="issues.csv",
                mime="text/csv"
            )

# ===================== 帮助 =====================
with st.expander("📘 填表规范（点开查看）"):
    st.markdown("""
- **必须的工作表与列名（严格匹配）**  
  - raw unit calculation: `Name`, `Unit calculation` (如 `100g/can`), `Type`  
  - raw to semi: `Semi/100g`, `Made From`, `Quantity`, `Unit`  
  - semi to semi: `Semi/Unit`, `Made From`, `Quantity`, `Unit`  
  - Semi to Product: `Product/Bowl`, `Made From`, `Quantity`, `Unit`  
  - Raw to Product: `Product`, `Made From`, `Quantity`, `Unit`  
  - AM_Opening_Raw / Purchases_Raw / PM_Ending_Raw: `Ingredient`, `Quantity`, `Unit`  
  - AM_Opening_semi / PM_Ending_semi: `Semi`, `Quantity`, `Unit`  
  - Dish_Production: `Product`, `Quantity`

- **权限**  
  - 在页面里编辑：Google Sheet 需设为 “Anyone with the link – **Editor**”；  
  - 抓 CSV：至少 “Anyone with the link – Viewer”。
""")
