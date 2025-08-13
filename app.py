import streamlit as st
import pandas as pd
import requests, urllib.parse, re
from collections import defaultdict

# ================== 基础设置 ==================
st.set_page_config(page_title="Daily Consumption Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption Check (Google Sheets + One-click Check)")

# 👉 改成你的 Google Sheet ID（就是链接里 /d/ 后面那串）
SHEET_ID = "11Ln80T1iUp8kAPoNhdBjS1Xi5dsxSSANhGoYPa08GoA"

# 你表里用到的工作表（**名字要与下方完全一致**）
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

# ================== 工具函数 ==================
def _norm(s):
    if pd.isna(s): return ""
    return str(s).strip()

def _num(x):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return 0.0
        return float(str(x).strip())
    except:
        return 0.0

RE_UNITCALC = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*(g|ml|piece)s?\s*/\s*([a-zA-Z]+)\s*$', re.IGNORECASE)

UNIT_SYNONYMS = {
    "pcs":"piece","pc":"piece","pieces":"piece",
    "bag":"bag","bags":"bag","box":"box","boxes":"box",
    "btl":"bottle","bottle":"bottle","bottles":"bottle",
    "can":"can","cans":"can",
}
BASE_UNIT_SYNONYMS = {"pieces":"piece","pcs":"piece","pc":"piece"}

def norm_unit(u: str) -> str:
    u = _norm(u).lower()
    u = UNIT_SYNONYMS.get(u, u)
    u = BASE_UNIT_SYNONYMS.get(u, u)
    return u

@st.cache_data(show_spinner=False, ttl=60)
def load_sheet_csv(sheet_name: str) -> pd.DataFrame:
    """从 Google Sheets 读取某个 tab（CSV 导出）；自动 URL 编码 sheet name。"""
    base = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export"
    url = f"{base}?format=csv&sheet={urllib.parse.quote(sheet_name)}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text))
    return df

def load_workbook_from_gsheet():
    dfs = {}
    errors = []
    for key, (sheet, cols) in SHEETS.items():
        try:
            df = load_sheet_csv(sheet)
            # 仅保留我们定义的列，缺的补空，顺序对齐
            keep = [c for c in df.columns if c in cols]
            df = df[keep]
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            # 丢弃全空行
            df = df.dropna(how="all")
            dfs[key] = df
        except Exception as e:
            errors.append(f"读取 {sheet} 失败：{e}")
            dfs[key] = pd.DataFrame(columns=cols)
    return dfs, errors

def build_pack_map(dfs):
    pack_map = {}
    df = dfs["raw_unit"]
    for _, r in df.iterrows():
        name = _norm(r.get("Name"))
        rule = _norm(r.get("Unit calculation"))
        if not name or not rule:
            continue
        m = RE_UNITCALC.match(rule)
        if not m:
            continue
        qty, base_u, pack_u = m.groups()
        base_u = norm_unit(base_u)
        pack_u = norm_unit(pack_u)
        pack_map[name.lower()] = {
            "base_qty": float(qty),
            "base_unit": base_u,
            "pack_unit": pack_u
        }
    return pack_map

def convert_to_base(name, qty, unit, pack_map):
    if qty == 0: return 0.0
    u = norm_unit(unit)
    nm_key = _norm(name).lower()
    if u in ["g","ml","piece"]:
        return qty
    rule = pack_map.get(nm_key)
    if rule and u == rule["pack_unit"]:
        return qty * rule["base_qty"]
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
        theo = theoretical_map.get(name, 0.0)
        act  = actual_map.get(name, 0.0)
        diff = act - theo
        # Diff%
        if abs(theo) < 1e-9:
            if abs(act) < 1e-9:
                pct = 0.0
            else:
                pct = 1.0 if diff > 0 else -1.0
        else:
            pct = diff / theo
        # 颜色规则：红=多用；绿=少用；黑=容差内
        if pct > pct_tol:       color = "red"
        elif pct < -pct_tol:    color = "green"
        else:                   color = "black"
        if color != "black":
            items.append((abs(diff), name, theo, act, diff, pct, color))

    if not items:
        return f"<h3>{label} Pass ✅</h3>", pd.DataFrame()

    items.sort(reverse=True)  # 按“偏差量”降序显示
    rows = [
        f"<tr><td>{name}</td><td>{theo:.2f}</td><td>{act:.2f}</td>"
        f"<td style='color:{color}'>{diff:.2f} ({pct:+.0%})</td></tr>"
        for _, name, theo, act, diff, pct, color in items
    ]
    df_out = pd.DataFrame(
        [{"Name": name, "Theoretical": theo, "Actual": act, "Diff": diff, "Diff%": pct, "Type": label}
         for _, name, theo, act, diff, pct, color in items]
    )
    html = (
        f"<h3>{label} Issues</h3>"
        f"<table border=1><tr><th>Name</th><th>Theoretical</th><th>Actual</th><th>Diff</th></tr>"
        f"{''.join(rows)}</table>"
    )
    return html, df_out

# ================== 左侧：在线表格（iframe） ==================
with st.sidebar:
    st.header("📄 在线表（可协作编辑）")
    gs_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
    st.markdown(
        f"""
<iframe src="https://docs.google.com/spreadsheets/d/{SHEET_ID}/preview" 
        width="100%" height="420" frameborder="0"></iframe>
<p><a href="{gs_url}" target="_blank">在新窗口打开 Google Sheets</a></p>
""",
        unsafe_allow_html=True,
    )

# ================== 主操作区 ==================
col1, col2 = st.columns([1, 2])
with col1:
    pct_tol = st.slider("容差（±%）", 5, 50, 15, step=1) / 100
    run = st.button("▶️ 从 Google Sheets 拉取并校验")
    st.caption("颜色规则：红=用多、绿=用少、黑=容差内；Theoretical=0 且有消耗时按 ±100% 显示。")

with col2:
    st.write("")

if run:
    with st.spinner("正在从 Google Sheets 抓取并校验…"):
        dfs, errs = load_workbook_from_gsheet()
        if errs:
            for e in errs:
                st.warning(e)

        # 计算
        pack_map = build_pack_map(dfs)
        semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
        prod_qty = read_production(dfs)
        total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
        theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
        theo_semi = total_semi_need

        # 实际：RAW = AM + Purchases - PM
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

        # 实际：SEMI = AM - PM
        am_semi = defaultdict(float); pm_semi = defaultdict(float)
        for _, r in dfs["am_semi"].iterrows():
            am_semi[_norm(r["Semi"])] += _num(r["Quantity"])
        for _, r in dfs["pm_semi"].iterrows():
            pm_semi[_norm(r["Semi"])] += _num(r["Quantity"])
        actual_semi = defaultdict(float)
        for name in set(am_semi) | set(pm_semi):
            actual_semi[name] = am_semi.get(name,0.0) - pm_semi.get(name,0.0)

        # 报告
        raw_html,  raw_df  = compare_and_report(theo_raw,  actual_raw,  "RAW",  pct_tol)
        semi_html, semi_df = compare_and_report(theo_semi, actual_semi, "SEMI", pct_tol)

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


