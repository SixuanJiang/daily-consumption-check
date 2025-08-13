# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from urllib.parse import quote
from collections import defaultdict

# ============ 页面与常量 ============
st.set_page_config(page_title="Daily Consumption Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption Check (Google Sheets + One-click Check)")

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

# ============ 小工具 ============
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
    if name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    s = " ".join(s.strip().split())
    return s

def normalize_and_validate(df: pd.DataFrame, required_cols: list, sheet_label: str) -> pd.DataFrame:
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
        key = c.lower()
        mapped = alias_map.get(key)
        new_cols.append(mapped if mapped else c)
    df.columns = [_clean_header(c) for c in new_cols]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            f"【{sheet_label}】缺少必需列：{missing}。当前列：{list(df.columns)}\n"
            f"请确保第一行表头完全为：{required_cols}（大小写与空格一致）。"
        )
        for m in missing:
            df[m] = None
    # 调整顺序，必需列在前
    ordered = [*required_cols, *[c for c in df.columns if c not in required_cols]]
    df = df[ordered]
    return df

def norm_unit(u: str) -> str:
    u = _norm(u).lower()
    u = UNIT_SYNONYMS.get(u, u)
    u = BASE_UNIT_SYNONYMS.get(u, u)
    return u

# ============ 业务函数 ============
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
        if abs(theo) < 1e-9:
            pct = 0.0 if abs(act) < 1e-9 else (1.0 if diff > 0 else -1.0)
        else:
            pct = diff / theo
        if pct > pct_tol:       color = "red"
        elif pct < -pct_tol:    color = "green"
        else:                   color = "black"
        if color != "black":
            items.append((abs(diff), name, theo, act, diff, pct, color))

    if not items:
        return f"<h3>{label} Pass ✅</h3>", pd.DataFrame()

    items.sort(reverse=True)
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

# ============ Google Sheets 读取 ============
def gs_export_csv_url(sheet_id: str, tab_name: str) -> str:
    # 注意 sheet 参数要 URL 编码（tab 名大小写与空格必须与底部标签完全一致）
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={quote(tab_name)}"

@st.cache_data(show_spinner=False, ttl=60)
def load_from_gs(sheet_id: str):
    dfs = {}
    errors = []
    for key, (tab, cols) in SHEETS.items():
        url = gs_export_csv_url(sheet_id, tab)
        try:
            df = pd.read_csv(url, dtype=str).fillna("")
            # 尝试把数量列转为数字，其他保持字符串
            for c in df.columns:
                if c.lower() in ("quantity",):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = normalize_and_validate(df, cols, tab)
            dfs[key] = df
        except Exception as e:
            errors.append(f"读取 {tab} 失败：{e}")
            dfs[key] = pd.DataFrame(columns=cols)
    return dfs, errors

# ============ 上传 Excel 读取 ============
def load_from_xlsx(file):
    xls = pd.ExcelFile(file)
    dfs = {}
    errors = []
    for key, (tab, cols) in SHEETS.items():
        try:
            if tab in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=tab).dropna(how="all")
                df = normalize_and_validate(df, cols, tab)
                dfs[key] = df
            else:
                errors.append(f"上传文件缺少工作表：{tab}")
                dfs[key] = pd.DataFrame(columns=cols)
        except Exception as e:
            errors.append(f"读取 {tab} 失败：{e}")
            dfs[key] = pd.DataFrame(columns=cols)
    return dfs, errors

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

# ============ 界面：数据源 ============

with st.sidebar:
    st.header("📁 数据源")
    src = st.radio("选择数据源", ["Google Sheets", "上传 Excel"], horizontal=True)

    if src == "Google Sheets":
        sheet_id = st.text_input(
            "Google Sheet ID",
            placeholder="例如：11Ln80T1iUp8kAPoNhdBjS1Xi5dsxSSANhGoYPa08GoA",
        )
        if sheet_id:
            st.link_button("打开该表", f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit", help="新窗口预览")
    else:
        up = st.file_uploader("上传工作簿（.xlsx）", type=["xlsx"])

    st.divider()
    pct = st.slider("容差（±%）", 5, 50, 15, step=1) / 100
    run = st.button("🚀 运行校验", use_container_width=True)

# 预览 Google Sheets（可协作编辑）
if src == "Google Sheets":
    col_iframe, col_app = st.columns([0.45, 0.55])
    with col_iframe:
        st.subheader("在线表（可协作编辑）")
        if 'sheet_id' in locals() and sheet_id:
            st.components.v1.iframe(
                f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit?usp=sharing",
                height=520
            )
        else:
            st.info("在左侧输入 Google Sheet ID 后可预览。")
else:
    col_app = st.container()

with col_app:
    st.subheader("校验结果")
    if run:
        with st.spinner("正在抓取并校验…"):
            if src == "Google Sheets":
                if not sheet_id:
                    st.error("请在左侧输入 Google Sheet ID。")
                    st.stop()
                dfs, errs = load_from_gs(sheet_id)
            else:
                if not up:
                    st.error("请先上传 Excel 文件。")
                    st.stop()
                dfs, errs = load_from_xlsx(up)

            for msg in errs:
                st.warning(msg)

            # ——— 计算 ———
            pack_map = build_pack_map(dfs)
            semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
            prod_qty = read_production(dfs)
            total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
            theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
            theo_semi = total_semi_need

            # 实际
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

# ============ 备用：导出当前数据到 Excel（仅当是上传/已读入时可用） ============
with st.expander("⬇️ 导出当前工作簿（.xlsx）"):
    st.write("当你是从 Google Sheets 拉取时，这里导出的仅是当前拉取到的快照。")
    if src == "Google Sheets":
        if 'sheet_id' in locals() and sheet_id and st.button("导出（Google Sheets 快照）"):
            dfs, _ = load_from_gs(sheet_id)
            buf = export_workbook(dfs)
            st.download_button("点击下载", data=buf, file_name="inventory.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        if 'up' in locals() and up:
            dfs, _ = load_from_xlsx(up)
            buf = export_workbook(dfs)
            st.download_button("点击下载", data=buf, file_name="inventory.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============ 填表规范 ============
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

- **颜色规则**：红=用多（> +容差），绿=用少（< −容差）；当 **Theoretical=0** 且有消耗时，按 **±100%** 显示。  
- **单位**：`g / ml / piece` 或包单位（bag/box/can/bottle…）；包单位换算在 **raw unit calculation** 的 `Unit calculation` 里配置（如 `100g/can`）。  
- **列名清洗**：自动去不可见字符/多余空格，常见别名会被自动映射（如 `qty`→`Quantity`），缺列会在页面直接提示。
""")

