# app.py
import streamlit as st
import pandas as pd
import re
import unicodedata
from io import BytesIO
from urllib.parse import quote
from collections import defaultdict

# ================= 页面与常量 =================
st.set_page_config(page_title="Daily Consumption Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption Check (Google Sheets + One-click Check)")

# 你的 Google Sheet（固定，不再需要手工输入）
SHEET_ID = "11Ln80T1iUp8kAPoNhdBjS1Xi5dsxSSANhGoYPa08GoA"

# 需要的工作表与标准列
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

# —— 可选：为每个标签预设 gid（填了就优先用 gid 拉取，永不串表）
# 复制某个标签的链接，?gid= 后面的数字就是它的 gid。
SHEET_GIDS_DEFAULT = {
    # 已知示例（你发来的链接里展示的 gid）
    # 不确定这个 gid 对应哪个标签，请在侧栏“高级：gid 设置”中对号入座再保存
    "raw to semi": "1286746668",
    # 其他标签可在侧栏逐一填写；留空就回退用标签名匹配（容易串表）
}

UNIT_SYNONYMS = {
    "pcs": "piece", "pc": "piece", "pieces": "piece",
    "bag": "bag", "bags": "bag",
    "box": "box", "boxes": "box",
    "btl": "bottle", "bottle": "bottle", "bottles": "bottle",
    "can": "can", "cans": "can",
}
BASE_UNIT_SYNONYMS = {"pieces": "piece", "pcs": "piece", "pc": "piece"}

RE_UNITCALC = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*(g|ml|piece)s?\s*/\s*([a-zA-Z]+)\s*$',
    re.IGNORECASE
)

# ================= 工具函数 =================
def _norm(s):
    if pd.isna(s): return ""
    return str(s).strip()

def _num(x):
    try:
        if pd.isna(x) or str(x).strip() == "":
            return 0.0
        return float(str(x).strip())
    except Exception:
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
    mapped = []
    for c in df.columns:
        key = c.lower()
        mapped.append(alias_map.get(key, c))
    df.columns = [_clean_header(c) for c in mapped]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            f"【{sheet_label}】缺少必需列：{missing}。当前列：{list(df.columns)}\n"
            f"请确保第一行表头完全为：{required_cols}（大小写与空格一致）。"
        )
        for m in missing:
            df[m] = None

    df = df[[*required_cols, *[c for c in df.columns if c not in required_cols]]]
    return df

def norm_unit(u: str) -> str:
    u = _norm(u).lower()
    u = UNIT_SYNONYMS.get(u, u)
    u = BASE_UNIT_SYNONYMS.get(u, u)
    return u

# ================= 业务构建 =================
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
        pack_map[name.lower()] = {
            "base_qty": float(qty),
            "base_unit": norm_unit(base_u),
            "pack_unit": norm_unit(pack_u),
        }
    return pack_map

def convert_to_base(name, qty, unit, pack_map):
    if qty == 0:
        return 0.0
    u = norm_unit(unit)
    if u in ["g", "ml", "piece"]:
        return qty
    rule = pack_map.get(_norm(name).lower())
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

# ================= Google Sheets 拉取（支持 gid） =================
def gs_export_csv_url_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def gs_export_csv_url_by_name(sheet_id: str, tab_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={quote(tab_name)}"

@st.cache_data(show_spinner=False, ttl=60)
def load_from_gs(sheet_id: str, name_to_gid: dict):
    dfs = {}
    errors = []
    debug_capture = []  # 记录每张表实际抓到的前几列，方便排查是否串表

    for key, (tab, cols) in SHEETS.items():
        gid = name_to_gid.get(tab, "").strip()
        if gid:
            url = gs_export_csv_url_by_gid(sheet_id, gid)
            src_hint = f"gid={gid}"
        else:
            url = gs_export_csv_url_by_name(sheet_id, tab)
            src_hint = f"sheet={tab}"

        try:
            df = pd.read_csv(url, dtype=str).fillna("")
            debug_capture.append((tab, src_hint, list(df.columns)[:6]))

            for c in df.columns:
                if c.lower() == "quantity":
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = normalize_and_validate(df, cols, tab)
            dfs[key] = df
        except Exception as e:
            errors.append(f"读取 {tab} 失败（{src_hint}）：{e}")
            dfs[key] = pd.DataFrame(columns=cols)

    return dfs, errors, debug_capture

# ================= 上传 Excel（可选） =================
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
                    df[c]


