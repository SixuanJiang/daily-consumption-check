# app.py
import streamlit as st
import pandas as pd
import re
import unicodedata
from io import BytesIO
from urllib.parse import quote
from collections import defaultdict
import requests

# ===================== 基本设置 =====================
st.set_page_config(page_title="Daily Consumption Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption Check (Google Sheets + One-click Check)")

# 固定：你的 Google Sheet ID（已内置，不需要每次输入）
SHEET_ID = "11Ln80T1iUp8kAPoNhdBjS1Xi5dsxSSANhGoYPa08GoA"

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

# —— 如需强指定某些标签的 gid（避免“串表”），在这里填
SHEET_GIDS_DEFAULT = {
    # 例：你截图里 raw to semi 的 gid
    "raw to semi": "1286746668",
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
    if name is None:
        return ""
    s = unicodedata.normalize("NFKC", str(name))
    s = " ".join(s.strip().split())
    return s

def normalize_headers_and_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """先清洗列名，再做别名映射，最后再清洗一次。"""
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
    return df

def normalize_and_validate(df: pd.DataFrame, required_cols: list, sheet_label: str) -> pd.DataFrame:
    df = normalize_headers_and_aliases(df)

    # 保证必需列存在
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            f"【{sheet_label}】缺少必需列：{missing}。当前列：{list(df.columns)}\n"
            f"请确保第一行表头完全为：{required_cols}（大小写与空格一致）。"
        )
        for m in missing:
            df[m] = None

    # 统一把名为 Quantity 的列转数值（别名映射后再转）
    for c in df.columns:
        if c.lower() == "quantity":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    ordered = [*required_cols, *[c for c in df.columns if c not in required_cols]]
    df = df[ordered]
    return df

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

# ===================== Google Sheets 抓取（支持 gid） =====================
def gs_export_csv_url_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def gs_export_csv_url_by_name(sheet_id: str, tab_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={quote(tab_name)}"

def fetch_csv_df(url: str) -> pd.DataFrame:
    """用 requests 加超时与清晰报错，再交给 pandas 读 CSV。"""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 403:
            raise RuntimeError("403 Forbidden：Google Sheet 可能未对“任何知道链接的人”开放‘可检视’。")
        if r.status_code == 404:
            raise RuntimeError("404 Not Found：sheet_id / gid / sheet 名称可能不对。")
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"拉取 CSV 失败：{e}")

@st.cache_data(show_spinner=False, ttl=60)
def load_from_gs(sheet_id: str, name_to_gid: dict):
    dfs = {}
    errors = []
    debug = []  # (tab, src, first_cols)

    for key, (tab, cols) in SHEETS.items():
        gid = name_to_gid.get(tab, "").strip()
        if gid:
            url = gs_export_csv_url_by_gid(sheet_id, gid)
            src_hint = f"gid={gid}"
        else:
            url = gs_export_csv_url_by_name(sheet_id, tab)
            src_hint = f"sheet={tab}"

        try:
            df_raw = fetch_csv_df(url)
            # 先做列名规范/别名，再统一转 Quantity 数值
            df = normalize_and_validate(df_raw, cols, tab)
            debug.append((tab, src_hint, list(df_raw.columns)[:6]))
            dfs[key] = df
        except Exception as e:
            errors.append(f"读取 {tab} 失败（{src_hint}）：{e}")
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

# ===================== 侧边栏（容差 & 高级设置） =====================
with st.sidebar:
    pct = st.slider("容差（±%）", 5, 50, 15, step=1) / 100
    run = st.button("🚀 运行校验", use_container_width=True)

    with st.expander("高级：gid 设置（填了就按 gid 抓，避免串表）", expanded=False):
        gid_state = {}
        for _key, (tab_name, _cols) in SHEETS.items():
            val = st.text_input(f"{tab_name}", value=SHEET_GIDS_DEFAULT.get(tab_name, ""))
            gid_state[tab_name] = val
        if st.button("保存 gid 设置", use_container_width=True):
            st.session_state["gid_map"] = gid_state
            st.success("已保存。")

    with st.expander("高级：临时覆盖 Sheet ID（可不填）", expanded=False):
        tmp_id = st.text_input("临时 Sheet ID（留空则使用内置）", value="")
        if tmp_id.strip():
            st.session_state["sheet_id_override"] = tmp_id.strip()

gid_map = st.session_state.get("gid_map", SHEET_GIDS_DEFAULT)
sheet_id_effective = st.session_state.get("sheet_id_override", "").strip() or SHEET_ID

# ===================== 主区：运行 =====================
col_app = st.container()
with col_app:
    st.subheader("校验结果")

    if run:
        with st.spinner("正在从 Google Sheets 抓取并校验…"):
            dfs, errs, debug = load_from_gs(sheet_id_effective, gid_map)
            for tab, src, cols in debug:
                st.caption(f"✔️ 抓取 `{tab}` via {src} → 原始列预览：{cols}")

            for msg in errs:
                st.warning(msg)

            # ——— 计算 ———
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

- **颜色规则**：红=用多（> +容差），绿=用少（< −容差）；当 **Theoretical=0** 且有消耗时，按 **±100%** 显示。  
- **单位**：`g / ml / piece` 或包单位（bag/box/can/bottle…）；包单位换算在 **raw unit calculation** 的 `Unit calculation` 里配置（如 `100g/can`）。  
- **列名清洗**：自动去不可见字符/多余空格，常见别名会被自动映射（如 `qty`→`Quantity`）；缺列会直接提示。  
- **权限**：若出现 403，请把 Google Sheet 设为“Anyone with the link can view（任何知道连结的人可检视）”。
""")
