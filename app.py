import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from io import BytesIO

# -------------------- 基本设置 --------------------
st.set_page_config(page_title="Daily Consumption Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption Check (Raw ↔ Semi ↔ Product)")

# （默认容差 15%：红=用多，绿=用少，黑=容差内）
DEFAULT_TOL = 0.15

# -------------------- 固定工作表定义（表名 & 期望列名，仅用于导出/提示） --------------------
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

def blank_book():
    return {key: pd.DataFrame(columns=cols) for key, (_, cols) in SHEETS.items()}

if "dfs" not in st.session_state:
    st.session_state.dfs = blank_book()

# -------------------- 小工具 --------------------
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

# ====== 列名候选 & 容错读取（关键补丁）======
CAND = {
    "prod":  ["Product", "Product/Bowl", "Dish", "产品", "product", "product/bowl"],
    "semi":  ["Semi", "Semi/Unit", "Semi/100g", "半成品", "semi", "semi/unit", "semi/100g"],
    "ing":   ["Ingredient", "Name", "原料", "ingredient", "name"],
    "made":  ["Made From", "From", "配方原料", "made from", "from"],
    "qty":   ["Quantity", "Qty", "QTY", "数量", "quantity", "qty"],
    "unit":  ["Unit", "Units", "单位", "unit", "units"],
}

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _match_col(df: pd.DataFrame, candidates) -> str | None:
    """忽略大小写/空格匹配第一个存在的列名；找不到返 None"""
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    return None

def get_col_or_stop(df: pd.DataFrame, candidates, ctx: str) -> str:
    """找不到就页面报错并停止，避免卡住"""
    col = _match_col(df, candidates)
    if col is None:
        st.error(f"❌ 当前表缺少必要列（{ctx}）：需要其一 {candidates}，实际列：{list(df.columns)}")
        st.stop()
    return col

def getv(row: pd.Series, col_name: str):
    try:
        return row[col_name]
    except Exception:
        return ""

# -------------------- 读取工作簿（上传） --------------------
def load_wb(file) -> dict:
    xls = pd.ExcelFile(file)
    dfs = {}
    for key, (sheet, cols) in SHEETS.items():
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet).dropna(how="all")
            df = _normalize_headers(df)
            dfs[key] = df
        else:
            dfs[key] = pd.DataFrame(columns=cols)
    return dfs

# -------------------- 单位换算表 --------------------
def build_pack_map(dfs):
    pack_map = {}
    df = dfs["raw_unit"]
    if df.empty:
        return pack_map
    df = _normalize_headers(df)
    name_col = get_col_or_stop(df, ["Name","name"], "raw unit calculation 的 Name 列")
    rule_col = get_col_or_stop(df, ["Unit calculation","unit calculation"], "raw unit calculation 的 Unit calculation 列")

    for _, r in df.iterrows():
        name = _norm(getv(r, name_col))
        rule = _norm(getv(r, rule_col))
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

# -------------------- BOM 构建（容错列名版） --------------------
def build_bom_maps(dfs):
    semi_raw  = defaultdict(lambda: defaultdict(float))
    semi_semi = defaultdict(lambda: defaultdict(float))
    prod_semi = defaultdict(lambda: defaultdict(float))
    prod_raw  = defaultdict(lambda: defaultdict(float))

    # raw_to_semi
    df = dfs["raw_to_semi"]
    if not df.empty:
        df = _normalize_headers(df)
        s_col = get_col_or_stop(df, CAND["semi"] + ["Semi/100g"], "raw to semi 的 Semi 列")
        m_col = get_col_or_stop(df, CAND["made"], "raw to semi 的 Made From 列")
        q_col = get_col_or_stop(df, CAND["qty"],  "raw to semi 的 Quantity 列")
        for _, r in df.iterrows():
            semi = _norm(getv(r, s_col)); made = _norm(getv(r, m_col)); qty = _num(getv(r, q_col))
            if semi and made:
                semi_raw[semi][made] += qty

    # semi_to_semi
    df = dfs["semi_to_semi"]
    if not df.empty:
        df = _normalize_headers(df)
        s_col = get_col_or_stop(df, CAND["semi"] + ["Semi/Unit"], "semi to semi 的 Semi 列")
        m_col = get_col_or_stop(df, CAND["made"], "semi to semi 的 Made From 列")
        q_col = get_col_or_stop(df, CAND["qty"],  "semi to semi 的 Quantity 列")
        for _, r in df.iterrows():
            semi = _norm(getv(r, s_col)); made = _norm(getv(r, m_col)); qty = _num(getv(r, q_col))
            if semi and made:
                semi_semi[semi][made] += qty

    # semi_to_prod
    df = dfs["semi_to_prod"]
    if not df.empty:
        df = _normalize_headers(df)
        p_col = get_col_or_stop(df, CAND["prod"] + ["Product/Bowl"], "Semi to Product 的 Product 列")
        m_col = get_col_or_stop(df, CAND["made"], "Semi to Product 的 Made From 列")
        q_col = get_col_or_stop(df, CAND["qty"],  "Semi to Product 的 Quantity 列")
        for _, r in df.iterrows():
            prod = _norm(getv(r, p_col)); made = _norm(getv(r, m_col)); qty = _num(getv(r, q_col))
            if prod and made:
                prod_semi[prod][made] += qty

    # raw_to_prod
    df = dfs["raw_to_prod"]
    if not df.empty:
        df = _normalize_headers(df)
        p_col = get_col_or_stop(df, CAND["prod"], "Raw to Product 的 Product 列")
        m_col = get_col_or_stop(df, CAND["made"], "Raw to Product 的 Made From 列")
        q_col = get_col_or_stop(df, CAND["qty"],  "Raw to Product 的 Quantity 列")
        for _, r in df.iterrows():
            prod = _norm(getv(r, p_col)); made = _norm(getv(r, m_col)); qty = _num(getv(r, q_col))
            if prod and made:
                prod_raw[prod][made] += qty

    return semi_raw, semi_semi, prod_semi, prod_raw

# -------------------- 生产读取（容错列名版） --------------------
def read_production(dfs):
    prod_qty = defaultdict(float)
    df = dfs["prod_qty"]
    if df.empty:
        return prod_qty
    df = _normalize_headers(df)
    p_col = get_col_or_stop(df, CAND["prod"], "Dish_Production 的产品列")
    q_col = get_col_or_stop(df, CAND["qty"],  "Dish_Production 的数量列")

    for _, r in df.iterrows():
        prod = _norm(getv(r, p_col))
        qty  = _num(getv(r, q_col))
        if prod:
            prod_qty[prod] += qty
    return prod_qty

# -------------------- 理论原料需求 --------------------
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

# -------------------- 实际消耗（容错列名版） --------------------
def collect_actuals(dfs, pack_map):
    # RAW: AM + Purchases - PM
    am_raw = defaultdict(float); purch = defaultdict(float); pm_raw = defaultdict(float)

    def _acc_raw(df_key, bucket, sign=+1):
        df = dfs[df_key]
        if df.empty: return
        df = _normalize_headers(df)
        i_col = get_col_or_stop(df, CAND["ing"],  f"{df_key} 的原料列")
        q_col = get_col_or_stop(df, CAND["qty"],  f"{df_key} 的数量列")
        u_col = _match_col(df, CAND["unit"])  # unit 可选
        for _, r in df.iterrows():
            ing = _norm(getv(r, i_col))
            qty = _num(getv(r, q_col))
            unit= _norm(getv(r, u_col)) if u_col else ""
            if ing:
                bucket[ing] += sign * convert_to_base(ing, qty, unit, pack_map)

    _acc_raw("am_raw",   am_raw,  +1)
    _acc_raw("purch_raw",purch,   +1)
    _acc_raw("pm_raw",   pm_raw,  +1)

    actual_raw = defaultdict(float)
    for name in set(am_raw) | set(purch) | set(pm_raw):
        actual_raw[name] = am_raw.get(name,0.0) + purch.get(name,0.0) - pm_raw.get(name,0.0)

    # SEMI: AM - PM
    def _acc_semi(df_key, sign=+1):
        out = defaultdict(float)
        df = dfs[df_key]
        if df.empty: return out
        df = _normalize_headers(df)
        s_col = get_col_or_stop(df, CAND["semi"], f"{df_key} 的半成品列")
        q_col = get_col_or_stop(df, CAND["qty"],  f"{df_key} 的数量列")
        for _, r in df.iterrows():
            semi = _norm(getv(r, s_col))
            qty  = _num(getv(r, q_col))
            if semi:
                out[semi] += sign * qty
        return out

    am_semi = _acc_semi("am_semi", +1)
    pm_semi = _acc_semi("pm_semi", +1)

    actual_semi = defaultdict(float)
    for name in set(am_semi) | set(pm_semi):
        actual_semi[name] = am_semi.get(name,0.0) - pm_semi.get(name,0.0)

    return actual_raw, actual_semi

# -------------------- 对比 & 报表（颜色：红=用多，绿=用少） --------------------
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

        # 颜色：红=用多（> +tol），绿=用少（< -tol），黑=容差内
        if pct > pct_tol:
            color = "red"
        elif pct < -pct_tol:
            color = "green"
        else:
            color = "black"

        if color != "black":
            items.append((abs(diff), name, theo, act, diff, pct, color))

    if not items:
        return f"<h3>{label} Pass ✅</h3>", pd.DataFrame()

    # 让更“严重”的排前面
    items.sort(reverse=True)
    rows = [
        f"<tr><td>{name}</td>"
        f"<td>{theo:.2f}</td>"
        f"<td>{act:.2f}</td>"
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

# -------------------- 侧边栏：数据源（上传 / 新建 / 导出） --------------------
with st.sidebar:
    st.header("📁 数据源")
    up = st.file_uploader("上传工作簿（.xlsx）", type=["xlsx"])
    c1, c2 = st.columns(2)
    if c1.button("从模板新建"):
        st.session_state.dfs = blank_book()
        st.success("已载入空白模板。")
    if up is not None:
        try:
            dfs_new = load_wb(up)
            st.session_state.dfs = dfs_new
            st.success("已载入上传文件。")
        except Exception as e:
            st.error(f"读取失败：{e}")

    # 导出整本 Excel（把当前页面数据按标准表名写回）
    def export_workbook(dfs):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as w:
            for key, (sheet, cols) in SHEETS.items():
                df = dfs[key].copy()
                # 保证列顺序一致（缺列补空）
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[cols]
                df.to_excel(w, sheet_name=sheet, index=False)
        output.seek(0)
        return output

    if c2.button("⬇️ 导出当前工作簿（.xlsx）"):
        buf = export_workbook(st.session_state.dfs)
        st.download_button("点击下载", data=buf, file_name="inventory.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# -------------------- Tab：编辑 / 校验 --------------------
tab_edit, tab_check = st.tabs(["✏️ 在线编辑", "✅ 运行校验"])

with tab_edit:
    st.write("直接修改数据，改完去“✅ 运行校验”执行检查，或在侧边栏导出为 Excel。")
    editable_keys = ["prod_qty", "am_raw", "purch_raw", "pm_raw", "am_semi", "pm_semi", "raw_unit"]
    for key in editable_keys:
        sheet_name, cols = SHEETS[key]
        st.subheader(sheet_name)
        df_show = _normalize_headers(st.session_state.dfs[key])
        edited = st.data_editor(
            df_show, num_rows="dynamic",
            column_config={c: st.column_config.Column(required=False) for c in df_show.columns},
            use_container_width=True, key=f"editor_{key}"
        )
        st.session_state.dfs[key] = edited

with tab_check:
    left, right = st.columns([1, 2])
    with left:
        pct = st.slider("容差（±%）", 5, 50, int(DEFAULT_TOL*100), step=1) / 100.0
        run = st.button("Run check", use_container_width=True)

    if run:
        with st.spinner("正在校验…"):
            dfs = {k: _normalize_headers(v) for k, v in st.session_state.dfs.items()}

            # 调试面板：查看各表列名（出问题先看这里）
            with st.expander("🔧 调试：各表当前列名（点开查看）", expanded=False):
                for k, df in dfs.items():
                    st.write(f"**{k}**（{SHEETS[k][0]}）→ {list(df.columns)}")

            # 计算
            pack_map = build_pack_map(dfs)
            semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
            prod_qty = read_production(dfs)
            total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
            theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
            theo_semi = total_semi_need

            # 实际
            actual_raw, actual_semi = collect_actuals(dfs, pack_map)

            # 报告（颜色：红=用多，绿=用少）
            raw_html,  raw_df  = compare_and_report(theo_raw,  actual_raw,  "RAW",  pct)
            semi_html, semi_df = compare_and_report(theo_semi, actual_semi, "SEMI", pct)

        st.markdown(raw_html,  unsafe_allow_html=True)
        st.markdown(semi_html, unsafe_allow_html=True)

        # 下载 Issues CSV
        if not raw_df.empty or not semi_df.empty:
            out = pd.concat([raw_df, semi_df], ignore_index=True)
            st.download_button(
                "⬇️ 下载 Issues (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="issues.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------------------- 页面底部帮助 --------------------
with st.expander("📘 填表规范（点开查看）"):
    st.markdown("""
- **工作表名（必须存在）**  
  - raw unit calculation  
  - raw to semi / semi to semi / Semi to Product / Raw to Product  
  - AM_Opening_Raw / Purchases_Raw / PM_Ending_Raw  
  - AM_Opening_semi / PM_Ending_semi  
  - Dish_Production

- **列名容错**：大小写/空格/常见别名均可，如 `Product` / `Product/Bowl` / `Dish`，`Semi` / `Semi/Unit` 等。  
  找不到关键列时会红字提示需要的候选名称并停止。

- **颜色规则**：红=用多（> +容差），绿=用少（< −容差），黑=容差内；当 **Theoretical=0** 且有消耗时，按 **±100%** 判断。  
- **单位**：`g / ml / piece` 或包单位（bag/box/can/bottle…）；包单位换算在 **raw unit calculation** 的 `Unit calculation` 里配置（如 `100g/can`）。
""")
