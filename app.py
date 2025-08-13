import streamlit as st
import pandas as pd
from collections import defaultdict
import re
from io import BytesIO

st.set_page_config(page_title="Daily Consumption Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption Check (Raw ↔ Semi ↔ Product)")

# ====== 固定的工作表定义（表名 & 列名）======
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

# ====== 初始化 session 中的 DataFrames（每张表一张空表，便于无上传也能编辑）======
def blank_book():
    return {
        key: pd.DataFrame(columns=cols) for key, (_, cols) in SHEETS.items()
    }

if "dfs" not in st.session_state:
    st.session_state.dfs = blank_book()

# ====== 小工具 ======
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
        # 颜色
        if pct > pct_tol:       color = "red"   # 多用
        elif pct < -pct_tol:    color = "green" # 少用
        else:                   color = "black" # 容差内
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

# ====== 侧边栏：上传/新建/导出 ======
with st.sidebar:
    st.header("📁 数据源")
    up = st.file_uploader("上传工作簿（.xlsx）", type=["xlsx"])
    col_a, col_b = st.columns(2)
    if col_a.button("从模板新建"):
        st.session_state.dfs = blank_book()
        st.success("已载入空白模板。")
    if up is not None:
        xls = pd.ExcelFile(up)
        new = {}
        for key, (sheet, cols) in SHEETS.items():
            if sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet).dropna(how="all")
                # 只保留定义中的列顺序（多余列忽略，缺失列补空）
                keep = [c for c in df.columns if c in cols]
                df = df[keep]
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[cols]
                new[key] = df
            else:
                new[key] = pd.DataFrame(columns=cols)
        st.session_state.dfs = new
        st.success("已载入上传文件。")

    # 导出整本 Excel（把当前页面数据按标准表名写回）
    def export_workbook(dfs):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as w:
            for key, (sheet, cols) in SHEETS.items():
                df = dfs[key].copy()
                # 保证列顺序一致
                for c in cols:
                    if c not in df.columns:
                        df[c] = None
                df = df[cols]
                df.to_excel(w, sheet_name=sheet, index=False)
        output.seek(0)
        return output

    st.divider()
    if st.button("⬇️ 导出当前工作簿（.xlsx）"):
        buf = export_workbook(st.session_state.dfs)
        st.download_button("点击下载", data=buf, file_name="inventory.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ====== Tab：编辑 / 校验 ======
tab_edit, tab_check = st.tabs(["✏️ 在线编辑", "✅ 运行校验"])

with tab_edit:
    st.write("在下方直接修改数据，改完可到“✅ 运行校验”执行检查，或在侧边栏导出为 Excel。")
    editable_keys = [
        "prod_qty", "am_raw", "purch_raw", "pm_raw", "am_semi", "pm_semi", "raw_unit"
    ]
    for key in editable_keys:
        sheet_name, cols = SHEETS[key]
        st.subheader(sheet_name)
        # 只展示/编辑定义列
        df_show = st.session_state.dfs[key]
        # data_editor 支持增/删/改
        edited = st.data_editor(
            df_show, num_rows="dynamic",
            column_config={c: st.column_config.Column(required=False) for c in cols},
            use_container_width=True, key=f"editor_{key}"
        )
        st.session_state.dfs[key] = edited

with tab_check:
    left, right = st.columns([1, 2])
    with left:
        pct = st.slider("容差（±%）", 5, 50, 15, step=1) / 100
        run = st.button("Run check")

    if run:
        with st.spinner("正在校验…"):
            dfs = st.session_state.dfs
            # 计算
            pack_map = build_pack_map(dfs)
            semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
            prod_qty = read_production(dfs)
            total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
            theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
            theo_semi = total_semi_need

            # 实际
            # RAW: AM + Purchases - PM
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

            # SEMI: AM - PM
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

        # 下载 Issues CSV
        if not raw_df.empty or not semi_df.empty:
            out = pd.concat([raw_df, semi_df], ignore_index=True)
            st.download_button(
                "⬇️ 下载 Issues (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name="issues.csv",
                mime="text/csv"
            )

# ====== 页面底部帮助 ======
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

- **颜色规则**：红=用多（> +容差），绿=用少（< −容差），黑=容差内；当 **Theoretical=0** 且有消耗时，按 **±100%** 判断。
- **单位**：`g / ml / piece` 或包单位（bag/box/can/bottle…）；包单位换算在 **raw unit calculation** 的 `Unit calculation` 里配置（如 `100g/can`）。
""")


