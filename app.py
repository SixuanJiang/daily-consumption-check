
import streamlit as st
import pandas as pd
import re
from collections import defaultdict

# ====== 页面标题 ======
st.title("📊 Daily Consumption Check (Raw ↔ Semi ↔ Product)")

# ====== 参数设置 ======
PCT_TOLERANCE = 0.15  # 默认±15%

# ====== 上传Excel ======
uploaded_file = st.file_uploader("上传每日库存Excel", type=["xlsx"])
if uploaded_file is not None:
    # -------------------- 配置：工作表与列名 --------------------
    SHEETS = {
        "raw_unit": ("raw unit calculation", ["Name", "Unit calculation", "Type"]),
        "raw_to_semi": ("raw to semi", ["Semi/100g", "Made From", "Quantity", "Unit"]),
        "semi_to_semi": ("semi to semi", ["Semi/Unit", "Made From", "Quantity", "Unit"]),
        "semi_to_prod": ("Semi to Product", ["Product/Bowl", "Made From", "Quantity", "Unit"]),
        "raw_to_prod": ("Raw to Product", ["Product", "Made From", "Quantity", "Unit"]),
        "am_raw": ("AM_Opening_Raw", ["Ingredient", "Quantity", "Unit"]),
        "am_semi": ("AM_Opening_semi", ["Semi", "Quantity", "Unit"]),
        "purch_raw": ("Purchases_Raw", ["Ingredient", "Quantity", "Unit"]),
        "pm_raw": ("PM_Ending_Raw", ["Ingredient", "Quantity", "Unit"]),
        "pm_semi": ("PM_Ending_semi", ["Semi", "Quantity", "Unit"]),
        "prod_qty": ("Dish_Production", ["Product", "Quantity"]),
    }

    # ====== 工具函数 ======
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
    UNIT_SYNONYMS = {"pcs":"piece","pc":"piece","pieces":"piece","bag":"bag","bags":"bag","box":"box","boxes":"box",
                     "btl":"bottle","bottle":"bottle","bottles":"bottle","can":"can","cans":"can"}
    BASE_UNIT_SYNONYMS = {"pieces":"piece","pcs":"piece","pc":"piece"}

    def norm_unit(u: str) -> str:
        u = _norm(u).lower()
        u = UNIT_SYNONYMS.get(u, u)
        u = BASE_UNIT_SYNONYMS.get(u, u)
        return u

    def load_wb(file):
        xls = pd.ExcelFile(file)
        dfs = {}
        for key, (sheet, cols) in SHEETS.items():
            if sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                df = df.dropna(how="all")
                dfs[key] = df
            else:
                dfs[key] = pd.DataFrame(columns=cols)
        return dfs

    # ====== 单位换算表 ======
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
        if qty == 0:
            return 0.0
        u = norm_unit(unit)
        nm_key = name.lower()
        if u in ["g", "ml", "piece"]:
            return qty
        rule = pack_map.get(nm_key)
        if rule and u == rule["pack_unit"]:
            return qty * rule["base_qty"]
        return qty

    # ====== BOM 构建 ======
    def build_bom_maps(dfs):
        semi_raw = defaultdict(lambda: defaultdict(float))
        semi_semi = defaultdict(lambda: defaultdict(float))
        prod_semi = defaultdict(lambda: defaultdict(float))
        prod_raw = defaultdict(lambda: defaultdict(float))
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

    def collect_actuals(dfs, pack_map):
        am_raw = defaultdict(float)
        purch = defaultdict(float)
        pm_raw = defaultdict(float)
        for _, r in dfs["am_raw"].iterrows():
            am_raw[_norm(r["Ingredient"])] += convert_to_base(r["Ingredient"], _num(r["Quantity"]), r["Unit"], pack_map)
        for _, r in dfs["purch_raw"].iterrows():
            purch[_norm(r["Ingredient"])] += convert_to_base(r["Ingredient"], _num(r["Quantity"]), r["Unit"], pack_map)
        for _, r in dfs["pm_raw"].iterrows():
            pm_raw[_norm(r["Ingredient"])] += convert_to_base(r["Ingredient"], _num(r["Quantity"]), r["Unit"], pack_map)
        actual_raw = defaultdict(float)
        for name in set(am_raw) | set(purch) | set(pm_raw):
            actual_raw[name] = am_raw.get(name, 0.0) + purch.get(name, 0.0) - pm_raw.get(name, 0.0)

        am_semi = defaultdict(float)
        pm_semi = defaultdict(float)
        for _, r in dfs["am_semi"].iterrows():
            am_semi[_norm(r["Semi"])] += _num(r["Quantity"])
        for _, r in dfs["pm_semi"].iterrows():
            pm_semi[_norm(r["Semi"])] += _num(r["Quantity"])
        actual_semi = defaultdict(float)
        for name in set(am_semi) | set(pm_semi):
            actual_semi[name] = am_semi.get(name, 0.0) - pm_semi.get(name, 0.0)

        return actual_raw, actual_semi

    def compare_and_report(theoretical_map, actual_map, label):
    rows = []
    for name in sorted(set(theoretical_map) | set(actual_map)):
        theo = theoretical_map.get(name, 0.0)
        act  = actual_map.get(name, 0.0)
        diff = act - theo

        # 计算 Diff%
        if abs(theo) < 1e-9:
            if abs(act) < 1e-9:
                pct = 0.0               # 0/0 → 0%
            else:
                pct = 1.0 if diff > 0 else -1.0  # 理论为0但有数 → 视为 ±100%
        else:
            pct = diff / theo

        # 颜色规则：红=用多、绿=用少、黑=容差内
        if pct > PCT_TOLERANCE:
            color = "red"        # overuse
        elif pct < -PCT_TOLERANCE:
            color = "green"      # underuse
        else:
            color = "black"      # within tolerance

        # 只在“超出容差或理论为0但有消耗”时才列为 issue；否则不进表
        if color != "black":
            rows.append(
                f"<tr>"
                f"<td>{name}</td>"
                f"<td>{theo:.2f}</td>"
                f"<td>{act:.2f}</td>"
                f"<td style='color:{color}'>{diff:.2f} ({pct:+.0%})</td>"
                f"</tr>"
            )

    if rows:
        return (
            f"<h3>{label} Issues</h3>"
            f"<table border=1>"
            f"<tr><th>Name</th><th>Theoretical</th><th>Actual</th><th>Diff</th></tr>"
            f"{''.join(rows)}"
            f"</table>"
        )
    else:
        return f"<h3>{label} Pass ✅</h3>"


    # ====== 主逻辑 ======
    dfs = load_wb(uploaded_file)
    pack_map = build_pack_map(dfs)
    semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
    prod_qty = read_production(dfs)
    total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
    theo_raw = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
    theo_semi = total_semi_need
    actual_raw, actual_semi = collect_actuals(dfs, pack_map)

    raw_report = compare_and_report(theo_raw, actual_raw, "RAW")
    semi_report = compare_and_report(theo_semi, actual_semi, "SEMI")

    st.markdown(raw_report, unsafe_allow_html=True)
    st.markdown(semi_report, unsafe_allow_html=True)
