# app.py
import streamlit as st
import pandas as pd
import re
import unicodedata
from io import BytesIO
from urllib.parse import quote, urlencode
from collections import defaultdict
import requests

# ===================== 基本设置 =====================
st.set_page_config(page_title="Daily Consumption – Editor & Check", page_icon="📊", layout="wide")
st.title("📊 Daily Consumption（在线编辑 + 一键校验）")

# 你的 Google Sheet ID（固定，不用用户输入）
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

# 可选：为避免串表，可填某些 tab 的 gid（浏览器地址栏 ?gid= 后面的数字）
SHEET_GIDS_DEFAULT = {
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
    return BASE_UNIT_SYNONYMS.get(u, u)

# ===================== 业务逻辑 =====================
def build_pack_map(dfs):
    pack_map = {}
    for _, r in dfs["raw_unit"].iterrows():
        name = _norm(r.get("Name")); rule = _norm(r.get("Unit calculation"))
        if not name or not rule: continue
        m = RE_UNITCALC.match(rule)
        if not m: continue
        qty, base_u, pack_u = m.groups()
        pack_map[name.lower()] = {"base_qty": float(qty), "base_unit": norm_unit(base_u), "pack_unit": norm_unit(pack_u)}
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
        pct = (0.0 if abs(theo) < 1e-9 else diff / theo) if not (abs(theo) < 1e-9 and abs(act) >= 1e-9) else (1.0 if diff > 0 else -1.0)
        color = "black"
        if pct > pct_tol: color = "red"
        elif pct < -pct_tol: color = "green"
        if color != "black": items.append((abs(diff), name, theo, act, diff, pct, color))
    if not items:
        return f"<h3>{label} Pass ✅</h3>", pd.DataFrame()
    items.sort(reverse=True)
    rows = [f"<tr><td>{n}</td><td>{t:.2f}</td><td>{a:.2f}</td><td style='color:{c}'>{d:.2f} ({p:+.0%})</td></tr>"
            for _, n, t, a, d, p, c in items]
    df_out = pd.DataFrame([{"Name": n, "Theoretical": t, "Actual": a, "Diff": d, "Diff%": p, "Type": label}
                           for _, n, t, a, d, p, c in items])
    html = f"<h3>{label} Issues</h3><table border=1><tr><th>Name</th><th>Theoretical</th><th>Actual</th><th>Diff</th></tr>{''.join(rows)}</table>"
    return html, df_out

# ===================== Google Sheets 抓取 =====================
def gs_export_csv_url_by_gid(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

def gs_export_csv_url_by_name(sheet_id: str, tab_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&sheet={quote(tab_name)}"

def fetch_csv_df(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 403: raise RuntimeError("403 Forbidden：Sheet 未对任何知道链接的人开放‘查看’。")
        if r.status_code == 404: raise RuntimeError("404 Not Found：sheet_id / gid / sheet 名称可能不对。")
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), dtype=str).fillna("")
    except Exception as e:
        raise RuntimeError(f"拉取 CSV 失败：{e}")

@st.cache_data(show_spinner=False, ttl=60)
def load_from_gs(sheet_id: str, name_to_gid: dict):
    dfs, errors, debug = {}, [], []
    for key, (tab, cols) in SHEETS.items():
        gid = name_to_gid.get(tab, "").strip()
        url = gs_export_csv_url_by_gid(sheet_id, gid) if gid else gs_export_csv_url_by_name(sheet_id, tab)
        src_hint = f"gid={gid}" if gid else f"sheet={tab}"
        try:
            df_raw = fetch_csv_df(url)
            df = normalize_and_validate(df_raw, cols, tab)
            debug.append((tab, src_hint, list(df_raw.columns)[:6]))
            dfs[key] = df
        except Exception as e:
            errors.append(f"读取 {tab} 失败（{src_hint}）：{e}")
            dfs[key] = pd.DataFrame(columns=cols)
    return dfs, errors, debug

# ===================== UI：两个 Tab =====================
tab_edit, tab_check = st.tabs(["📝 在线编辑（原生 Google Sheets）", "✅ 一键校验"])

# ---- Tab 1：在线编辑（iframe，不是 grid） ----
with tab_edit:
    st.subheader("直接在页面里编辑你的 Google Sheet")
    with st.sidebar:
        st.markdown("### 嵌入设置")
        height = st.slider("嵌入高度（px）", 600, 1400, 900, 20)
        gid_focus = st.text_input("可选：默认打开的标签页 gid（浏览器地址栏 ?gid= 后面的数字）", value="")
        st.info("⚠️ 必做：在 Google Sheet → Share → General access 设为 Anyone with the link – **Editor**。")
    base_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"
    url = base_url if not gid_focus.strip() else base_url + "?" + urlencode({"gid": gid_focus.strip()})
    st.caption("嵌入地址（可复制到新标签页）")
    st.code(url, language="text")
    st.link_button("在新标签页打开（编辑）", url, use_container_width=True)
    st.components.v1.iframe(src=url, height=height, scrolling=True)
    with st.expander("❓常见问题"):
        st.markdown(
            "- 能看但不能改：权限没开到 **Editor**。\n"
            "- 弹登录/403：让用户使用已登录 Google 的浏览器；或直接点“在新标签页打开”。\n"
            "- 定位标签页：在表里点到目标 tab，复制地址栏里的 `?gid=...`。"
        )

# ---- Tab 2：一键校验（沿用你原有逻辑） ----
with tab_check:
    st.subheader("校验结果")
    with st.sidebar:
        st.markdown("### 校验参数")
        pct = st.slider("容差（±%）", 5, 50, 15, step=1) / 100
        run = st.button("🚀 运行校验", use_container_width=True)
        with st.expander("高级：gid 固定（避免串表）", expanded=False):
            gid_state = {}
            for _key, (tab_name, _cols) in SHEETS.items():
                val = st.text_input(f"{tab_name}", value=SHEET_GIDS_DEFAULT.get(tab_name, ""))
                gid_state[tab_name] = val
            if st.button("保存 gid 设置", use_container_width=True):
                st.session_state["gid_map"] = gid_state
                st.success("已保存。")
    gid_map = st.session_state.get("gid_map", SHEET_GIDS_DEFAULT)

    if run:
        with st.spinner("正在从 Google Sheets 抓取并校验…"):
            dfs, errs, debug = load_from_gs(SHEET_ID, gid_map)
            for tab, src, cols in debug:
                st.caption(f"✔️ 抓取 `{tab}` via {src} → 原始列预览：{cols}")
            for msg in errs: st.warning(msg)

            # 计算
            try:
                pack_map = build_pack_map(dfs)
                semi_raw, semi_semi, prod_semi, prod_raw = build_bom_maps(dfs)
                prod_qty = read_production(dfs)
                total_semi_need = expand_semi_demand(prod_qty, prod_semi, semi_semi)
                theo_raw  = calc_theoretical_raw_need(prod_qty, prod_raw, total_semi_need, semi_raw)
                theo_semi = total_semi_need
            except Exception as e:
                st.error(f"构建理论用量失败：{e}"); st.stop()

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
                st.error(f"汇总实际用量失败：{e}"); st.stop()

            # 报告
            raw_html,  raw_df  = compare_and_report(theo_raw,  actual_raw,  "RAW",  pct)
            semi_html, semi_df = compare_and_report(theo_semi, actual_semi, "SEMI", pct)

        st.markdown(raw_html,  unsafe_allow_html=True)
        st.markdown(semi_html, unsafe_allow_html=True)
        if not raw_df.empty or not semi_df.empty:
            out = pd.concat([raw_df, semi_df], ignore_index=True)
            st.download_button("⬇️ 下载 Issues (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="issues.csv", mime="text/csv")

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
- **权限**：嵌入页想要可编辑，必须把 Google Sheet 设为 “Anyone with the link – **Editor**”。
""")

