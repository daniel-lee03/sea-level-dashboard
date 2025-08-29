# -*- coding: utf-8 -*-
# 실행: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
import io
import os
from base64 import b64encode
import numpy as np
import pandas as pd
import requests
import streamlit as st

from pyecharts.charts import Line
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from streamlit_echarts import st_pyecharts

# -------------------------------------------------
# 기본 설정 + 폰트 주입
# -------------------------------------------------
st.set_page_config(page_title="해수면 상승 대시보드 (ECharts)", layout="wide", page_icon="🌊")

def inject_font_css(font_path="/fonts/Pretendard-Bold.ttf", family="Pretendard"):
    if not os.path.exists(font_path):
        st.warning(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        return
    with open(font_path, "rb") as f:
        font_data = b64encode(f.read()).decode("utf-8")
    css = f"""
    <style>
    @font-face {{
        font-family: '{family}';
        src: url(data:font/ttf;base64,{font_data}) format('truetype');
        font-weight: 700; font-style: normal; font-display: swap;
    }}
    html, body, [class*="css"] {{
        font-family: '{family}', -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, "Noto Sans KR", "Apple SD Gothic Neo",
                     "Nanum Gothic", "Malgun Gothic", sans-serif !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_font_css("/fonts/Pretendard-Bold.ttf", family="Pretendard")

# -------------------------------------------------
# 헤더
# -------------------------------------------------
st.title("🌊 전지구 평균 해수면(GMSL) — 1900s → 2025")
st.caption("조위계 재구성(CSIRO) + 위성 고도계(NOAA/DataHub) 결합 · Streamlit × ECharts")

with st.expander("📎 데이터 소스 안내"):
    st.markdown(
        """
- **장기 시계열(1880~2009)**: CSIRO 조위계 기반 재구성 (DataHub 경유 CSV 사용)
- **위성 시기(1993~현재)**: DataHub 월별 altimetry CSV → 실패 시 NOAA STAR 원본 CSV 파서로 대체
- 네트워크/URL 변경 시를 대비해 데모 데이터를 자동 사용합니다.
        """
    )

# -------------------------------------------------
# 원격 CSV 로더 & 파서
# -------------------------------------------------
def fetch_csv_from_candidates(candidates, **read_csv_kwargs) -> pd.DataFrame:
    """후보 URL을 차례대로 시도하여 읽기"""
    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            if "parse_dates" in read_csv_kwargs or "usecols" in read_csv_kwargs:
                df = pd.read_csv(io.BytesIO(r.content), **read_csv_kwargs)
            else:
                df = pd.read_csv(io.BytesIO(r.content))
            df["__source_url__"] = url
            return df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("No URL candidates provided")

def read_noaa_altimetry_csv(url: str) -> pd.DataFrame:
    """NOAA STAR CSV(주석/가변 포맷) → date, gmsl_mm 로 표준화"""
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    txt = r.text

    # 1) 주석/빈줄 제거
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("NOAA CSV: 데이터 줄이 없습니다.")
    raw = "\n".join(lines)

    # 2) 구분자 추정(콤마 우선, 아니면 공백)
    sep = "," if lines[0].count(",") > 0 else r"\s+"

    # 3) 관대한 파서
    df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python", header=None)

    # === 날짜열 추정 ===
    c0 = df.iloc[:, 0]

    def decyear_to_datetime(y):
        y = float(y)
        year = int(np.floor(y))
        rem = y - year
        return pd.Timestamp(year, 1, 1) + pd.to_timedelta(rem * 365.25, unit="D")

    # 문자열 날짜 / 소수연도 / (year,month) 케이스별 처리
    if c0.astype(str).str.contains(r"[-/]", regex=True).any():
        d = pd.to_datetime(c0, errors="coerce")
        val_candidates = list(range(1, df.shape[1]))
    elif pd.to_numeric(c0, errors="coerce").notna().all() and (c0.astype(float).between(1800, 2100)).all():
        d = c0.astype(float).map(decyear_to_datetime)
        val_candidates = list(range(1, df.shape[1]))
    elif df.shape[1] >= 2 and pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().all() and \
         pd.to_numeric(df.iloc[:, 1], errors="coerce").notna().all():
        d = pd.to_datetime(dict(year=df.iloc[:, 0].astype(int), month=df.iloc[:, 1].astype(int), day=1), errors="coerce")
        val_candidates = list(range(2, df.shape[1]))
    else:
        d = pd.to_datetime(c0, errors="coerce")
        val_candidates = list(range(1, df.shape[1]))

    # === 값 열 추정 ===
    valcol = None
    for col in val_candidates:
        if pd.api.types.is_numeric_dtype(df.iloc[:, col]) or \
           pd.to_numeric(df.iloc[:, col], errors="coerce").notna().mean() > 0.9:
            valcol = col
            break
    if valcol is None:
        # 뒤에서부터 숫자 많은 열 선택
        for col in reversed(range(1, df.shape[1])):
            if pd.to_numeric(df.iloc[:, col], errors="coerce").notna().mean() > 0.5:
                valcol = col
                break
    if valcol is None:
        raise ValueError("NOAA CSV 값 열을 찾지 못했습니다.")

    out = pd.DataFrame({"date": d, "gmsl_mm": pd.to_numeric(df.iloc[:, valcol], errors="coerce")})
    out = out.dropna().sort_values("date")
    out["__source_url__"] = url
    return out

# -------------------------------------------------
# 데이터 소스 URL
# -------------------------------------------------
CSIRO_CANDIDATES = [
    # DataHub: CSIRO 재구성(1880–2009) 월별
    "https://datahub.io/core/sea-level-rise/r/csiro_recons_gmsl_mo_2015.csv"
]
ALTIMETRY_CANDIDATES = [
    # DataHub: 위성 월별 altimetry (Time, GMSL (monthly), GMSL (smoothed))
    "https://datahub.io/core/sea-level-rise/r/csiro_alt_seas_inc.csv",
    # NOAA STAR 원본 (주석 포함, 전용 파서 사용)
    "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_free_all_66.csv",
]

# -------------------------------------------------
# 데모 백업
# -------------------------------------------------
def demo_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates_a = pd.date_range("1900-01-01", "1992-12-01", freq="MS")
    mm_a = np.linspace(0, 120, len(dates_a)) + np.random.normal(0, 2, len(dates_a))
    df_a = pd.DataFrame({"date": dates_a, "gmsl_mm": mm_a, "__source_url__": "DEMO: CSIRO-like"})

    dates_b = pd.date_range("1993-01-01", "2025-12-01", freq="MS")
    mm_b = mm_a[-1] + np.linspace(0, 110, len(dates_b)) + np.random.normal(0, 2, len(dates_b))
    df_b = pd.DataFrame({"date": dates_b, "gmsl_mm": mm_b, "__source_url__": "DEMO: NOAA-like"})
    return df_a, df_b

# -------------------------------------------------
# 로드 & 결합
# -------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60*30)
def load_sources():
    c, n = None, None

    # CSIRO 재구성(월별)
    try:
        c = fetch_csv_from_candidates(
            CSIRO_CANDIDATES,
            usecols=["Time", "GMSL"],      # DataHub 스키마
            parse_dates=["Time"],
        )
        c = c.rename(columns={"Time": "date", "GMSL": "gmsl_mm"})
        c = c[["date", "gmsl_mm", "__source_url__"]].dropna().sort_values("date")
    except Exception as e:
        st.warning(f"CSIRO 실데이터 로드 실패 → 데모 대체: {e}")
        c = None

    # 위성(altimetry)
    try:
        # 1) DataHub altimetry 우선
        try:
            n = fetch_csv_from_candidates(
                ALTIMETRY_CANDIDATES[:1],
                usecols=["Time", "GMSL (monthly)"],
                parse_dates=["Time"],
            )
            n = n.rename(columns={"Time": "date", "GMSL (monthly)": "gmsl_mm"})
            n = n[["date", "gmsl_mm", "__source_url__"]].dropna().sort_values("date")
        except Exception:
            # 2) 실패 시 NOAA 원본 파서 사용
            n = read_noaa_altimetry_csv(ALTIMETRY_CANDIDATES[1])
    except Exception as e:
        st.warning(f"위성 실데이터 로드 실패 → 데모 대체: {e}")
        n = None

    # Fallback
    if c is None or n is None:
        demo_c, demo_n = demo_fallback()
        c = demo_c if c is None else c
        n = demo_n if n is None else n

    return c, n

def unify_concat(df_a, df_b):
    """1993년 이후는 위성값 우선으로 결합"""
    a = df_a.copy(); b = df_b.copy()
    a["date"] = pd.to_datetime(a["date"]); b["date"] = pd.to_datetime(b["date"])
    a["src"] = "조위계 재구성 (CSIRO)"
    b["src"] = "위성 고도계 (NOAA)"
    start_b = b["date"].min()
    a = a[a["date"] < start_b]
    return pd.concat([a, b], ignore_index=True).sort_values("date")

df_csi, df_noaa = load_sources()
full = unify_concat(df_csi, df_noaa)

# -------------------------------------------------
# 사이드바(한국어)
# -------------------------------------------------
st.sidebar.header("⚙️ 설정")
theme = ThemeType.LIGHT  # 고정(원하면 라디오로 토글 가능)
unit = st.sidebar.radio("단위", ["mm", "inch"], index=0, horizontal=True)
smooth = st.sidebar.checkbox("12개월 이동평균(스무딩)", value=True)
show_markers = st.sidebar.checkbox("포인트 표시", value=False)

min_year = int(full["date"].dt.year.min())
max_year = int(full["date"].dt.year.max())
default_start = max(1900, min_year)
default_end = min(2025, max_year)

st.sidebar.markdown("---")
year_range = st.sidebar.slider("표시 기간(연도)", min_value=min_year,
                               max_value=max(2025, max_year),
                               value=(default_start, default_end), step=1)

# -------------------------------------------------
# 가공
# -------------------------------------------------
plot_df = full.copy()
plot_df["year"] = plot_df["date"].dt.year
plot_df = plot_df[(plot_df["year"] >= year_range[0]) & (plot_df["year"] <= year_range[1])]
plot_df = plot_df.sort_values("date")

if smooth:
    plot_df["gmsl_mm_smooth"] = plot_df.groupby("src")["gmsl_mm"].transform(lambda s: s.rolling(12, min_periods=1).mean())
    value_col = "gmsl_mm_smooth"
else:
    value_col = "gmsl_mm"

if unit == "inch":
    plot_df["value"] = plot_df[value_col] / 25.4
    unit_label = "in"
    unit_label_ko = "인치"
else:
    plot_df["value"] = plot_df[value_col]
    unit_label = "mm"
    unit_label_ko = "mm"

# -------------------------------------------------
# 통계: 변화량/연평균 상승률
# -------------------------------------------------
def annual_rate(df):
    s = df.sort_values("date")[["date", "value"]].dropna()
    if len(s) < 2: return np.nan, np.nan
    change = s["value"].iloc[-1] - s["value"].iloc[0]
    years = (s["date"].iloc[-1] - s["date"].iloc[0]).days / 365.25
    if years <= 0: return np.nan, np.nan
    return change, change / years

change, rate = annual_rate(plot_df)

# -------------------------------------------------
# 추세선(선형회귀)
# -------------------------------------------------
def trend_series(df):
    g = df[["date", "value"]].dropna().sort_values("date")
    if len(g) < 2: return pd.DataFrame(columns=["date","trend"]), np.nan
    year_float = g["date"].dt.year + (g["date"].dt.dayofyear - 1) / 365.25
    p = np.polyfit(year_float.values, g["value"].values, 1)  # slope, intercept
    trend_vals = np.polyval(p, year_float.values)
    return pd.DataFrame({"date": g["date"].values, "trend": trend_vals}), p[0]

trend_df, slope_per_year = trend_series(plot_df)

# -------------------------------------------------
# ECharts 빌더 (KOR 라벨 + 폰트 + 툴박스 + 기준선)
# -------------------------------------------------
def build_line_chart(df, trend_df, theme, unit_label, unit_label_ko, show_markers=False):
    # 전체 월 범위로 x축 구성, 각 시리즈는 누락월 None으로
    start = df["date"].min()
    end   = df["date"].max()
    x = pd.period_range(start, end, freq="M").strftime("%Y-%m").tolist()

    color_map = {
        "조위계 재구성 (CSIRO)": "#ff7f50",
        "위성 고도계 (NOAA)": "#3b82f6",
        "추세선": "#10b981",
    }

    chart = Line(init_opts=opts.InitOpts(theme=theme, width="1000px", height="540px"))
    chart.add_xaxis(xaxis_data=x)

    for name, g in df.groupby("src"):
        g = g.sort_values("date").copy()
        g["xkey"] = g["date"].dt.strftime("%Y-%m")
        m = dict(zip(g["xkey"], g["value"].round(2)))
        y_full = [m.get(xx, None) for xx in x]
        chart.add_yaxis(
            series_name=name,
            y_axis=y_full,
            is_smooth=True,
            symbol="circle",
            symbol_size=4,
            is_symbol_show=show_markers,
            is_connect_nones=False,
            linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.95, color=color_map[name]),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.18, color=color_map[name]),
        )

    if not trend_df.empty:
        trend_df = trend_df.sort_values("date").copy()
        trend_df["xkey"] = trend_df["date"].dt.strftime("%Y-%m")
        tmap = dict(zip(trend_df["xkey"], trend_df["trend"].round(2)))
        y_tr_full = [tmap.get(xx, None) for xx in x]
        chart.add_yaxis(
            series_name="추세선",
            y_axis=y_tr_full,
            is_smooth=False,
            symbol="none",
            is_connect_nones=False,
            linestyle_opts=opts.LineStyleOpts(width=2, type_="dashed", color=color_map["추세선"]),
            areastyle_opts=opts.AreaStyleOpts(opacity=0),
        )

    chart.set_global_opts(
        legend_opts=opts.LegendOpts(
            pos_top="2%", pos_left="center",
            textstyle_opts=opts.TextStyleOpts(font_family="Pretendard")
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross",
            value_formatter=f"{{value}} {unit_label_ko}"
        ),
        datazoom_opts=[opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
                       opts.DataZoomOpts(type_="inside")],
        xaxis_opts=opts.AxisOpts(
            type_="category", boundary_gap=False,
            axislabel_opts=opts.LabelOpts(font_family="Pretendard"),
            name_textstyle_opts=opts.TextStyleOpts(font_family="Pretendard"),
        ),
        yaxis_opts=opts.AxisOpts(
            name=f"누적 해수면 변화 ({unit_label_ko})",
            splitline_opts=opts.SplitLineOpts(is_show=True),
            axislabel_opts=opts.LabelOpts(font_family="Pretendard"),
            name_textstyle_opts=opts.TextStyleOpts(font_family="Pretendard"),
        ),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="PNG 저장"),
                restore=opts.ToolBoxFeatureRestoreOpts(),
                data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False),
            )
        ),
    )
    chart.set_series_opts(
        markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(y=0, name="기준선 (0)")],
            linestyle_opts=opts.LineStyleOpts(type_="dashed", opacity=0.5),
        )
    )
    return chart

chart = build_line_chart(plot_df[["date","src","value"]],
                         trend_df, theme, unit_label, unit_label_ko,
                         show_markers=show_markers)
st_pyecharts(chart, height="540px",
             key=f"echarts-{unit}-{year_range}-{smooth}-{show_markers}")

# -------------------------------------------------
# 메트릭 카드 + 표
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("기간", f"{year_range[0]}–{year_range[1]}")
with col2:
    st.metric(f"변화량 ({unit_label_ko})", f"{(0 if np.isnan(change) else change):.1f}")
with col3:
    st.metric(f"연평균 상승률 ({unit_label_ko}/년)", f"{(0 if np.isnan(rate) else rate):.2f}")
with col4:
    st.metric("데이터 포인트", f"{len(plot_df):,}")

with st.expander("🧾 원자료 미리보기"):
    st.dataframe(
        plot_df[["date","src","value"]]
        .rename(columns={"date":"날짜","src":"자료원","value":f"해수면({unit})"})
        .reset_index(drop=True),
        use_container_width=True
    )

st.markdown("---")
st.caption("ⓒ 미림마이스터고 1학년 4반 4조 **마음바다건강조** · 범례 클릭으로 시리즈 숨김/강조, 하단 슬라이더 기간 탐색, 우상단 툴바 PNG 저장 가능")
