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
st.caption("조위계 재구성(CSIRO) + 위성 고도계(NOAA STAR) · 1900년 기준 0 재정렬 + 1993–2010 평균선 표시")

with st.expander("📎 데이터 소스 안내"):
    st.markdown(
        """
- **장기 시계열(1880~2009)**: CSIRO 조위계 기반 재구성 (DataHub CSV)
- **위성 시기(1993~현재)**: NOAA NESDIS/STAR 최신 CSV 2종(연주기 유지/제거) 중 성공하는 것 자동 선택
- 네트워크/URL 변경 시를 대비해 데모 데이터를 자동 사용합니다.
        """
    )

# -------------------------------------------------
# 원격 CSV 로더 & 파서
# -------------------------------------------------
def fetch_csv_from_candidates(candidates, **read_csv_kwargs) -> pd.DataFrame:
    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), **read_csv_kwargs) if read_csv_kwargs else pd.read_csv(io.BytesIO(r.content))
            df["__source_url__"] = url
            return df
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("No URL candidates provided")

def read_noaa_altimetry_csv(url: str) -> pd.DataFrame:

    """
    NOAA STAR CSV(주석/가변포맷) → 표준화:
      - 날짜: (year, month) 또는 (decimal year) 또는 문자열 날짜 자동 처리
      - 값열: 뒤쪽 열부터 숫자 비율 높은 열 선택
      - NaN/NaT 안전 가드 (정수 변환은 dropna 이후에만!)
    """
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    txt = r.text

    # 1) 주석/빈줄 제거
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("NOAA CSV: 유효한 데이터 줄이 없습니다.")
    raw = "\n".join(lines)

    # 2) 유연 파싱
    df = pd.read_csv(
        io.StringIO(raw),
        sep=r"[,\s]+",
        engine="python",
        header=None,
        comment="#",
        skip_blank_lines=True
    ).dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("NOAA CSV: 열 수가 부족합니다.")

    c0 = df.iloc[:, 0]
    c1 = df.iloc[:, 1] if df.shape[1] > 1 else None

    def decyear_to_datetime(y):
        # NaN/inf 가드
        try:
            if pd.isna(y):
                return pd.NaT
            y = float(y)
            if not np.isfinite(y):
                return pd.NaT
        except Exception:
            return pd.NaT
        year = int(np.floor(y))
        rem = y - year
        return pd.Timestamp(year, 1, 1) + pd.to_timedelta(rem * 365.25, unit="D")

    d = None

        # (A) year, month 형태 (둘 다 수치 + 유효 범위)
    c0_num = pd.to_numeric(c0, errors="coerce")
    c1_num = pd.to_numeric(c1, errors="coerce") if c1 is not None else None
    if c1_num is not None:
        y_ok = c0_num.between(1800, 2100)
        m_ok = c1_num.between(1, 12)

        # ✅ 연·월이 “동시에 유효”하고 NaN이 아닌 행만 엄격 필터링
        mask = y_ok & m_ok & c0_num.notna() & c1_num.notna()
        frac = mask.mean()

        if frac > 0.9:
            # 유효 행 인덱스만 사용하여 “정수 캐스팅”
            idx = mask[mask].index
            years = c0_num.loc[idx].astype("int64")
            months = c1_num.loc[idx].astype("int64")

            # 안전하게 1일 고정
            d_conv = pd.to_datetime(
                dict(year=years, month=months, day=np.ones(len(idx), dtype="int64")),
                errors="coerce"
            )

            # 원본 인덱스에 맞춰 NaT로 채워두고 유효 위치만 대입
            d_full = pd.Series(pd.NaT, index=df.index)
            d_full.loc[idx] = d_conv
            d = d_full

            # 이후 값 열 후보는 2열부터
            value_candidates = list(range(2, df.shape[1]))


    # (B) 소수 연도
    if d is None:
        if c0_num.notna().mean() > 0.9 and c0_num.between(1800, 2100).mean() > 0.9:
            d = c0_num.map(decyear_to_datetime)

    # (C) 문자열 날짜
    if d is None:
        d_try = pd.to_datetime(c0, errors="coerce")
        if d_try.notna().mean() > 0.9:
            d = d_try

    if d is None:
        raise ValueError("NOAA CSV: 날짜 열을 해석할 수 없습니다.")

    # 값 열: 뒤에서 앞으로 숫자 비율 높은 열 사용
    valcol = None
    for col in reversed(range(1, df.shape[1])):
        col_vals = pd.to_numeric(df.iloc[:, col], errors="coerce")
        if col_vals.notna().mean() > 0.7:
            valcol = col
            break
    if valcol is None:
        raise ValueError("NOAA CSV 값 열을 찾지 못했습니다.")

    out = pd.DataFrame({
        "date": pd.to_datetime(d, errors="coerce"),
        "gmsl_mm": pd.to_numeric(df.iloc[:, valcol], errors="coerce")
    }).dropna(subset=["date", "gmsl_mm"]).sort_values("date")

    # 오늘(서울) 이후 데이터 컷
    today_seoul = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    out = out[out["date"] <= today_seoul]

    out["__source_url__"] = url
    return out


# -------------------------------------------------
# 데이터 소스 URL
# -------------------------------------------------
CSIRO_CANDIDATES = [
    # DataHub: CSIRO 재구성(1880–2009) 월별 (Time, GMSL, GMSL uncertainty)
    "https://datahub.io/core/sea-level-rise/r/csiro_recons_gmsl_mo_2015.csv"
]
ALTIMETRY_CANDIDATES = [
    # NOAA STAR 최신(연주기 유지)
    "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_keep_ref_90.csv",
    # NOAA STAR 최신(연주기 제거)
    "https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_gbl_free_all_66.csv",
]

# -------------------------------------------------
# 데모 백업
# -------------------------------------------------
def demo_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 오늘(서울) 기준까지만 생성
    today_seoul = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)

    # 1) CSIRO 구간: 1900-01 ~ 1992-12 (단, 오늘보다 미래면 오늘로 컷)
    end_a = min(today_seoul, pd.Timestamp("1992-12-01"))
    dates_a = pd.date_range("1900-01-01", end_a, freq="MS")
    mm_a = np.linspace(0, 120, len(dates_a)) + np.random.normal(0, 2, len(dates_a))
    df_a = pd.DataFrame(
        {"date": dates_a, "gmsl_mm": mm_a, "__source_url__": "DEMO: CSIRO-like"}
    )

    # 2) 위성 구간: 1993-01 ~ 오늘(서울) (최대 2025-12로 상한, 하지만 보통은 오늘로 컷)
    start_b = pd.Timestamp("1993-01-01")
    end_b = min(today_seoul, pd.Timestamp("2025-12-01"))
    if end_b < start_b:
        # 극단 상황 방어: 위성 구간이 아직 시작 전인 경우 빈 프레임
        df_b = pd.DataFrame(columns=["date", "gmsl_mm", "__source_url__"])
    else:
        dates_b = pd.date_range(start_b, end_b, freq="MS")
        last_base = float(mm_a[-1]) if len(mm_a) else 0.0
        mm_b = last_base + np.linspace(0, 110, len(dates_b)) + np.random.normal(0, 2, len(dates_b))
        df_b = pd.DataFrame(
            {"date": dates_b, "gmsl_mm": mm_b, "__source_url__": "DEMO: NOAA-like"}
        )

    return df_a, df_b
# -------------------------------------------------
# 로드 & 결합
# -------------------------------------------------
@st.cache_data(show_spinner=True, ttl=60*30)
def load_sources():
    c, n = None, None

    # CSIRO 재구성
    try:
        c = fetch_csv_from_candidates(
            CSIRO_CANDIDATES,
            usecols=["Time", "GMSL"],
            parse_dates=["Time"],
        )
        c = c.rename(columns={"Time": "date", "GMSL": "gmsl_mm"})
        c = c[["date", "gmsl_mm", "__source_url__"]].dropna().sort_values("date")
    except Exception as e:
        st.warning(f"CSIRO 실데이터 로드 실패 → 데모 대체: {e}")
        c = None

    # NOAA 위성
    try:
        last_err = None
        n = None
        for url in ALTIMETRY_CANDIDATES:
            try:
                n = read_noaa_altimetry_csv(url)
                break
            except Exception as e:
                last_err = e
        if n is None:
            raise last_err if last_err else RuntimeError("NOAA candidates failed")
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

# 결합 후 미래 데이터 제거 (서울 기준)
_cutoff = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
full = full[pd.to_datetime(full["date"], errors="coerce") <= _cutoff]
st.cache_data.clear()

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

    # 위성(altimetry) — NOAA STAR 2종 중 성공하는 것 사용
    try:
        last_err = None
        n = None
        for url in ALTIMETRY_CANDIDATES:
            try:
                n = read_noaa_altimetry_csv(url)
                break
            except Exception as e:
                last_err = e
        if n is None:
            raise last_err if last_err else RuntimeError("NOAA candidates failed")
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

_dates = pd.to_datetime(full["date"], errors="coerce").dropna()
if _dates.empty:
    st.error("유효한 날짜가 없습니다. 데이터 소스를 확인하세요.")
    st.stop()
min_year = int(_dates.min().year)
max_year = int(_dates.max().year)
default_start = max(1900, min_year)
default_end = min(2025, max_year)

st.sidebar.markdown("---")
year_range = st.sidebar.slider("표시 기간(연도)", min_value=min_year,
                               max_value=max(2025, max_year),
                               value=(default_start, default_end), step=1)

with st.sidebar.expander("데이터 소스 상태", expanded=False):
    st.write("CSIRO:", df_csi["__source_url__"].iloc[0])
    st.write("NOAA :", df_noaa["__source_url__"].iloc[0])
    st.write("NOAA 최신 시점:", pd.to_datetime(df_noaa["date"]).max().date())

# -------------------------------------------------
# 가공 (스무딩/단위 → 1900년 기준 0 재정렬 + 1993–2010 평균선)
# -------------------------------------------------
plot_df = full.copy()
plot_df["year"] = plot_df["date"].dt.year
plot_df = plot_df[(plot_df["year"] >= year_range[0]) & (plot_df["year"] <= year_range[1])]
plot_df = plot_df.sort_values("date")

# ✅ NaT 제거 (여기 추가)
_plot_dates = pd.to_datetime(plot_df["date"], errors="coerce")
plot_df = plot_df.loc[_plot_dates.notna()].copy()
plot_df["date"] = _plot_dates.loc[_plot_dates.notna()]

# 스무딩
if smooth:
    plot_df["gmsl_mm_smooth"] = plot_df.groupby("src")["gmsl_mm"].transform(lambda s: s.rolling(12, min_periods=1).mean())
    value_col = "gmsl_mm_smooth"
else:
    value_col = "gmsl_mm"

# 단위
if unit == "inch":
    plot_df["value"] = plot_df[value_col] / 25.4
    unit_label = "in"
    unit_label_ko = "인치"
else:
    plot_df["value"] = plot_df[value_col]
    unit_label = "mm"
    unit_label_ko = "mm"

# === 1900년 기준 0 재정렬 ===
baseline_year = 1900
baseline_mask = plot_df["date"].dt.year == baseline_year
if baseline_mask.any():
    baseline_val = plot_df.loc[baseline_mask, "value"].mean()
else:
    # 1900년 데이터가 없을 때는 가장 이른 연도의 평균을 사용
    first_year = int(plot_df["date"].dt.year.min())
    baseline_val = plot_df.loc[plot_df["date"].dt.year == first_year, "value"].mean()

plot_df["value_adj"] = plot_df["value"] - baseline_val

# === 1993–2010 구간 평균(수평선 용) ===
win_start = pd.Timestamp(1993, 1, 1)
win_end   = pd.Timestamp(2010, 12, 31)
mask_9310 = (plot_df["date"] >= win_start) & (plot_df["date"] <= win_end)
avg_9310 = float(plot_df.loc[mask_9310, "value_adj"].mean()) if mask_9310.any() else float(plot_df["value_adj"].mean())

# -------------------------------------------------
# 통계: 변화량/연평균 상승률 (1900=0 보정 후)
# -------------------------------------------------
def annual_rate(df):
    s = df.sort_values("date")[["date", "value_adj"]].dropna()
    if len(s) < 2: return np.nan, np.nan
    change = s["value_adj"].iloc[-1] - s["value_adj"].iloc[0]
    years = (s["date"].iloc[-1] - s["date"].iloc[0]).days / 365.25
    if years <= 0: return np.nan, np.nan
    return change, change / years

change, rate = annual_rate(plot_df)

# -------------------------------------------------
# 추세선(선형회귀) — 1900=0 보정 후 값으로 계산
# -------------------------------------------------
def trend_series(df):
    g = df[["date", "value_adj"]].dropna().sort_values("date")
    if len(g) < 2: return pd.DataFrame(columns=["date","trend"]), np.nan
    year_float = g["date"].dt.year + (g["date"].dt.dayofyear - 1) / 365.25
    p = np.polyfit(year_float.values, g["value_adj"].values, 1)  # slope, intercept
    trend_vals = np.polyval(p, year_float.values)
    return pd.DataFrame({"date": g["date"].values, "trend": trend_vals}), p[0]

trend_df, slope_per_year = trend_series(plot_df)

# -------------------------------------------------
# ECharts 빌더 (KOR 라벨 + 폰트 + 툴박스 + 기준선 + 1993–2010 평균선)
# -------------------------------------------------
def build_line_chart(df, trend_df, theme, unit_label, unit_label_ko, avg_9310, show_markers=False):
    # 전체 월 범위로 x축 구성, 각 시리즈는 누락월 None으로

    # ✅ NaT 안전 가드 추가 (들여쓰기 주의!)
    _d = pd.to_datetime(df["date"], errors="coerce").dropna()
    start = _d.min()
    end = _d.max()
    if pd.isna(start) or pd.isna(end) or start > end:
        # 데이터가 비정상이면 빈 차트 반환
        empty = Line(init_opts=opts.InitOpts(theme=theme, width="1000px", height="560px"))
        empty.add_xaxis([])
        return empty
    x = pd.period_range(start, end, freq="M").strftime("%Y-%m").tolist()

    color_map = {
        "조위계 재구성 (CSIRO)": "#ff7f50",
        "위성 고도계 (NOAA)": "#3b82f6",
        "추세선": "#10b981",
        "1993–2010 평균": "#6366f1",  # indigo-ish
    }

    chart = Line(init_opts=opts.InitOpts(theme=theme, width="1000px", height="560px"))
    chart.add_xaxis(xaxis_data=x)

    for name, g in df.groupby("src"):
        g = g.sort_values("date").copy()
        g["xkey"] = g["date"].dt.strftime("%Y-%m")
        m = dict(zip(g["xkey"], g["value_adj"].round(2)))
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

    # 추세선
    if trend_df is not None and not trend_df.empty:
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

    # ✅ 1993–2010 평균선(수평선)
    chart.set_series_opts(
        markline_opts=opts.MarkLineOpts(
            data=[
                opts.MarkLineItem(y=0, name="기준선 (1900=0)"),
                opts.MarkLineItem(y=avg_9310, name="1993–2010 평균"),
            ],
            linestyle_opts=opts.LineStyleOpts(type_="dashed", opacity=0.6),
            label_opts=opts.LabelOpts(font_family="Pretendard")
        )
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
            name=f"누적 해수면 변화 (1900=0, {unit_label_ko})",
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
    return chart

chart = build_line_chart(plot_df[["date","src","value_adj"]],
                         trend_df, theme, unit_label, unit_label_ko,
                         avg_9310=avg_9310,
                         show_markers=show_markers)
st_pyecharts(chart, height="560px",
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
        plot_df[["date","src","value_adj"]]
        .rename(columns={"date":"날짜","src":"자료원","value_adj":f"해수면({unit}, 1900=0)"})
        .reset_index(drop=True),
        use_container_width=True
    )

st.markdown("---")
st.caption("ⓒ 미림마이스터고 1학년 4반 4조 **마음바다건강조**")
