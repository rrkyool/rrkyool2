import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# 상태 초기화
# -----------------------------
for key in ["df", "processed", "forecast", "target_col", "view", "selected"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state["view"] is None:
    st.session_state["view"] = "dashboard"

# -----------------------------
# 1. 데이터 로드
# -----------------------------
def load_data(file):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    return None

# -----------------------------
# 2. 전처리
# -----------------------------
def hampel_filter(series, window=5, n=3):
    series = series.astype(float)
    new = series.copy()
    k = 1.4826

    for i in range(len(series)):
        start = max(i-window, 0)
        end = min(i+window+1, len(series))

        win = series.iloc[start:end]
        med = np.median(win)
        mad = np.median(np.abs(win-med))

        if mad == 0:
            continue

        if abs(series.iloc[i]-med) > n*k*mad:
            new.iloc[i] = med

    return new

def denoise_series(series):
    return pd.Series(
        savgol_filter(series, 11, 2),
        index=series.index
    )

def preprocess_series(series):
    s = series.astype(float)
    s = s.interpolate(limit_direction='both')
    s = hampel_filter(s)
    s = denoise_series(s)
    return s

def plot_preprocessing(raw, processed):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=raw, name="원본", opacity=0.5))
    fig.add_trace(go.Scatter(y=processed, name="전처리"))
    return fig

# -----------------------------
# 3. 정상성
# -----------------------------
def run_stationarity_test(series):
    stat, p, *_ = adfuller(series.dropna())
    return {"p_value": p}

def run_ljungbox_test(series):
    p = acorr_ljungbox(series.dropna(), lags=[10], return_df=True)['lb_pvalue'].iloc[0]
    return {"p_value": p}

# -----------------------------
# 4. 분해
# -----------------------------
def decompose_series(series, period):
    return seasonal_decompose(series, period=period)

def plot_decomposition(result):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(y=result.observed), row=1, col=1)
    fig.add_trace(go.Scatter(y=result.trend), row=2, col=1)
    fig.add_trace(go.Scatter(y=result.seasonal), row=3, col=1)
    fig.add_trace(go.Scatter(y=result.resid), row=4, col=1)
    fig.update_layout(height=800)
    return fig

# -----------------------------
# 5. 예측
# -----------------------------
def get_forecast(train, horizon):
    last = train.iloc[-1]
    mean = np.repeat(last, horizon)

    return {
        "mean": mean,
        "lower": mean * 0.9,
        "upper": mean * 1.1
    }

def plot_forecast_result(train, result):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=train, name="Actual"))

    future = result["mean"]
    full = np.concatenate([train.values, future])

    fig.add_trace(go.Scatter(y=full, name="Forecast"))

    return fig

# -----------------------------
# UI 상태 전환
# -----------------------------
def go_detail(name):
    st.session_state.view = "detail"
    st.session_state.selected = name

def go_home():
    st.session_state.view = "dashboard"
    st.session_state.selected = None

# -----------------------------
# 대시보드
# -----------------------------
def render_dashboard():
    st.title("📊 수요 예측 대시보드")

    df = st.session_state["df"]
    processed = st.session_state["processed"]

    # -----------------------------
    # 1행: 업로드 / 전처리
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.subheader("📂 데이터 업로드")
            file = st.file_uploader("CSV 업로드")

            if file:
                df = load_data(file)
                st.session_state["df"] = df

            if df is not None:
                st.dataframe(df.head(), use_container_width=True)

    with col2:
        with st.container():
            st.subheader("⚙️ 전처리")

            if df is not None:
                col = st.selectbox("컬럼 선택", df.columns)

                if st.button("전처리 실행"):
                    st.session_state["target_col"] = col
                    processed = preprocess_series(df[col])
                    st.session_state["processed"] = processed

            if processed is not None:
                raw = df[st.session_state["target_col"]]
                st.plotly_chart(plot_preprocessing(raw, processed), use_container_width=True)
                st.button("🔍 확대", on_click=lambda: go_detail("preprocess"))

    # -----------------------------
    # 2행: 정상성 / 분해
    # -----------------------------
    if processed is not None:
        col3, col4 = st.columns(2)

        with col3:
            with st.container():
                st.subheader("📈 정상성 검정")

                adf = run_stationarity_test(processed)
                lb = run_ljungbox_test(processed)

                st.metric("ADF p-value", f"{adf['p_value']:.4f}")
                st.metric("Ljung-Box p-value", f"{lb['p_value']:.4f}")

                st.button("🔍 확대", on_click=lambda: go_detail("stationarity"))

        with col4:
            with st.container():
                st.subheader("📊 시계열 분해")

                result = decompose_series(processed, 7)
                st.plotly_chart(plot_decomposition(result), use_container_width=True)

                st.button("🔍 확대", on_click=lambda: go_detail("decompose"))

    # -----------------------------
    # 3행: 예측 (단독 row → 2칸 사용)
    # -----------------------------
    if processed is not None:
        col5, col6 = st.columns(2)

        with col5:
            with st.container():
                st.subheader("🔮 예측")

                if st.button("예측 실행"):
                    st.session_state["forecast"] = get_forecast(processed, 14)

                if st.session_state["forecast"] is not None:
                    st.plotly_chart(
                        plot_forecast_result(processed, st.session_state["forecast"]),
                        use_container_width=True
                    )

                    st.button("🔍 확대", on_click=lambda: go_detail("forecast"))

        # 오른쪽은 비워두거나 요약 넣기
        with col6:
            with st.container():
                st.subheader("📌 요약")

                if st.session_state["forecast"] is not None:
                    f = st.session_state["forecast"]["mean"]
                    st.metric("예측 평균", f"{np.mean(f):.2f}")
                    st.metric("최대값", f"{np.max(f):.2f}")
                    st.metric("최소값", f"{np.min(f):.2f}")
