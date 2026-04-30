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
# dashboard
# -----------------------------

def render_dashboard():
    st.title("📊 단변량 수요 예측 시스템")
    
    # 1단계: 데이터 업로드 (단변량 데이터 전제)
    if st.session_state["df"] is None:
        with st.container(border=True):
            st.subheader("📂 데이터 업로드")
            file = st.file_uploader("분석할 CSV 파일을 업로드하세요 (단변량 데이터 전용)", key="main_loader")
            if file:
                df = load_data(file)
                st.session_state["df"] = df
                # 업로드 즉시 첫 번째 수치형 컬럼을 자동으로 타겟 설정
                # 단변량 데이터이므로 첫 번째 수치 컬럼을 바로 사용함
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.session_state["target_col"] = numeric_cols[0]
                    # 자동으로 전처리 실행 로직 연결
                    st.session_state["processed"] = preprocess_series(df[numeric_cols[0]])
                st.rerun()
        return

    df = st.session_state["df"]
    target = st.session_state["target_col"]
    
    # ---------------------------------------------------------
    # [1단] 데이터 정보 및 전처리 결과 비교 (상단 2열 배치로 크기 확대)
    # ---------------------------------------------------------
    col1, col2 = st.columns([1, 2]) # 가로 비중을 키워 차트 시인성 확보 [cite: 36]

    with col1:
        with st.container(border=True, height=500): # 컨테이너 높이 확대 [cite: 32]
            st.subheader("📋 데이터 요약 정보")
            st.write(f"**대상 컬럼:** `{target}`")
            st.write(df[target].describe())
            
            st.divider()
            if st.button("🔄 전처리 재실행", use_container_width=True):
                st.session_state["processed"] = preprocess_series(df[target])
                st.toast("전처리가 완료되었습니다!")
                st.rerun()

    with col2:
        with st.container(border=True, height=500):
            st.subheader("📈 전처리 전/후 시계열 비교")
            if st.session_state["processed"] is not None:
                # 기존 plot_preprocessing 함수 적용 
                fig = plot_preprocessing(df[target], st.session_state["processed"])
                st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # [2단] 기술적 분석 (정상성 및 분해)
    # ---------------------------------------------------------
    if st.session_state["processed"] is not None:
        st.divider()
        col3, col4 = st.columns([1, 2])

        with col3:
            with st.container(border=True, height=600):
                st.subheader("🔍 통계적 진단")
                # 실제 함수 적용 및 결과 표시 
                adf_res = run_stationarity_test(st.session_state["processed"])
                lb_res = run_ljungbox_test(st.session_state["processed"])
                
                st.metric("ADF (정상성) p-value", f"{adf_res['p_value']:.4f}")
                st.metric("Ljung-Box (백색잡음) p-value", f"{lb_res['p_value']:.4f}")
                
                st.info("p-value < 0.05 이면 통계적 유의성이 있음")

        with col4:
            with st.container(border=True, height=600):
                st.subheader("🧩 시계열 구성 요소 분해")
                # 기존 decompose_series 함수 적용 
                dec_res = decompose_series(st.session_state["processed"], 7)
                st.plotly_chart(plot_decomposition(dec_res), use_container_width=True)

        # ---------------------------------------------------------
        # [3단] 최종 예측 (하단 전체 너비 활용)
        # ---------------------------------------------------------
        st.divider()
        with st.container(border=True):
            st.subheader("🔮 향후 수요 예측 결과")
            f_col1, f_col2 = st.columns([1, 3])
            
            with f_col1:
                horizon = st.number_input("예측 기간 (일/시 단위)", min_value=1, max_value=365, value=14)
                if st.button("🎯 예측 실행", use_container_width=True):
                    # 기존 get_forecast 함수 적용 
                    st.session_state["forecast"] = get_forecast(st.session_state["processed"], horizon)
                    st.rerun()
                
                if st.session_state["forecast"] is not None:
                    f_mean = st.session_state["forecast"]["mean"]
                    st.metric("평균 예상 수요", f"{f_mean.mean():.2f}")

            with f_col2:
                if st.session_state["forecast"] is not None:
                    # 기존 plot_forecast_result 함수 적용 
                    fig_final = plot_forecast_result(st.session_state["processed"], st.session_state["forecast"])
                    st.plotly_chart(fig_final, use_container_width=True)
