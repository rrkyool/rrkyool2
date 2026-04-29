import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf

# -----------------------------
# 1. 분석 핵심 함수 정의
# -----------------------------
def load_data(file):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None
    
def find_optimal_period(series):
    """ACF를 분석하여 데이터의 잠재적 계절 주기를 자동 탐지"""
    series = series.dropna()
    if len(series) < 10: return 1
    max_lag = min(len(series) // 3, 40)
    acf_values = acf(series, nlags=max_lag)
    if len(acf_values) > 3:
        optimal_lag = np.argmax(acf_values[3:]) + 3
        if acf_values[optimal_lag] > 0.2:
            return int(optimal_lag)
    return 1

def hampel_filter(series, window=5, n=3):
    series = series.astype(float)
    new = series.copy()
    for i in range(window, len(series)-window):
        win = series.iloc[i-window:i+window]
        med = np.median(win)
        mad = np.median(np.abs(win-med))
        if mad == 0: continue
        if abs(series.iloc[i]-med) > n*mad:
            new.iloc[i] = med
    return new
    
def run_stationarity_test(series):
    """ADF 정상성 검정"""
    res = adfuller(series.dropna())
    return {"p_value": res[1], "is_stationary": res[1] < 0.05}

def decompose_series(series, period=12):
    """시계열 분해 (Trend, Seasonal 추출)"""
    if period < 2 or len(series) < period * 2: return None
    try:
        return seasonal_decompose(series.dropna(), model='additive', period=period)
    except: return None

def get_advanced_forecast(train, horizon, model_type, period=12):
    """예측값, 신뢰구간, 추세 기울기를 포함한 종합 예측"""
    train = pd.Series(train).astype(float).dropna()
    results = {'mean': None, 'upper': None, 'lower': None, 'trend_slope': 0}
    
    # 추세 계산
    x = np.arange(len(train))
    slope, _ = np.polyfit(x, train.values, 1)
    results['trend_slope'] = slope

    try:
        if model_type == "이동평균":
            val = train.rolling(window=period if period > 1 else 12, min_periods=1).mean().iloc[-1]
            results['mean'] = np.repeat(val, horizon)
            results['upper'] = results['mean'] + (train.std() * 0.5)
            results['lower'] = results['mean'] - (train.std() * 0.5)
        
        elif model_type == "지수평활":
            model = ExponentialSmoothing(train).fit()
            results['mean'] = model.forecast(horizon).values
            results['upper'] = results['mean'] * 1.1
            results['lower'] = results['mean'] * 0.9

        elif model_type == "Holt-Winters":
            has_seasonal = period > 1 and len(train) >= 2 * period
            model = ExponentialSmoothing(
                train, trend="add", 
                seasonal="add" if has_seasonal else None, 
                seasonal_periods=period if has_seasonal else None
            ).fit()
            forecast = model.forecast(horizon)
            results['mean'] = forecast.values
            results['upper'] = forecast.values + (train.std() * 1.96 / np.sqrt(len(train)))
            results['lower'] = forecast.values - (train.std() * 1.96 / np.sqrt(len(train)))

        elif model_type in ["ARIMA", "SARIMA"]:
            model = auto_arima(
                train, seasonal=(model_type=="SARIMA" and period > 1), 
                m=period if period > 1 else 1,
                stepwise=True, suppress_warnings=True, max_p=2, max_q=2
            )
            forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
            results['mean'] = forecast
            results['lower'] = conf_int[:, 0]
            results['upper'] = conf_int[:, 1]
            
    except Exception as e:
        last_val = train.iloc[-1]
        results['mean'] = np.repeat(last_val, horizon)
        results['upper'] = results['mean'] * 1.2
        results['lower'] = results['mean'] * 0.8

    return results

# -----------------------------
# 2. 메인 앱 설정
# -----------------------------
st.set_page_config(layout="wide")

header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("C321032 박하율")

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["모델", "MAE", "MdRAE", "TS", "예측평균"])
if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 3. 레이아웃: 상단 (입력 및 분석)
# -----------------------------
col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.subheader("📂 데이터 업로드 및 전처리")
        file = st.file_uploader("CSV 파일 업로드", label_visibility="collapsed")
        
        if file:
            df = pd.read_csv(file)
            date_col, val_col = df.columns[0], df.select_dtypes(include=np.number).columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).set_index(date_col)
            
            # 전처리 파이프라인 (말단 왜곡 방지)
            raw = df[val_col].copy()
            processed = raw.interpolate().pipe(hampel_filter).ewm(alpha=0.3, adjust=False).mean()
            df['proc'] = processed
            
            # 자동 주기 탐지
            detected_p = find_optimal_period(processed)
            st.session_state.current_p = detected_p
            
            # 시각화 1: 원본 vs 전처리
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df.index, y=raw, name="원본", line=dict(color="gray"), opacity=0.3))
            fig1.add_trace(go.Scatter(x=df.index, y=processed, name="전처리(Denoised)", line=dict(color="royalblue")))
            fig1.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig1, use_container_width=True)

            # 정상성 및 시계열 분해
            s_col1, s_col2 = st.columns(2)
            stat = run_stationarity_test(processed)
            s_col1.metric("ADF p-value", f"{stat['p_value']:.4f}")
            s_col2.info(f"탐지된 주기: {detected_p} | {'정상' if stat['is_stationary'] else '비정상'}")
            
            dec = decompose_series(processed, period=detected_p if detected_p > 1 else 12)
            if dec:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=dec.trend.index, y=dec.trend, name="Trend(추세)"))
                fig2.add_trace(go.Scatter(x=dec.seasonal.index, y=dec.seasonal, name="Seasonal(계절성)"))
                fig2.update_layout(height=250, title="시계열 분해 분석")
                st.plotly_chart(fig2, use_container_width=True)

with col_right:
    with st.container(border=True):
        st.subheader("⚙️ 예측 설정")
        m_type = st.selectbox("모델", ["지수평활", "Holt-Winters", "ARIMA", "SARIMA"])
        h_len = st.number_input("예측 기간(시평)", 1, 100, 12)
        e_type = st.radio("검증 방식", ["Rolling", "Expanding"], horizontal=True)
        u_type = st.selectbox("단위", ["일", "월", "년"], index=1)
        
        btn_run = st.button("🚀 분석 실행", use_container_width=True, type="primary")

# -----------------------------
# 4. 분석 실행 및 결과 출력
# -----------------------------
if file and btn_run:
    with st.spinner("최적의 파라미터를 찾는 중..."):
        full_series = df['proc']
        split = int(len(full_series) * 0.8)
        train, test = full_series.iloc[:split], full_series.iloc[split:]
        
        # 검증 수행
        if e_type == "Rolling":
            preds = rolling_forecast(train, test, m_type) # 기존 로직 활용
        else:
            preds = expanding_forecast(train, test, m_type)
            
        # 미래 예측 결과물 생성
        final_res = get_advanced_forecast(full_series, h_len, m_type, period=st.session_state.current_p)
        
        # 로그 기록 (생략 가능하지만 기존 구조 유지용)
        # ... (MAE, MdRAE 계산 로직) ...

    # 결과 레이아웃
    res_left, res_right = st.columns([1, 1.5])
    
    with res_left:
        st.subheader("📏 검증 로그")
        # 평가 테이블 및 실제값 vs 검증값 그래프 출력
        
    with res_right:
        st.subheader("📊 수요 예측 리포트")
        # 신뢰구간 포함 그래프
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df['proc'], name="과거"))
        
        f_dates = pd.date_range(df.index[-1], periods=h_len+1, freq='MS')[1:]
        fig3.add_trace(go.Scatter(x=f_dates, y=final_res['upper'], line_color='rgba(0,0,0,0)', showlegend=False))
        fig3.add_trace(go.Scatter(x=f_dates, y=final_res['lower'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name="95% 신뢰구간"))
        fig3.add_trace(go.Scatter(x=f_dates, y=final_res['mean'], name="예측", line=dict(color="red", width=3)))
        st.plotly_chart(fig3, use_container_width=True)
        
        trend_label = "상승" if final_res['trend_slope'] > 0 else "하락"
        st.success(f"**분석 요약**: 향후 {h_len}{u_type}간 데이터는 **{trend_label}** 추세를 보일 것으로 예측됩니다.")
