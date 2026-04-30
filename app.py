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
# [로직 개선] 1. Rolling Forecast (모델 설정 반영 버전)
# -----------------------------
def rolling_forecast_fast(train, test, model_type, period=12):
    history = list(train)
    preds = []
    
    # 모델 타입에 따라 다르게 작동하도록 수정
    for t in range(len(test)):
        # 매 스텝마다 현재까지의 history로 예측 (단, 속도를 위해 ARIMA/SARIMA는 차수 고정 사용 가능하나 
        # 여기서는 모델별 차별화를 위해 get_forecast의 로직을 활용)
        res = get_forecast(history, 1, model_type, period)
        yhat = res["mean"][0]
        preds.append(yhat)
        history.append(test.iloc[t])
        
    return np.array(preds)

# -----------------------------
# [기존 함수 유지/보정]
# -----------------------------
def load_data(file):
    encodings = ["utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    raise ValueError("파일 인코딩 오류")

def hampel_filter(series, window=5, n=3):
    series = series.astype(float)
    new = series.copy()
    k = 1.4826
    for i in range(len(series)):
        start, end = max(i - window, 0), min(i + window + 1, len(series))
        win = series.iloc[start:end]
        med, mad = np.median(win), np.median(np.abs(win - med))
        if mad != 0 and abs(series.iloc[i] - med) > (n * k * mad):
            new.iloc[i] = med
    return new

def denoise_series(series, window=11, poly=2):
    return pd.Series(savgol_filter(series, window_length=window, polyorder=poly), index=series.index)

def preprocess_series(series):
    s = series.astype(float).interpolate(limit_direction='both')
    s = hampel_filter(s)
    return denoise_series(s)

def run_stationarity_test(series):
    stat, p_value, *_ = adfuller(series.dropna())
    return {"p_value": p_value, "is_stationary": p_value < 0.05}

def run_ljungbox_test(series, lags=12):
    result = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    p_value = result['lb_pvalue'].iloc[0]
    return {"p_value": p_value, "has_autocorrelation": p_value < 0.05}

def decompose_series(series, period):
    return seasonal_decompose(series.dropna(), model='additive', period=period)

def summarize_decomposition(result):
    return {"trend_strength": round(result.trend.std() / result.observed.std(), 2),
            "seasonal_strength": round(result.seasonal.std() / result.observed.std(), 2)}

def analyze_time_index(index):
    diffs = index.to_series().diff().dropna()
    freq = diffs.mode()[0]
    if freq <= pd.Timedelta("1D"): p = [7, 30]
    elif freq <= pd.Timedelta("7D"): p = [4, 12]
    else: p = [12]
    return {"frequency": freq, "start": index.min(), "end": index.max(), "suggested_periods": p}

# -----------------------------
# [로직 개선] 2. 모델별 예측 함수 (핵심)
# -----------------------------
def get_forecast(train, horizon, model_type, period=12):
    train = pd.Series(train).astype(float).dropna()
    try:
        if "MA" in model_type:
            val = train.rolling(window=period, min_periods=1).mean().iloc[-1]
            mean = np.repeat(val, horizon)
        elif "ES" in model_type:
            mean = ExponentialSmoothing(train).fit().forecast(horizon).values
        elif "Holt" in model_type:
            mean = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=period).fit().forecast(horizon).values
        elif "STL" in model_type:
            mean = STLForecast(train, ARIMA, model_kwargs={"order": (1,1,1)}, period=period).fit().forecast(horizon).values
        elif "SARIMA" in model_type:
            # 실시간 파라미터 최적화 반영
            model = auto_arima(train, seasonal=True, m=period, stepwise=True, suppress_warnings=True)
            mean = model.predict(n_periods=horizon)
        elif "ARIMA" in model_type:
            model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
            mean = model.predict(n_periods=horizon)
        else:
            mean = np.repeat(train.iloc[-1], horizon)
        
        std = train.std()
        return {"mean": mean, "lower": mean - (1.96 * std), "upper": mean + (1.96 * std)}
    except:
        last = train.iloc[-1]
        return {"mean": np.repeat(last, horizon), "lower": np.repeat(last*0.8, horizon), "upper": np.repeat(last*1.2, horizon)}

def evaluate_metrics(y_true, y_pred, model_name, method):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    err = y_true - y_pred
    return {"모델": model_name, "평가방법": method,
            "MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err**2)),
            "MAPE": np.mean(np.abs(err / (y_true + 1e-8))) * 100,
            "TS": np.sum(err) / (np.mean(np.abs(err)) + 1e-8)}

# -----------------------------
# 3. UI 및 레이아웃 구성
# -----------------------------
st.set_page_config(layout="wide", page_title="시계열 수요 예측 리포트")

if "perf_log" not in st.session_state: st.session_state["perf_log"] = pd.DataFrame()
if "forecast_res" not in st.session_state: st.session_state["forecast_res"] = None
if "current_y_pred" not in st.session_state: st.session_state["current_y_pred"] = None

with st.sidebar:
    st.title("📁 분석 설정")
    file = st.file_uploader("CSV 업로드", type=["csv"])
    if file:
        df = load_data(file)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
        st.session_state["df"] = df
        st.session_state["processed"] = preprocess_series(df.iloc[:, 0])

    if "processed" in st.session_state:
        st.subheader("⚙️ 모델 상세 설정")
        model_type = st.selectbox("모델", ["MA", "ES", "Holt-Winter's", "STL", "ARIMA", "SARIMA"], key="sel_model")
        method = st.selectbox("평가 방식", ["Rolling", "Block"], key="sel_method")
        horizon = st.number_input("예측 기간", min_value=1, value=7, key="in_horizon")
        
        if st.button("수요 예측 실행", type="primary", use_container_width=True):
            ps = st.session_state["processed"]
            time_info = analyze_time_index(ps.index)
            period = time_info['suggested_periods'][0]
            
            # [수정] 평가 구간을 시평(horizon)에 맞게 유동적으로 설정
            split_idx = len(ps) - horizon if len(ps) > horizon else int(len(ps)*0.8)
            train_p, test_p = ps.iloc[:split_idx], ps.iloc[split_idx:]
            
            if method == "Rolling":
                y_pred = rolling_forecast_fast(train_p, test_p, model_type, period)
            else:
                y_pred = get_forecast(train_p, len(test_p), model_type, period)["mean"]
            
            st.session_state["current_y_pred"] = y_pred
            st.session_state["perf_log"] = pd.concat([st.session_state["perf_log"], 
                                                      pd.DataFrame([evaluate_metrics(test_p, y_pred, model_type, method)])], ignore_index=True)
            st.session_state["forecast_res"] = get_forecast(ps, horizon, model_type, period)

st.title("📈 시계열 수요 예측 리포트")
st.caption("C321032 박하율")

if "processed" in st.session_state:
    ps = st.session_state["processed"]
    
    # 1행: 전처리 및 정상성
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("전처리 결과")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state["df"].iloc[:,0], name="Original", opacity=0.3))
        fig.add_trace(go.Scatter(y=ps, name="Processed", line=dict(color="blue")))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("통계 진단")
        adf = run_stationarity_test(ps)
        st.metric("ADF (정상성)", f"{adf['p_value']:.4f}", "정상" if adf['is_stationary'] else "비정상")
        st.info(f"추정 주기: {analyze_time_index(ps.index)['suggested_periods'][0]}")

    st.divider()

    # 2행: 최종 예측 결과
    if st.session_state["forecast_res"]:
        st.subheader("📑 최종 수요 예측 및 상세 분석")
        res = st.session_state["forecast_res"]
        future_idx = pd.date_range(ps.index[-1], periods=len(res['mean'])+1, freq=ps.index.freq)[1:]
        
        c_res1, c_res2 = st.columns([2, 1])
        with c_res1:
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=ps.index, y=ps, name="Actual"))
            fig_f.add_trace(go.Scatter(x=future_idx, y=res['mean'], name="Forecast", line=dict(color="red")))
            fig_f.add_trace(go.Scatter(x=future_idx, y=res['upper'], line=dict(width=0), showlegend=False))
            fig_f.add_trace(go.Scatter(x=future_idx, y=res['lower'], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0)))
            st.plotly_chart(fig_f, use_container_width=True)
        with c_res2:
            st.write("**상세 예측값**")
            st.dataframe(pd.DataFrame({"예측": res['mean'], "하한": res['lower'], "상한": res['upper']}, index=future_idx))

    st.divider()

    # 3행: [수정 요청사항] 누적 로그와 검증 시각화 나란히 배치
    st.subheader("📏 성능 평가 및 모델 검증")
    log_col, chart_col = st.columns([1, 1.2])
    
    with log_col:
        st.write("**📊 누적 성능 지표**")
        if not st.session_state["perf_log"].empty:
            st.table(st.session_state["perf_log"])
        else:
            st.info("예측 실행 시 로그가 기록됩니다.")
            
    with chart_col:
        st.write("**🔍 모델 검증 (Actual vs Prediction)**")
        if st.session_state["current_y_pred"] is not None:
            y_pred = st.session_state["current_y_pred"]
            test_p = ps.iloc[-len(y_pred):]
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(y=test_p.values, name="Actual", line=dict(dash="dot")))
            fig_v.add_trace(go.Scatter(y=y_pred, name="Prediction", line=dict(color="orange")))
            fig_v.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_v, use_container_width=True)
