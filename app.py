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
# 1. 데이터 로드
# -----------------------------
def load_data(file):
    encodings = ["utf-8", "cp949", "euc-kr"]
    last_error = None
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            return df
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            raise ValueError(f"파일을 읽는 중 오류 발생: {e}")
    raise ValueError(f"지원되지 않는 인코딩입니다. 마지막 오류: {last_error}")

# -----------------------------
# 2. 전처리
# -----------------------------
def hampel_filter(series, window=5, n=3):
    series = series.astype(float)
    new = series.copy()
    k = 1.4826
    for i in range(len(series)):
        start = max(i - window, 0)
        end = min(i + window + 1, len(series))
        win = series.iloc[start:end]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        if mad == 0: continue
        threshold = n * k * mad
        if abs(series.iloc[i] - med) > threshold:
            new.iloc[i] = med
    return new

def denoise_series(series, window=11, poly=2):
    return pd.Series(savgol_filter(series, window_length=window, polyorder=poly), index=series.index)

def preprocess_series(series):
    s = series.astype(float)
    s = s.interpolate(limit_direction='both')
    s = hampel_filter(s, window=5, n=3)
    s = denoise_series(s)
    return s

def plot_preprocessing(raw, processed):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=raw, name="원본", opacity=0.5, line=dict(color="gray")))
    fig.add_trace(go.Scatter(y=processed, name="전처리", line=dict(color="blue")))
    return fig

# -----------------------------
# 3. 정상성
# -----------------------------
def run_stationarity_test(series):
    series = series.dropna()
    if len(series) < 10: raise ValueError("데이터가 너무 짧습니다.")
    stat, p_value, *_ = adfuller(series)
    return {"p_value": p_value, "is_stationary": p_value < 0.05}

def run_ljungbox_test(series, lags=12):
    series = series.dropna()
    result = acorr_ljungbox(series, lags=[lags], return_df=True)
    p_value = result['lb_pvalue'].iloc[0]
    return {"p_value": p_value, "has_autocorrelation": p_value < 0.05}

# -----------------------------
# 4. 분해
# -----------------------------
def decompose_series(series, period):
    series = series.dropna()
    if len(series) < period * 2: raise ValueError("데이터 길이가 주기 대비 너무 짧습니다.")
    return seasonal_decompose(series, model='additive', period=period)

def summarize_decomposition(result):
    trend_strength = result.trend.std() / result.observed.std()
    seasonal_strength = result.seasonal.std() / result.observed.std()
    return {"trend_strength": round(trend_strength, 2), "seasonal_strength": round(seasonal_strength, 2)}

def infer_frequency(index):
    if not isinstance(index, pd.DatetimeIndex): raise ValueError("DatetimeIndex 필요")
    diffs = index.to_series().diff().dropna()
    return diffs.mode()[0]

def get_time_span(index):
    return {"start": index.min(), "end": index.max(), "duration": index.max() - index.min()}

def suggest_periods(freq):
    if freq <= pd.Timedelta("1H"): return [24, 168]
    elif freq <= pd.Timedelta("1D"): return [7, 30]
    elif freq <= pd.Timedelta("7D"): return [4, 12]
    elif freq <= pd.Timedelta("31D"): return [12]
    return [1]

def analyze_time_index(index):
    freq = infer_frequency(index)
    span = get_time_span(index)
    return {"frequency": freq, "start": span["start"], "end": span["end"], "duration": span["duration"], "suggested_periods": suggest_periods(freq)}

# -----------------------------
# 5. 예측 및 평가 함수
# -----------------------------
def ma_forecast(train, horizon, window):
    val = train.rolling(window=window, min_periods=1).mean().iloc[-1]
    mean = np.repeat(val, horizon)
    return mean, mean - train.std(), mean + train.std()

def exp_forecast(train, horizon):
    model = ExponentialSmoothing(train).fit()
    forecast = model.forecast(horizon)
    return forecast.values, forecast.values - train.std(), forecast.values + train.std()

def hw_forecast(train, horizon, period):
    try: model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=period).fit()
    except: model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=period).fit()
    forecast = model.forecast(horizon)
    return forecast.values, forecast.values - train.std(), forecast.values + train.std()

def stl_forecast(train, horizon, period):
    model = STLForecast(train, ARIMA, model_kwargs={"order": (1,1,1)}, period=period)
    res = model.fit()
    forecast = res.forecast(horizon)
    resid_std = np.std(res.resid)
    return forecast.values, (forecast - 1.96*resid_std).values, (forecast + 1.96*resid_std).values

def arima_forecast(train, horizon, period, seasonal):
    model = auto_arima(train, seasonal=seasonal, m=period if seasonal else 1, stepwise=True, suppress_warnings=True, error_action="ignore")
    forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
    return forecast, conf_int[:,0], conf_int[:,1]

def get_forecast(train, horizon, model_type, period=12):
    train = pd.Series(train).astype(float).dropna()
    slope, _ = np.polyfit(np.arange(len(train)), train.values, 1)
    try:
        if model_type == "MA": mean, lower, upper = ma_forecast(train, horizon, period)
        elif model_type == "ES": mean, lower, upper = exp_forecast(train, horizon)
        elif model_type == "HW": mean, lower, upper = hw_forecast(train, horizon, period)
        elif model_type == "STL": mean, lower, upper = stl_forecast(train, horizon, period)
        elif model_type == "ARIMA": mean, lower, upper = arima_forecast(train, horizon, period, False)
        elif model_type == "SARIMA": mean, lower, upper = arima_forecast(train, horizon, period, True)
        else: raise ValueError("모델 타입 오류")
    except:
        mean = np.repeat(train.iloc[-1], horizon)
        lower, upper = mean * 0.9, mean * 1.1
    return {"mean": mean, "lower": lower, "upper": upper, "trend_slope": slope}

def rolling_forecast_fast(train, test, model_type):
    history = list(train)
    preds = []
    if model_type in ["ARIMA", "SARIMA"]:
        if model_type == "ARIMA": model = ARIMA(history, order=(1,1,1)).fit()
        else: model = SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        for t in range(len(test)):
            yhat = model.forecast(steps=1)[0]
            preds.append(yhat)
            model = model.append([test.iloc[t]], refit=False)
    else:
        for t in range(len(test)):
            yhat = get_forecast(pd.Series(history), 1, model_type)["mean"][0]
            preds.append(yhat)
            history.append(test.iloc[t])
    return np.array(preds)

def block_forecast(train, test, model_type, horizon=12):
    preds = []
    for i in range(0, len(test), horizon):
        hist = pd.concat([train, test[:i]])
        result = get_forecast(hist, horizon, model_type)
        preds.extend(result["mean"][:min(horizon, len(test)-i)])
    return np.array(preds)

def mae(y, yhat): return np.mean(np.abs(np.array(y) - np.array(yhat)))
def rmse(y, yhat): return np.sqrt(np.mean((np.array(y) - np.array(yhat))**2))
def mape(y, yhat): return np.mean(np.abs((np.array(y) - np.array(yhat)) / (np.array(y) + 1e-8))) * 100
def tracking_signal(y, yhat):
    err = np.array(y) - np.array(yhat)
    return np.sum(err) / (np.mean(np.abs(err)) + 1e-8)

def evaluate_metrics(y_true, y_pred, model_name, method):
    return {"모델": model_name, "평가방법": method, "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred), "TS": tracking_signal(y_true, y_pred), "예측평균": round(np.mean(y_pred), 2)}

def update_log(log_df, new_result):
    return pd.concat([log_df, pd.DataFrame([new_result])], ignore_index=True)

def summarize_forecast(forecast_result):
    return {"avg": np.mean(forecast_result["mean"]), "min": np.min(forecast_result["mean"]), "max": np.max(forecast_result["mean"]), "ci_low": np.min(forecast_result["lower"]), "ci_high": np.max(forecast_result["upper"])}

def aggregate_forecast(forecast_result, freq="D"):
    df = pd.DataFrame({"y": forecast_result["mean"]})
    if freq in ["W", "M"]: return df["y"].sum()
    return df["y"].mean()

# -----------------------------
# 메인 인터페이스 및 세션 관리
# -----------------------------
if "df" not in st.session_state: st.session_state["df"] = None
if "processed" not in st.session_state: st.session_state["processed"] = None
if "perf_log" not in st.session_state: st.session_state["perf_log"] = pd.DataFrame()
if "forecast_res" not in st.session_state: st.session_state["forecast_res"] = None
if "eval_preds" not in st.session_state: st.session_state["eval_preds"] = {}

st.set_page_config(layout="wide", page_title="시계열 수요 예측 대시보드")

with st.sidebar:
    st.title("📁 분석 설정")
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일 업로드", type=["csv"])
        if file:
            df = load_data(file)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            st.session_state["df"] = df
            if st.session_state["processed"] is None:
                st.session_state["processed"] = preprocess_series(df.iloc[:, 0])

    if st.session_state["processed"] is not None:
        with st.container(border=True):
            st.subheader("⚙️ 모델 상세 설정")
            model_type = st.selectbox("예측 모델 선택", ["MA", "ES", "HW", "STL", "ARIMA", "SARIMA"], key="sel_model")
            method = st.selectbox("평가 방식", ["Rolling", "Block"], key="sel_method") 
            horizon_val = st.number_input("예측 기간 숫자", min_value=1, value=7, key="in_horizon")
            time_unit = st.selectbox("시간 단위 선택", ["일", "주", "월", "년"], key="sel_unit")

            if st.button("수요 예측 실행", use_container_width=True, type="primary"):
                ps = st.session_state["processed"]
                data_freq = ps.index.inferred_freq if hasattr(ps.index, 'inferred_freq') and ps.index.inferred_freq else "D"
                
                actual_steps = int(horizon_val)

                split_idx = int(len(ps) * 0.8)
                train_p, test_p = ps.iloc[:split_idx], ps.iloc[split_idx:]
                
                # 평가용 예측 및 로그 업데이트
                y_pred = rolling_forecast_fast(train_p, test_p, model_type) if method == "Rolling" else block_forecast(train_p, test_p, model_type, actual_steps)
                st.session_state["perf_log"] = update_log(st.session_state["perf_log"], evaluate_metrics(test_p[:len(y_pred)], y_pred, model_type, method))
                st.session_state["eval_preds"][f"{model_type}_{method}"] = y_pred
                
                # [중요] 미래 예측 수행 및 날짜 생성
                time_info = analyze_time_index(ps.index)
                st.session_state["forecast_res"] = get_forecast(ps, actual_steps, model_type, time_info['suggested_periods'][0])
                freq_map = {
                    "일": "D",
                    "주": "W",
                    "월": "M",
                    "년": "Y"
                }
                
                selected_freq = freq_map.get(time_unit, data_freq)
                
                st.session_state["future_dates"] = pd.date_range(
                    start=ps.index[-1],
                    periods=actual_steps + 1,
                    freq=selected_freq
                )[1:]
                st.session_state["current_y_pred"] = y_pred

            if st.button("🗑️ 로그 초기화", use_container_width=True):
                st.session_state["perf_log"], st.session_state["eval_preds"], st.session_state["forecast_res"] = pd.DataFrame(), {}, None
                st.rerun()

# -----------------------------
# 리포트 메인 화면
# -----------------------------
st.title("📈 시계열 분석 Project1 수요 예측 리포트")
st.subheader("C321032 박하율")

if st.session_state["processed"] is not None:
    ps, raw = st.session_state["processed"], st.session_state["df"].iloc[:, 0]
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=450):
            st.subheader("전처리 결과 비교"); st.plotly_chart(plot_preprocessing(raw, ps), use_container_width=True)
    with col2:
        with st.container(border=True, height=450):
            st.subheader("정상성 및 통계 진단")
            adf, lb = run_stationarity_test(ps), run_ljungbox_test(ps)
            c1, c2 = st.columns(2)
            c1.metric("ADF p-value", f"{adf['p_value']:.4f}", "정상" if adf['is_stationary'] else "비정상")
            c2.metric("Ljung-Box p-value", f"{lb['p_value']:.4f}", "패턴 없음" if lb['p_value'] > 0.05 else "자기상관")
            time_info = analyze_time_index(ps.index)
            st.divider(); st.write(f"📅 기간: `{time_info['start'].date()}` ~ `{time_info['end'].date()}`")
            freq = time_info['frequency']

            # freq를 사람이 읽을 수 있게 변환
            if freq <= pd.Timedelta("1H"):
                freq_str = "시간 단위"
            elif freq <= pd.Timedelta("1D"):
                freq_str = "일 단위"
            elif freq <= pd.Timedelta("7D"):
                freq_str = "주 단위"
            elif freq <= pd.Timedelta("31D"):
                freq_str = "월 단위"
            else:
                freq_str = "연 단위"

            st.write(f"""
            🔄 추정 주기: `{time_info['suggested_periods'][0]}`
            📏 데이터 빈도: `{freq_str}`
            """)

    st.divider()
    with st.container(border=True):
        st.subheader("시계열 분해 결과(Decomposition)")
        current_period = analyze_time_index(ps.index)['suggested_periods'][0]
        decomp_res = decompose_series(ps, current_period)
        d_col1, d_col2 = st.columns([2, 1])
        with d_col1:
            fig_d = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("📈 Observed & Trend", "🍂 Seasonal & Residual"))
            fig_d.add_trace(go.Scatter(y=decomp_res.observed, name="Original", opacity=0.4), row=1, col=1)
            fig_d.add_trace(go.Scatter(y=decomp_res.trend, name="Trend", line=dict(width=3)), row=1, col=1)
            fig_d.add_trace(go.Scatter(y=decomp_res.seasonal, name="Seasonal"), row=2, col=1)
            fig_d.add_trace(go.Scatter(y=decomp_res.resid, name="Residual", mode='markers'), row=2, col=1)
            st.plotly_chart(fig_d, use_container_width=True)
        with d_col2:
            summary = summarize_decomposition(decomp_res)
            st.metric("📈 추세 강도", summary['trend_strength']); st.metric("🍂 계절성 강도", summary['seasonal_strength'])

    if st.session_state["forecast_res"] is not None:
        st.divider(); st.subheader("📑 최종 수요 예측 결과 및 분석 리포트")
        f_res, future_dates = st.session_state["forecast_res"], st.session_state.get("future_dates")
        
        # [수정] 날짜-예측값 길이 불일치 방어 코드
        if future_dates is None or len(future_dates) != len(f_res['mean']):
            freq = ps.index.inferred_freq if hasattr(ps.index, 'inferred_freq') else "D"
            future_dates = pd.date_range(start=ps.index[-1], periods=len(f_res['mean'])+1, freq=freq)[1:]

        r_col1, r_col2 = st.columns([1.5, 1])
        with r_col1:
            with st.container(border=True, height=600):
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=ps.index, y=ps.values, name="실제값"))
                fig_f.add_trace(go.Scatter(x=future_dates, y=f_res['mean'], name="예측치", line=dict(color="#ef553b", width=4)))
                fig_f.add_trace(go.Scatter(x=future_dates, y=f_res['upper'], line=dict(width=0), showlegend=False))
                fig_f.add_trace(go.Scatter(x=future_dates, y=f_res['lower'], fill='tonexty', fillcolor='rgba(239,85,59,0.1)', line=dict(width=0), name="95% 신뢰구간"))
                st.plotly_chart(fig_f, use_container_width=True)
        with r_col2:
            with st.container(border=True, height=600):
                summ = summarize_forecast(f_res)
                m1, m2, m3 = st.columns(3)
                m1.metric("평균", f"{summ['avg']:,.1f}"); m2.metric("최대", f"{summ['max']:,.1f}"); m3.metric("최소", f"{summ['min']:,.1f}")
                st.divider()
                u = st.session_state.get("sel_unit", "일")
                st.info(f"✔️ {u} 단위 환산: {aggregate_forecast(f_res, {'일':'D','주':'W','월':'M','년':'Y'}.get(u, 'D')):,.2f}")
                st.dataframe(pd.DataFrame({"날짜": future_dates, "예측값": f_res['mean'], "하한": f_res['lower'], "상한": f_res['upper']}), use_container_width=True, height=250)

    st.divider(); st.subheader("📏 성능 평가 및 모델 검증")
    e_col1, e_col2 = st.columns([1, 1.2])
    with e_col1:
        if not st.session_state["perf_log"].empty: st.table(st.session_state["perf_log"])
        else: st.warning("기록된 로그가 없습니다.")
    with e_col2:
        if st.session_state["eval_preds"]:
            test_p = ps.iloc[int(len(ps)*0.8):]
            fig_v = go.Figure(); fig_v.add_trace(go.Scatter(x=test_p.index, y=test_p.values, name="Actual", line=dict(color="green", dash='dot')))
            for label, p_val in st.session_state["eval_preds"].items():
                fig_v.add_trace(go.Scatter(x=test_p.index, y=p_val, name=f"Pred({label})"))
            fig_v.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_v, use_container_width=True)
