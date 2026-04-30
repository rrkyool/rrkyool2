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
    k = 1.4826  # scale factor
    for i in range(len(series)):
        start = max(i - window, 0)
        end = min(i + window + 1, len(series))
        win = series.iloc[start:end]
        med = np.median(win)
        mad = np.median(np.abs(win - med))
        if mad == 0:
            continue
        threshold = n * k * mad
        if abs(series.iloc[i] - med) > threshold:
            new.iloc[i] = med
    return new

def denoise_series(series, window=11, poly=2):
    return pd.Series(
        savgol_filter(series, window_length=window, polyorder=poly),
        index=series.index
    )
    
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
    if len(series) < 10:
        raise ValueError("데이터가 너무 짧아서 ADF 검정을 수행할 수 없습니다.")
    stat, p_value, *_ = adfuller(series)
    return {
        "p_value": p_value,
        "is_stationary": p_value < 0.05
    }

def run_ljungbox_test(series, lags=12):
    series = series.dropna()
    result = acorr_ljungbox(series, lags=[lags], return_df=True)
    p_value = result['lb_pvalue'].iloc[0]
    return {
        "p_value": p_value,
        "has_autocorrelation": p_value < 0.05
    }

# -----------------------------
# 4. 분해
# -----------------------------
def decompose_series(series, period):
    series = series.dropna()
    if len(series) < period * 2:
        raise ValueError("데이터 길이가 주기 대비 너무 짧습니다.")
    result = seasonal_decompose(series, model='additive', period=period)
    return result

def summarize_decomposition(result):
    trend_strength = result.trend.std() / result.observed.std()
    seasonal_strength = result.seasonal.std() / result.observed.std()
    return {
        "trend_strength": round(trend_strength, 2),
        "seasonal_strength": round(seasonal_strength, 2)
    }

def infer_frequency(index):
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex가 필요합니다.")
    if len(index) < 3:
        raise ValueError("데이터가 너무 적어 주기를 추정할 수 없습니다.")
    diffs = index.to_series().diff().dropna()
    most_common_diff = diffs.mode()[0]
    return most_common_diff

def get_time_span(index):
    start = index.min()
    end = index.max()
    duration = end - start
    return {
        "start": start,
        "end": end,
        "duration": duration
    }

def suggest_periods(freq):
    if freq <= pd.Timedelta("1H"):
        return [24, 168]
    elif freq <= pd.Timedelta("1D"):
        return [7, 30]
    elif freq <= pd.Timedelta("7D"):
        return [4, 12]
    elif freq <= pd.Timedelta("31D"):
        return [12]
    else:
        return [1]

def analyze_time_index(index):
    freq = infer_frequency(index)
    span = get_time_span(index)
    periods = suggest_periods(freq)
    return {
        "frequency": freq,
        "start": span["start"],
        "end": span["end"],
        "duration": span["duration"],
        "suggested_periods": periods
    }
    
# -----------------------------
# 5. 예측
# -----------------------------
def ma_forecast(train, horizon, window):
    val = train.rolling(window=window, min_periods=1).mean().iloc[-1]
    mean = np.repeat(val, horizon)
    std = train.std()
    return mean, mean - std, mean + std

def exp_forecast(train, horizon):
    model = ExponentialSmoothing(train).fit()
    forecast = model.forecast(horizon)
    std = train.std()
    return forecast.values, forecast.values - std, forecast.values + std

def hw_forecast(train, horizon, period):
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=period
    ).fit()
    forecast = model.forecast(horizon)
    std = train.std()
    return forecast.values, forecast.values - std, forecast.values + std

def stl_forecast(train, horizon, period):
    model = STLForecast(
        train,
        ARIMA,
        model_kwargs={"order": (1,1,1)},
        period=period
    )
    res = model.fit()
    forecast = res.forecast(horizon)
    resid_std = np.std(res.resid)
    upper = forecast + 1.96 * resid_std
    lower = forecast - 1.96 * resid_std
    return forecast.values, lower.values, upper.values

def arima_forecast(train, horizon, period, seasonal):
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=period if seasonal else 1,
        max_p=2, max_q=2, max_d=1,
        max_P=1, max_Q=1, max_D=1,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
    return forecast, conf_int[:,0], conf_int[:,1]

def get_forecast(train, horizon, model_type, period=12):
    train = pd.Series(train).astype(float).dropna()
    x = np.arange(len(train))
    slope, _ = np.polyfit(x, train.values, 1)
    try:
        if model_type == "MA":
            mean, lower, upper = ma_forecast(train, horizon, period)
        elif model_type == "ES":
            mean, lower, upper = exp_forecast(train, horizon)
        elif model_type == "HW":
            mean, lower, upper = hw_forecast(train, horizon, period)
        elif model_type == "STL":
            mean, lower, upper = stl_forecast(train, horizon, period)
        elif model_type == "ARIMA":
            mean, lower, upper = arima_forecast(train, horizon, period, False)
        elif model_type == "SARIMA":
            mean, lower, upper = arima_forecast(train, horizon, period, True)
        else:
            raise ValueError("지원하지 않는 모델")
    except Exception as e:
        last = train.iloc[-1]
        mean = np.repeat(last, horizon)
        lower = mean * 0.9
        upper = mean * 1.1
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "trend_slope": slope
    }

def rolling_forecast_fast(train, test, model_type):
    history = list(train)
    preds = []
    
    # ARIMA 계열은 속도를 위해 전용 클래스 사용, 나머지는 get_forecast 활용
    if model_type in ["ARIMA", "SARIMA"]:
        if model_type == "ARIMA":
            model = ARIMA(history, order=(1,1,1)).fit()
        else:
            model = SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        
        for t in range(len(test)):
            yhat = model.forecast(steps=1)[0]
            preds.append(yhat)
            model = model.append([test.iloc[t]], refit=False)
    else:
        # MA, ES, HW, STL 등 선택 시 모델 타입이 정확히 전달됨
        for t in range(len(test)):
            current = pd.Series(history)
            # 중요: get_forecast에 model_type을 그대로 전달하여 결과 차별화
            yhat = get_forecast(current, 1, model_type)["mean"][0]
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

def evaluate_forecast(train, test, model_type, horizon):
    preds = []
    for i in range(0, len(test), horizon):
        hist = pd.concat([train, test[:i]])
        result = get_forecast(hist, horizon, model_type)
        step_preds = result["mean"]
        preds.extend(step_preds[:min(horizon, len(test)-i)])
    return np.array(preds)

def mae(y, yhat): return np.mean(np.abs(np.array(y) - np.array(yhat)))
def rmse(y, yhat): return np.sqrt(np.mean((np.array(y) - np.array(yhat))**2))
def mape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    return np.mean(np.abs((y - yhat) / (y + 1e-8))) * 100
def tracking_signal(y, yhat):
    err = np.array(y) - np.array(yhat)
    mad = np.mean(np.abs(err))
    return np.sum(err) / (mad + 1e-8)

def evaluate_metrics(y_true, y_pred, model_name, method):
    return {
        "모델": model_name, "평가방법": method,
        "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred), "TS": tracking_signal(y_true, y_pred),
        "예측평균": round(np.mean(y_pred), 2)  # 추가된 부분
    }

def update_log(log_df, new_result):
    return pd.concat([log_df, pd.DataFrame([new_result])], ignore_index=True)

def plot_forecast_vs_actual(y_true, preds_dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, name="Actual", line=dict(dash="dot")))
    for name, pred in preds_dict.items():
        fig.add_trace(go.Scatter(y=pred, name=name))
    return fig

def summarize_forecast(forecast_result):
    mean = forecast_result["mean"]
    return {
        "avg": np.mean(mean), "min": np.min(mean), "max": np.max(mean),
        "ci_low": np.min(forecast_result["lower"]), "ci_high": np.max(forecast_result["upper"])
    }

def forecast_table(forecast_result, future_index):
    return pd.DataFrame({
        "날짜": future_index, "예측값": forecast_result["mean"],
        "하한(95%)": forecast_result["lower"], "상한(95%)": forecast_result["upper"]
    })

def aggregate_forecast(forecast_result, freq="D"):
    df = pd.DataFrame({"y": forecast_result["mean"]})
    if freq in ["W", "M"]: return df["y"].sum()
    return df["y"].mean()

def convert_horizon(value, unit, freq_str):
    """
    사용자가 입력한 숫자와 단위를 데이터의 실제 Step 수로 변환
    freq_str: 데이터의 빈도 (예: 'D', 'W', 'MS' 등)
    """
    # 기본 배수 설정 (일 기준)
    multipliers = {"일": 1, "주": 7, "월": 30, "년": 365}
    
    # 데이터 자체가 '월별(M)' 데이터라면 단위를 다르게 해석해야 함
    if freq_str and ('M' in freq_str or 'm' in freq_str):
        multipliers = {"일": 0.03, "주": 0.25, "월": 1, "년": 12}
    elif freq_str and ('W' in freq_str):
        multipliers = {"일": 0.14, "주": 1, "월": 4, "년": 52}

    total_steps = int(value * multipliers.get(unit, 1))
    return max(total_steps, 1) # 최소 1개는 예측

#########################################################################################

if "df" not in st.session_state: st.session_state["df"] = None
if "processed" not in st.session_state: st.session_state["processed"] = None
if "perf_log" not in st.session_state: st.session_state["perf_log"] = pd.DataFrame()
if "forecast_res" not in st.session_state: st.session_state["forecast_res"] = None
if "eval_preds" not in st.session_state: st.session_state["eval_preds"] = {}

st.set_page_config(layout="wide", page_title="C321032박하율_시계열 수요 예측")

with st.sidebar:
    st.title("📁 분석 설정")
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
        if file:
            df = load_data(file)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            st.session_state["df"] = df
            if st.session_state["processed"] is None:
                st.session_state["processed"] = preprocess_series(df.iloc[:, 0])
    
    # -----------------------------
    # 5. 예측 실행 섹션 (수정 완료)
    # -----------------------------
    if st.session_state["processed"] is not None:
        with st.container(border=True):
            st.subheader("⚙️ 모델 상세 설정")
            model_type = st.selectbox("예측 모델 선택", ["MA", "ES", "HW", "STL", "ARIMA", "SARIMA"], key="sel_model")
            method = st.selectbox("평가 방식", ["Rolling", "Block"], key="sel_method") 
            horizon_val = st.number_input("예측 기간 숫자", min_value=1, value=7, key="in_horizon")
            time_unit = st.selectbox("시간 단위 선택", ["일", "주", "월", "년"], key="sel_unit")
    
            # -----------------------------
            # [수정] 버튼 클릭 및 예측 실행 로직
            # -----------------------------
            if st.button("수요 예측 실행", use_container_width=True, type="primary"):
                ps = st.session_state["processed"]
                
                # 1. 실제 데이터 포인트(Step) 환산
                # 데이터 인덱스에서 실제 빈도 추출 (기본값 'D')
                data_freq = ps.index.inferred_freq if hasattr(ps.index, 'inferred_freq') and ps.index.inferred_freq else "D"
                
                # [핵심] 사용자가 입력한 숫자(horizon_val)와 단위(time_unit)를 데이터 개수(actual_steps)로 변환
                def get_actual_steps(val, unit, freq):
                    multipliers = {"일": 1, "주": 7, "월": 30, "년": 365}
                    # 만약 데이터 자체가 월간 데이터라면 단위를 1로 고정
                    if 'M' in freq or 'm' in freq:
                        multipliers = {"일": 1, "주": 1, "월": 1, "년": 12} 
                    return int(val * multipliers.get(unit, 1))
            
                actual_steps = get_actual_steps(horizon_val, time_unit, data_freq)
                
                split_idx = int(len(ps) * 0.8)
                train_p, test_p = ps.iloc[:split_idx], ps.iloc[split_idx:]
                
                # 2. 성능 평가용 예측 (Test 데이터 길이에 맞춤)
                if method == "Rolling":
                    y_pred = rolling_forecast_fast(train_p, test_p, model_type)
                else:
                    y_pred = block_forecast(train_p, test_p, model_type, actual_steps)
                
                # 3. 로그 및 누적 시각화 데이터 저장
                metrics = evaluate_metrics(test_p[:len(y_pred)], y_pred, model_type, method)
                st.session_state["perf_log"] = update_log(st.session_state["perf_log"], metrics)
                
                if "eval_preds" not in st.session_state: st.session_state["eval_preds"] = {}
                st.session_state["eval_preds"][f"{model_type}_{method}"] = y_pred
                
                # 4. [중요] 미래 예측 (환산된 actual_steps 적용)
                time_info = analyze_time_index(ps.index)
                forecast_res = get_forecast(ps, actual_steps, model_type, time_info['suggested_periods'][0])
                
                # 5. 미래 날짜 인덱스 생성 (예측 개수와 날짜 개수 일치)
                # 데이터의 빈도(freq)를 그대로 사용하여 날짜 생성
                future_dates = pd.date_range(
                    start=ps.index[-1], 
                    periods=actual_steps + 1, 
                    freq=data_freq
                )[1:]
                
                # 결과 저장
                st.session_state["forecast_res"] = forecast_res
                st.session_state["future_dates"] = future_dates
                st.session_state["current_y_pred"] = y_pred
            
                #st.success(f"✅ 단위 변환 완료: {time_unit} 단위를 반영하여 미래 {actual_steps}포인트를 예측합니다.")
    
            if st.button("🗑️ 로그 초기화", use_container_width=True):
                st.session_state["perf_log"] = pd.DataFrame()
                st.session_state["eval_preds"] = {} 
                st.session_state["forecast_res"] = None
                st.rerun()
                
st.title("📈 시계열 분석 Project1 수요 예측 리포트")
st.subheader("C321032 박하율")

if st.session_state["processed"] is not None:
    ps = st.session_state["processed"]
    raw = st.session_state["df"].iloc[:, 0]

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=450):
            st.subheader("전처리 결과 비교")
            st.plotly_chart(plot_preprocessing(raw, ps), use_container_width=True)
    with col2:
        with st.container(border=True, height=450):
            st.subheader("정상성 및 통계 진단")
            adf, lb = run_stationarity_test(ps), run_ljungbox_test(ps)
            c_stat1, c_stat2 = st.columns(2)
            c_stat1.metric("ADF p-value", f"{adf['p_value']:.4f}", "정상" if adf['is_stationary'] else "비정상")
            c_stat2.metric("Ljung-Box p-value", f"{lb['p_value']:.4f}", "패턴 없음" if lb['p_value'] > 0.05 else "자기상관 존재")
            time_info = analyze_time_index(ps.index)
            st.divider()
            st.write(f"📅 **기간:** `{time_info['start'].date()}` ~ `{time_info['end'].date()}`") 
            st.write(f"🔄 **추정 주기:** `{time_info['suggested_periods'][0]}`") 

    st.divider()

    with st.container(border=True):
        st.subheader("시계열 분해 결과(Decomposition)")
        time_info = analyze_time_index(ps.index)
        current_period = time_info['suggested_periods'][0]
        decomp_res = decompose_series(ps, current_period)
        decomp_col1, decomp_col2 = st.columns([2, 1])
        with decomp_col1:
            fig_decomp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("📈 Observed & Trend", "🍂 Seasonal & Residual"))
            fig_decomp.add_trace(go.Scatter(y=decomp_res.observed, name="Original", opacity=0.4, line=dict(color="gray")), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(y=decomp_res.trend, name="Trend", line=dict(color="#1f77b4", width=3)), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(y=decomp_res.seasonal, name="Seasonal", line=dict(color="#2ca02c")), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(y=decomp_res.resid, name="Residual", mode='markers', marker=dict(size=4, color="#ff7f0e")), row=2, col=1)
            fig_decomp.update_layout(height=520, margin=dict(t=40, b=20))
            st.plotly_chart(fig_decomp, use_container_width=True)
        with decomp_col2:
            st.write("시계열 분해 리포트")
            summary = summarize_decomposition(decomp_res)
            st.metric("📈 추세 강도", f"{summary['trend_strength']:.2f}")
            st.metric("🍂 계절성 강도", f"{summary['seasonal_strength']:.2f}")
            st.info(f"분석 주기: {current_period}")

    if st.session_state["forecast_res"] is not None:
        st.divider()
        st.subheader("📑 최종 수요 예측 결과 및 분석 리포트")
        res_row_col1, res_row_col2 = st.columns([1.5, 1])
        f_res = st.session_state["forecast_res"]
        ps = st.session_state["processed"]
        
        future_dates = st.session_state.get("future_dates")
        with res_row_col1:
            with st.container(border=True, height=600):
                fig_all = go.Figure()
                fig_all.add_trace(go.Scatter(x=ps.index, y=ps.values, name="과거 실제값"))
                fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['mean'], name="미래 예측치", line=dict(color="#ef553b", width=4)))
                fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['upper'], line=dict(width=0), showlegend=False))
                fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['lower'], fill='tonexty', fillcolor='rgba(239,85,59,0.1)', line=dict(width=0), name="신뢰구간"))
                st.plotly_chart(fig_all, use_container_width=True)
         with res_row_col2:
            with st.container(border=True, height=600):
                summary = summarize_forecast(f_res)
                m1, m2, m3 = st.columns(3)
                m1.metric("평균 예측치", f"{summary['avg']:,.1f}")
                m2.metric("최대 수요", f"{summary['max']:,.1f}")
                m3.metric("최소 수요", f"{summary['min']:,.1f}")
                st.divider()
                selected_unit = st.session_state.get("sel_unit", "일")
                agg_val = aggregate_forecast(f_res, freq={"일":"D","주":"W","월":"M","년":"Y"}.get(selected_unit, "D"))
                st.info(f"✔️ {selected_unit} 단위 환산: {agg_val:,.2f}")
                st.dataframe(forecast_table(f_res, future_dates), use_container_width=True, height=250)

            
    st.divider()
    st.subheader("📏 성능 평가 결과 및 모델 검증")
    
    eval_col1, eval_col2 = st.columns([1, 1.2]) # 그래프를 조금 더 넓게 배치
    
    with eval_col1:
        st.write("**📊 누적 성능 평가 로그 (History Log)**")
        if not st.session_state["perf_log"].empty:
            # 최신 로그가 위로 오게 하려면 .iloc[::-1] 사용 가능
            st.table(st.session_state["perf_log"])
            st.info("💡 MAE·RMSE(낮음 우수), MAPE(오차율 %), TS(±4 정상 범위), 예측평균(단위당)")
        else:
            st.warning("기록된 로그가 없습니다.")
    
    
    with eval_col2:
        st.write("**🔍 모델 검증 데이터 비교 (Actual vs Prediction)**")
        if "eval_preds" in st.session_state and st.session_state["eval_preds"]:
            test_p = ps.iloc[int(len(ps)*0.8):]
            fig_val = go.Figure()
            
            # 실제값은 하나만 (초록 점선)
            fig_val.add_trace(go.Scatter(x=test_p.index, y=test_p.values, name="Actual", line=dict(color="green", dash='dot')))
            
            # [누적] 저장된 모든 예측 모델의 트레이스를 추가
            for label, pred_values in st.session_state["eval_preds"].items():
                fig_val.add_trace(go.Scatter(x=test_p.index, y=pred_values, name=f"Pred({label})"))
                
            fig_val.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_val, use_container_width=True)
