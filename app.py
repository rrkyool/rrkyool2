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
    fig.add_trace(go.Scatter(y=processed, name="전처리", line=dict(color="darkgreen")))
    
    # 사이즈 및 여백 조정
    fig.update_layout(
        height=265,  # 기존보다 작게 조정 (원하는 수치로 변경 가능)
        margin=dict(l=10, r=10, t=30, b=10),  # 상하좌우 여백 최소화 [cite: 96, 161]
        legend=dict(
            orientation="h",     # 범례를 가로로 배치 
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"    # 마우스 올렸을 때 정보 집약
    )
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
    
    actual_lags = min(lags, len(series) - 1)
    
    if actual_lags <= 0:
        return {"p_value": np.nan, "has_autocorrelation": False, "status": "데이터 부족"}

    try:
        # return_df=True 일 때 lags 값을 리스트로 명시
        result = acorr_ljungbox(series, lags=[actual_lags], return_df=True)
        p_value = result['lb_pvalue'].iloc[0]
        return {"p_value": p_value, "has_autocorrelation": p_value < 0.05, "status": "OK"}
    except:
        return {"p_value": np.nan, "has_autocorrelation": False, "status": "연산 오류"}

# -----------------------------
# 4. 분해
# -----------------------------
def decompose_series(series, period):
    series = series.dropna()
    if len(series) < period * 2: raise ValueError("데이터 길이가 주기 대비 너무 짧습니다.")
    return seasonal_decompose(series, model='additive', period=period)

def summarize_decomposition(result):
    # 1. 추세 강도 계산
    # 공식: max(0, 1 - Var(Resid) / Var(Trend + Resid))
    resid_var = np.nanvar(result.resid)
    trend_resid_var = np.nanvar(result.trend + result.resid)
    trend_strength = max(0, 1 - (resid_var / trend_resid_var))
    
    # 2. 계절성 강도 계산
    # 공식: max(0, 1 - Var(Resid) / Var(Seasonal + Resid))
    seasonal_resid_var = np.nanvar(result.seasonal + result.resid)
    seasonal_strength = max(0, 1 - (resid_var / seasonal_resid_var))
    
    return {
        "trend_strength": round(trend_strength, 2), 
        "seasonal_strength": round(seasonal_strength, 2)
    }
    
def infer_frequency(index):
    if not isinstance(index, pd.DatetimeIndex): raise ValueError("DatetimeIndex 필요")
    diffs = index.to_series().diff().dropna()
    return diffs.mode()[0]

def get_time_span(index):
    return {"start": index.min(), "end": index.max(), "duration": index.max() - index.min()}

def suggest_periods(freq):
    if freq is None: return [1]
    
    if freq <= pd.Timedelta("1H"): 
        return [168, 24]  # 주간(168시간)을 우선순위로 하여 추세를 매끄럽게 함
    elif freq <= pd.Timedelta("1D"): 
        return [30, 7]    # 월간(30일)을 우선순위로 설정
    elif freq <= pd.Timedelta("7D"): 
        return [13, 12, 4] # 연간(52주)을 우선순위로 설정하여 선형 추세 유도
    elif freq <= pd.Timedelta("31D"): 
        return [12, 4]    # 연간(12개월) 우선
    return [1]

def get_auto_period(freq, series_length):
    """
    데이터 빈도에 따라 선형적 추세를 유도할 수 있는 최적 주기를 자동 계산
    """
    if freq is None:
        return 2 # 최소 주기
    
    # 1. 빈도별 기본 마디 설정
    if freq <= pd.Timedelta("1H"): 
        base_period = 24  # 일간 패턴
        multiplier = 7    # -> 주간(168)으로 확장 시도
    elif freq <= pd.Timedelta("1D"): 
        base_period = 7   # 주간 패턴
        multiplier = 4    # -> 월간(28~30)으로 확장 시도
    elif freq <= pd.Timedelta("7D"): 
        base_period = 4   # 월간 패턴(주 단위 데이터 기준)
        multiplier = 12   # -> 연간(48~52)으로 확장 시도
    else:
        base_period = 12  # 연간 패턴
        multiplier = 1
        
    # 2. 임의로 곱해서 장기 주기 도출
    target_period = base_period * multiplier
    
    # 3. 안전 장치: 데이터 길이가 주기보다 최소 2배는 길어야 함
    if series_length < target_period * 2:
        # 데이터가 부족하면 기본 마디(base_period)만 사용
        return base_period if series_length >= base_period * 2 else 2
        
    return target_period

def get_freq_label(freq):
    """Timedelta 빈도를 사용자가 이해하기 쉬운 한글 텍스트로 변환"""
    if freq is None: return "판단 불가"
    days = freq.days
    hours = freq.components.hours
    
    if days == 7: return "7일(1주일) 단위"
    if days == 1: return "1일(Daily) 단위"
    if days >= 28 and days <= 31: return "1개월(Monthly) 단위"
    if hours == 1: return "1시간 단위"
    
    # 그 외의 경우
    if days > 0: return f"{days}일 단위"
    return f"{hours}시간 단위"

def get_period_unit_text(freq, period):
    """빈도와 주기를 결합하여 사람이 읽기 좋은 텍스트로 변환"""
    if freq is None: return f"{period}개 포인트"
    
    days = freq.days
    hours = freq.components.hours
    
    # 1. 시간 단위 데이터
    if hours == 1 or freq <= pd.Timedelta("1H"):
        if period == 24: return "24시간 (1일)"
        if period == 168: return "168시간 (1주일)"
        return f"{period}시간"
    
    # 2. 일 단위 데이터
    if days == 1:
        if period == 7: return "7일 (1주일)"
        if period == 30: return "30일 (약 1개월)"
        if period == 365: return "365일 (1년)"
        return f"{period}일"
    
    # 3. 주 단위 데이터 (사용자님 케이스)
    if days == 7:
        if period == 4: return "4주 (약 1개월)"
        if period == 12: return "12주 (약 1분기)"
        if period == 52: return "52주 (1년)"
        return f"{period}주"
    
    # 4. 월 단위 데이터
    if 28 <= days <= 31:
        if period == 3: return "3개월 (1분기)"
        if period == 12: return "12개월 (1년)"
        return f"{period}개월"
        
    return f"{period}개 마디"

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
    # 데이터 부족 시 계절성 해제 로직
    if len(train) < period * 2:
        seasonal = False
    
    if not seasonal:
        # 1. ARIMA (기존 성능 유지)
        max_p, max_q = 3, 3      # 원래의 넓은 검색 범위 유지
        d_val = None             # 최적 차분 자동 탐색
        approximation = False    # 정밀한 연산
    else:
        # 2. SARIMA (안정성 및 직선 방지 최적화)
        max_p, max_q = 1, 1      # 무한 로딩 방지를 위해 범위 제한
        d_val = 1                # 강제 1차 차분으로 수평선(Level) 현상 방지
        approximation = True     # 연산 속도 확보 (Oh No 화면 방지)

    try:
        model = auto_arima(train, 
                           seasonal=seasonal, 
                           m=period if seasonal else 1, 
                           d=d_val,               # SARIMA일 때 강제 차분 적용
                           stepwise=True,
                           approximation=approximation,
                           max_p=max_p, max_q=max_q, 
                           max_P=1, max_Q=1, 
                           start_p=1, start_q=1, 
                           suppress_warnings=True, 
                           error_action="ignore",
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        
        forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
        return forecast, conf_int[:,0], conf_int[:,1]
    except Exception as e:
        # 실패 시 최후의 수단
        st.error(f"모델 연산 오류: {e}")
        mean = np.repeat(train.iloc[-1], horizon)
        return mean, mean * 0.9, mean * 1.1

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

def rolling_forecast_fast(train, test, model_type, period=12):
    history = train.copy()
    preds = []
    
    if model_type in ["ARIMA", "SARIMA"]:
        try:
            # 초기 모델 피팅
            if model_type == "ARIMA":
                model_res = ARIMA(history, order=(1,1,1)).fit()
            else:
                # [수정] 주기가 1이면 계절성 없이 ARIMA로 동작하게 방어
                s_order = (1,1,1, period) if period > 1 and len(history) >= period*2 else (0,0,0,0)
                model_res = SARIMAX(history, 
                                    order=(1,1,1), 
                                    seasonal_order=s_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
            
            for t in range(len(test)):
                yhat = model_res.forecast(steps=1).iloc[0]
                preds.append(yhat)
                # [수정] 데이터 추가 시 빈도 정보가 유실되지 않도록 index와 함께 전달
                new_obs = pd.Series([test.iloc[t]], index=[test.index[t]])
                model_res = model_res.append(new_obs, refit=False)
        except Exception as e:
            st.warning(f"SARIMA 연산 중 오류 발생: {e}. 마지막 값으로 대체합니다.")
            preds = np.repeat(history.iloc[-1], len(test))
    else:
        # MA, ES 등은 빈도 정보에 덜 민감하므로 기존 로직 유지
        history_list = list(train)
        for t in range(len(test)):
            yhat = get_forecast(pd.Series(history_list), 1, model_type, period)["mean"][0]
            preds.append(yhat)
            history_list.append(test.iloc[t])
            
    return np.array(preds)

def block_forecast(train, test, model_type, horizon=12, period=12):
    preds = []
    for i in range(0, len(test), horizon):
        # 1. 이전까지의 데이터를 합쳐 학습 데이터 구성
        hist = pd.concat([train, test[:i]])
        
        # 2. [수정] get_forecast 호출 시 파라미터로 받은 period를 명시적으로 전달
        # 이 부분이 수정되어야 SARIMA 모델이 주기를 인식합니다.
        result = get_forecast(hist, horizon, model_type, period=period)
        
        # 3. 예측값 저장
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
    st.title("분석 설정")
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일 업로드", type=["csv"])
        if file:
            df = load_data(file)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.sort_values(df.columns[0]).set_index(df.columns[0])
            
            # 1. 빈도(Frequency) 설정
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)
            else:
                df = df.asfreq('W-SUN') 
            
            # 2. 결측치 처리 (ffill 대신 interpolate 적용)
            # limit_direction='both'를 사용해야 데이터 맨 앞이나 뒤의 결측치까지 완벽히 채워집니다.
            df = df.interpolate(method='linear', limit_direction='both') 
            
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
                
                h_val = int(horizon_val)
                if time_unit == "월":
                    actual_steps = h_val * 4
                elif time_unit == "주":
                    actual_steps = h_val
                elif time_unit == "년":
                    actual_steps = h_val * 52
                else: 
                    actual_steps = max(1, h_val // 7) if ("D" in data_freq or "W" in data_freq) else h_val
            
                split_idx = int(len(ps) * 0.8)
                train_p, test_p = ps.iloc[:split_idx], ps.iloc[split_idx:]
                
                time_info = analyze_time_index(ps.index)
                current_p = time_info['suggested_periods'][0]
                
                # [수정] Block 방식에서도 period를 전달하도록 보강
                if method == "Rolling":
                    y_pred = rolling_forecast_fast(train_p, test_p, model_type, period=current_p)
                else:
                    y_pred = block_forecast(train_p, test_p, model_type, actual_steps, period=current_p)
                    
                st.session_state["perf_log"] = update_log(st.session_state["perf_log"], evaluate_metrics(test_p[:len(y_pred)], y_pred, model_type, method))
                st.session_state["eval_preds"][f"{model_type}_{method}"] = y_pred
                
                # 미래 예측 수행
                st.session_state["forecast_res"] = get_forecast(ps, actual_steps, model_type, current_p)
                
                # 날짜 생성
                st.session_state["future_dates"] = pd.date_range(
                    start=ps.index[-1],
                    periods=actual_steps + 1,
                    freq=data_freq
                )[1:]
                
                st.session_state["current_y_pred"] = y_pred

            if st.button("로그 초기화", use_container_width=True):
                st.session_state["perf_log"], st.session_state["eval_preds"], st.session_state["forecast_res"] = pd.DataFrame(), {}, None
                st.rerun()

# -----------------------------
# 리포트 메인 화면
# -----------------------------
st.title("📈 시계열 분석 Project1 수요 예측")
st.subheader("C321032 박하율")
st.divider()

if st.session_state["processed"] is not None:
    ps, raw = st.session_state["processed"], st.session_state["df"].iloc[:, 0]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("전처리 결과")
        with st.container(border=True, height=500):
            st.plotly_chart(plot_preprocessing(raw, ps), use_container_width=True)
            st.divider()
            st.markdown("결측치 대체: Interpolate")
            st.markdown("이상치 대체: Hample Filter")
            st.markdown("Denoising: savgol_filter")
            
    with col2:
        st.subheader("정상성 및 통계 진단")
        with st.container(border=True, height=500):
            # 1. 테스트 실행
            adf = run_stationarity_test(ps)
            # 백색잡음 검정 (lag=1)
            wn_test = run_ljungbox_test(ps, lags=1) 
            # 자기상관 검정 (lag=k)
            current_period = analyze_time_index(ps.index)['suggested_periods'][0]
            ac_test = run_ljungbox_test(ps, lags=current_period)
            
            # 2. 3컬럼 배치
            c1, c2, c3 = st.columns(3)
            
            # [c1] ADF 검정
            c1.metric("ADF (단위근 검정)", f"{adf['p_value']:.4f}", 
                      "정상" if adf['is_stationary'] else "비정상")
            
            # [c2] 백색잡음 여부 (p-value가 0.05보다 크면 백색잡음)
            is_wn = wn_test['p_value'] > 0.05
            c2.metric("백색잡음 (lag=1)", f"{wn_test['p_value']:.4f}", 
                      "백색잡음" if is_wn else "패턴 존재")
            
            # [c3] 자기상관 여부 (p-value가 0.05보다 작으면 자기상관 있음)
            has_ac = ac_test['p_value'] < 0.05
            c3.metric(f"자기상관 (lag={current_period})", f"{ac_test['p_value']:.4f}", 
                      "모델개선 가능" if has_ac else "개선 불가")
            
            time_info = analyze_time_index(ps.index)
            freq_obj = time_info['frequency']
            suggested_p = time_info['suggested_periods'][0]
            
            # 주기 설명 텍스트 생성 (예: "4주 (약 1개월)")
            period_desc = get_period_unit_text(freq_obj, suggested_p)
            # 데이터 빈도 텍스트 (예: "7일(1주일) 단위")
            freq_label = get_freq_label(freq_obj)
            
            # ADF 검정 결과에 따른 차분 가이드
            if adf['is_stationary']:
                st.success("✅ **정상성 만족**")
            else:
                st.error("⚠️ **데이터에 추세나 계절성이 강해 모델의 예측력이 떨어질 수 있음. 차분 권장**")

            # Ljung-Box 검정 결과에 따른 모델링 적합성 가이드
            if is_wn:
                st.warning("⚠️ **백색잡음 주의. 모델 성능이 낮을 수 있음**")
            else:
                st.info("✅ **패턴 존재. 모형 개선 가능**")

            if has_ac:
                st.info("✅ **자기상관 존재. 모형 개선 가능**")
            else:
                st.warning("⚠️ **자기상관 없음. 주기적인 패턴 없음**")
                
            st.divider()

           
            st.write(f"📅 기간: `{time_info['start'].date()}` ~ `{time_info['end'].date()}`")
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
             ♾️ **데이터 기록 빈도**: `{freq_label}`  
            """)

    st.divider()

    st.subheader("시계열 분해 결과(Decomposition)")
    with st.container(border=True):
        try:
            # 1. 데이터 빈도 분석
            time_info = analyze_time_index(ps.index)
            freq = time_info['frequency']
            
            # 2. 장기 주기 자동 산출 (사용자 선택 없이 자동 적용)
            auto_period = get_auto_period(freq, len(ps))
            
            # 3. 시계열 분해 실행 (가법 모델 적용)
            decomp_res = seasonal_decompose(ps, model='additive', period=auto_period)
            summary = summarize_decomposition(decomp_res)
                    
            # 상단 지표 레이아웃
            m1, m2, m3 = st.columns([1, 1, 2])
            m1.metric("📈 추세 강도", f"{summary['trend_strength']:.2f}")
            m2.metric("🍂 계절성 강도", f"{summary['seasonal_strength']:.2f}")
            
            st.divider()

            # 4. 차트 생성 (기존 레이아웃 유지)
            fig_d = make_subplots(
                rows=1, cols=2, 
                shared_xaxes=True, 
                subplot_titles=("📈 Original & Trend", "🍂 Seasonal & Residual")
            )

            fig_d.add_trace(go.Scatter(y=decomp_res.observed, name="Original", opacity=0.4, line=dict(color="gray")), row=1, col=1)
            fig_d.add_trace(go.Scatter(y=decomp_res.trend, name="Trend", line=dict(color="#1f77b4", width=3)), row=1, col=1)
            fig_d.add_trace(go.Scatter(y=decomp_res.seasonal, name="Seasonal", line=dict(color="#2ca02c")), row=1, col=2)
            fig_d.add_trace(go.Scatter(y=decomp_res.resid, name="Residual", mode='markers', marker=dict(size=4, color="#ff7f0e")), row=1, col=2)

            fig_d.update_layout(
                height=370, 
                margin=dict(t=60, b=20, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
            )

            st.plotly_chart(fig_d, use_container_width=True, key="decomp_plot_final")
                
        except Exception as e:
            st.error(f"시계열 분해 중 오류 발생: {e}")

   # [4행] 최종 수요 예측 결과 및 분석 리포트
    if st.session_state.get("forecast_res") is not None:
        # 전처리 데이터 안전하게 가져오기 (ps가 정의되지 않았을 경우 대비)
        ps = st.session_state.get("processed")
        
        st.divider()
        st.subheader("📑 최종 수요 예측 결과 및 분석 리포트")
        
        f_res = st.session_state["forecast_res"]
        # 날짜 인덱스 복구 및 일관성 유지 [cite: 250]
        freq = ps.index.inferred_freq if hasattr(ps.index, 'inferred_freq') else "D"
        future_dates = pd.date_range(start=ps.index[-1], periods=len(f_res['mean'])+1, freq=freq)[1:]
    
        r_col1, r_col2 = st.columns([1.5, 1])
        with r_col1:
            with st.container(border=True, height=600):
                fig_f = go.Figure()
                split_idx = int(len(ps) * 0.8)
                
                # 1. 학습/검증 데이터 시각화
                fig_f.add_trace(go.Scatter(x=ps.index[:split_idx], y=ps.iloc[:split_idx], name="학습 데이터(Train)", line=dict(color="#1f77b4")))
                fig_f.add_trace(go.Scatter(x=ps.index[split_idx-1:], y=ps.iloc[split_idx-1:], name="검증 데이터(Test)", line=dict(color="#ff7f0e", dash="dot")))
                
                # --- 신뢰구간 스타일 정의 ---
                pink_line = 'rgba(255, 128, 128, 0.3)'  # 상/하한선 경계 (연한 분홍)
                pink_fill = 'rgba(255, 128, 128, 0.1)'  # 밴드 내부 채우기 (매우 투명한 분홍)
                
                # 2. 신뢰구간 상한선 (Upper Bound)
                fig_f.add_trace(go.Scatter(
                    x=future_dates, y=f_res['upper'], 
                    line=dict(color=pink_line, width=1), # 하한선과 동일하게 선 추가
                    showlegend=False, 
                    hoverinfo='skip'
                ))
                
                # 3. 신뢰구간 하한선 및 채우기 (Lower Bound & Fill)
                fig_f.add_trace(go.Scatter(
                    x=future_dates, y=f_res['lower'], 
                    line=dict(color=pink_line, width=1), # 상한선과 동일한 스타일
                    fill='tonexty', 
                    fillcolor=pink_fill, 
                    name="95% 신뢰구간"
                ))
                
                # 4. 미래 예측치 (밴드 위에 가장 선명하게 표시)
                fig_f.add_trace(go.Scatter(
                    x=future_dates, y=f_res['mean'], 
                    name="미래 예측치", 
                    line=dict(color="#ef553b", width=4), # 붉은색 계열로 대비
                    mode='lines+markers'
                ))
                
                # 레이아웃 설정
                fig_f.update_layout(
                    height=500, 
                    margin=dict(l=10, r=10, t=30, b=10), 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_f, use_container_width=True, key="main_forecast_plot")
    
        with r_col2:
            with st.container(border=True, height=600):
                summ = summarize_forecast(f_res)
                m1, m2, m3 = st.columns(3)
                m1.metric("평균", f"{summ['avg']:,.1f}"); m2.metric("최대", f"{summ['max']:,.1f}"); m3.metric("최소", f"{summ['min']:,.1f}")
                st.divider()
                
                u = st.session_state.get("sel_unit", "일")
                # 단위를 데이터프레임 빈도 코드로 변환
                unit_map = {'일': 'D', '주': 'W', '월': 'MS', '년': 'YS'} # MS는 월초, YS는 연초 기준
                target_freq = unit_map.get(u, 'D')
                
                st.write("📅 상세 예측 데이터")
            
                # [수정 포인트] 선택한 단위(u)에 맞춰 실제 예측된 개수(len(f_res['mean']))만큼 날짜 생성
                # 만약 월 단위 예측이라면, 마지막 날짜로부터 1개월씩 증가하는 인덱스를 생성합니다.
                corrected_future_dates = pd.date_range(
                    start=ps.index[-1], 
                    periods=len(f_res['mean']) + 1, 
                    freq=target_freq
                )[1:]
            
                # 생성된 날짜와 예측값을 매칭하여 출력
                st.dataframe(
                    pd.DataFrame({
                        "날짜": corrected_future_dates.strftime('%Y-%m-%d'), # 가독성을 위해 날짜 포맷팅
                        "예측값": f_res['mean'], 
                        "하한": f_res['lower'], 
                        "상한": f_res['upper']
                    }), 
                    use_container_width=True, 
                    height=350
                )
        
        # [5행] 성능 평가 및 모델 검증 (오류 수정 핵심 영역)
        st.divider()
        st.subheader("성능 평가 및 모델 검증")
        e_col1, e_col2 = st.columns([1, 1.2])
        
        with e_col1:
            with st.container(border=True):
                st.markdown("#### 성능 평가 결과")
                
                if not st.session_state["perf_log"].empty:
                    # 1. 전체 로그 데이터 표시
                    df_log = st.session_state["perf_log"]
                    if st.session_state.get("forecast_res") is not None:
                        current_forecast_mean = np.mean(st.session_state["forecast_res"]['mean'])
                        # 가장 마지막(최신) 행의 '예측평균' 컬럼 값을 업데이트
                        df_log.iloc[-1, df_log.columns.get_loc('예측평균')] = round(current_forecast_mean, 2)
                        
                    st.dataframe(df_log, use_container_width=True)
                    
                    # 2. 지표별 Best 모델 산출 로직
                    # MAE, RMSE, MAPE는 최소값(idxmin)
                    best_mae_idx = df_log['MAE'].idxmin()
                    best_rmse_idx = df_log['RMSE'].idxmin()
                    best_mape_idx = df_log['MAPE'].idxmin()
                    
                    # TS는 0에 가장 가까운 값 (절대값이 최소인 것)
                    best_ts_idx = df_log['TS'].abs().idxmin()
                    
                    # 3. 요약 텍스트 출력 (st.info 활용하여 가독성 높임)
                    # 모델명은 'Model' 또는 '모델명' 컬럼에 있다고 가정합니다.
                    model_col = 'Model' if 'Model' in df_log.columns else df_log.columns[0]
                    
                    best_summary = (
                        f"🏆 **Best 평가지표** \n"
                        f"▫️ **MAE:** {df_log.loc[best_mae_idx, model_col]} ({df_log.loc[best_mae_idx, 'MAE']:.2f}) | "
                        f"▫️ **RMSE:** {df_log.loc[best_rmse_idx, model_col]} ({df_log.loc[best_rmse_idx, 'RMSE']:.2f})  \n"
                        f"▫️ **MAPE:** {df_log.loc[best_mape_idx, model_col]} ({df_log.loc[best_mape_idx, 'MAPE']:.2f}%) | "
                        f"▫️ **TS(0 근접):** {df_log.loc[best_ts_idx, model_col]} ({df_log.loc[best_ts_idx, 'TS']:.2f})"
                    )
                    
                    st.info(best_summary)
                    
                else:
                    st.warning("기록된 로그가 없습니다. 먼저 예측을 실행해 주세요.")
                
        with e_col2:
            with st.container(border=True):
                st.markdown("#### 모델 검증 및 비교(Actual vs Prediction)")
    
                # eval_preds가 존재할 때만 시각화 실행
                eval_data = st.session_state.get("eval_preds")
                
                if eval_data:
                    # 1. 테스트 데이터 전체 범위 설정
                    split_idx = int(len(ps) * 0.8)
                    test_p = ps.iloc[split_idx:]
                    
                    fig_v = go.Figure()
                
                    # 2. 실제값 (테스트 데이터 전체 구간)
                    fig_v.add_trace(go.Scatter(
                        x=test_p.index, 
                        y=test_p.values, 
                        name="Actual (실제값)", 
                        line=dict(color="green", dash='dot', width=2)
                    ))
                    
                    # 3. 모델별 예측값 시각화
                    for label, p_val in eval_data.items():
                        # Rolling Forecast 결과(p_val)가 test_p와 길이가 같다고 가정
                        # 만약 길이가 다르더라도 인덱스를 매칭하여 전체 범위에 표시
                        fig_v.add_trace(go.Scatter(
                            x=test_p.index[-len(p_val):], # 예측 데이터의 길이에 맞춰 최신 구간부터 매칭
                            y=p_val, 
                            name=f"Pred({label})",
                            line=dict(width=2)
                        ))
                    
                    # 4. 레이아웃 설정
                    fig_v.update_layout(
                        height=400, 
                        margin=dict(l=10, r=10, t=10, b=10), 
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1
                        ),
                        hovermode="x unified" # 마우스 커서 위치의 모든 데이터 동시 확인
                    )
                    
                    st.plotly_chart(fig_v, use_container_width=True, key="validation_plot")
                else:
                    st.info("💡 예측 실행 후 검증 결과가 표시됩니다.")
