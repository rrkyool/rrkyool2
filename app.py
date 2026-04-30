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
            # 인코딩 문제가 아닌 경우 바로 실패 처리
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
    s = denoise_series(s)  # Savitzky-Golay 추천
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

def plot_decomposition(result):
    # 2행 1열 구조로 변경 (상단: 원본+추세, 하단: 계절성+잔차)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("📈 원본 데이터 및 추세 (Observed & Trend)", "🍂 계절성 및 잔차 (Seasonal & Residual)")
    )

    # 1. 상단: 원본 데이터(회색) + 추세선(파란색)
    fig.add_trace(go.Scatter(y=result.observed, name="Original", line=dict(color="gray", width=1), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(y=result.trend, name="Trend", line=dict(color="#1f77b4", width=2)), row=1, col=1)

    # 2. 하단: 계절성(녹색) + 잔차(주황색/점선)
    fig.add_trace(go.Scatter(y=result.seasonal, name="Seasonal", line=dict(color="#2ca02c", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(y=result.resid, name="Residual", line=dict(color="#ff7f0e", width=1, dash="dot")), row=2, col=1)

    fig.update_layout(
        height=450,  # 컨테이너 높이에 맞춰 최적화
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def summarize_decomposition(result):
    trend_strength = result.trend.std() / result.observed.std()
    seasonal_strength = result.seasonal.std() / result.observed.std()

    return {
        "trend_strength": round(trend_strength, 2),
        "seasonal_strength": round(seasonal_strength, 2)
    }

#시간 간격 계산
def infer_frequency(index):
    """DatetimeIndex 기준 데이터 간격 추정"""
    
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex가 필요합니다.")

    if len(index) < 3:
        raise ValueError("데이터가 너무 적어 주기를 추정할 수 없습니다.")

    # 간격 계산
    diffs = index.to_series().diff().dropna()
    most_common_diff = diffs.mode()[0]

    return most_common_diff

def get_time_span(index):
    """전체 기간 계산"""
    
    start = index.min()
    end = index.max()
    duration = end - start

    return {
        "start": start,
        "end": end,
        "duration": duration
    }

def suggest_periods(freq):
    """데이터 간격 기반 계절성 주기 후보 제안"""

    # pandas Timedelta 기준
    if freq <= pd.Timedelta("1H"):
        return [24, 168]  # 하루, 일주일

    elif freq <= pd.Timedelta("1D"):
        return [7, 30]  # 주간, 월간

    elif freq <= pd.Timedelta("7D"):
        return [4, 12]  # 월간, 연간

    elif freq <= pd.Timedelta("31D"):
        return [12]  # 연간

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

from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA

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

from pmdarima import auto_arima

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

    # 추세
    x = np.arange(len(train))
    slope, _ = np.polyfit(x, train.values, 1)

    try:
        if model_type == "MA":
            mean, lower, upper = ma_forecast(train, horizon, period)

        elif model_type == "ES":
            mean, lower, upper = exp_forecast(train, horizon)

        elif model_type == "HW":
            mean, lower, upper = hw_forecast(train, horizon, period)

        elif model_type == "STL":   # ⭐ 추가
            mean, lower, upper = stl_forecast(train, horizon, period)

        elif model_type == "ARIMA":
            mean, lower, upper = arima_forecast(train, horizon, period, False)

        elif model_type == "SARIMA":
            mean, lower, upper = arima_forecast(train, horizon, period, True)

        else:
            raise ValueError("지원하지 않는 모델")

    except Exception as e:
        # fallback (단순하지만 명확)
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

#rolling
def rolling_forecast_fast(train, test, model_type):
    history = list(train)
    preds = []

    if model_type in ["ARIMA", "SARIMA"]:
        # 최초 1회만 학습
        if model_type == "ARIMA":
            model = ARIMA(history, order=(1,1,1)).fit()
        else:
            model = SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)

        for t in range(len(test)):
            # 예측
            yhat = model.forecast(steps=1)[0]
            preds.append(yhat)

            # 업데이트 (재학습 X)
            model = model.append([test.iloc[t]], refit=False)

    else:
        # 비모수 모델만 rolling 유지
        for t in range(len(test)):
            current = history
            yhat = get_forecast(current, 1, model_type)["mean"][0]

            preds.append(yhat)
            history.append(test.iloc[t])

    return np.array(preds)

#block forecasting
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

#성능평가지표
def mae(y, yhat):
    return np.mean(np.abs(np.array(y) - np.array(yhat)))


def rmse(y, yhat):
    return np.sqrt(np.mean((np.array(y) - np.array(yhat))**2))


def mape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    return np.mean(np.abs((y - yhat) / (y + 1e-8))) * 100


def tracking_signal(y, yhat):
    err = np.array(y) - np.array(yhat)
    mad = np.mean(np.abs(err))
    return np.sum(err) / (mad + 1e-8)

def evaluate_metrics(y_true, y_pred, model_name, method):
    return {
        "모델": model_name,
        "평가방법": method,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "TS": tracking_signal(y_true, y_pred)
    }

def update_log(log_df, new_result):
    return pd.concat([log_df, pd.DataFrame([new_result])], ignore_index=True)

def plot_forecast_vs_actual(y_true, preds_dict):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_true,
        name="Actual",
        line=dict(dash="dot")
    ))

    for name, pred in preds_dict.items():
        fig.add_trace(go.Scatter(
            y=pred,
            name=name
        ))

    return fig

#demand forecasting
def plot_forecast_result(train, forecast_result, index):
    mean = forecast_result["mean"]
    lower = forecast_result["lower"]
    upper = forecast_result["upper"]

    fig = go.Figure()

    # 과거 데이터
    fig.add_trace(go.Scatter(
        y=train,
        name="실제값",
        line=dict(color="blue")
    ))

    # 예측
    future_index = pd.date_range(start=index[-1], periods=len(mean)+1, freq=index.freq)[1:]

    fig.add_trace(go.Scatter(
        x=future_index,
        y=mean,
        name="예측값",
        line=dict(color="red")
    ))

    # 신뢰구간 (밴드)
    fig.add_trace(go.Scatter(
        x=future_index,
        y=upper,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_index,
        y=lower,
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.15)',
        line=dict(width=0),
        name="신뢰구간"
    ))

    return fig

def summarize_forecast(forecast_result):
    mean = forecast_result["mean"]
    lower = forecast_result["lower"]
    upper = forecast_result["upper"]

    return {
        "avg": np.mean(mean),
        "min": np.min(mean),
        "max": np.max(mean),
        "ci_low": np.min(lower),
        "ci_high": np.max(upper)
    }

def forecast_table(forecast_result, future_index):
    df = pd.DataFrame({
        "날짜": future_index,
        "예측값": forecast_result["mean"],
        "하한(95%)": forecast_result["lower"],
        "상한(95%)": forecast_result["upper"]
    })
    return df

def aggregate_forecast(forecast_result, freq="D"):
    df = pd.DataFrame({
        "y": forecast_result["mean"]
    })

    if freq == "W":
        return df["y"].sum()
    elif freq == "M":
        return df["y"].sum()
    else:
        return df["y"].mean()

#########################################################################################

# 0. 세션 상태 초기화
if "df" not in st.session_state: st.session_state["df"] = None
if "processed" not in st.session_state: st.session_state["processed"] = None
if "perf_log" not in st.session_state: st.session_state["perf_log"] = pd.DataFrame()
if "forecast_res" not in st.session_state: st.session_state["forecast_res"] = None

st.set_page_config(layout="wide", page_title="C321032박하율_시계열 수요 예측")

# -----------------------------
# 1. 사이드바 (업로드 + 모델 설정)
# -----------------------------
with st.sidebar:
    st.title("📁 분석 설정")
    
    # 1-1. 데이터 업로드 컨테이너
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
        if file:
            df = load_data(file)
            # 날짜 컬럼 자동 인식 및 인덱스 설정
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            st.session_state["df"] = df
            
            # [자동화] 업로드 즉시 첫 번째 컬럼 전처리 수행
            if st.session_state["processed"] is None:
                with st.spinner("데이터 분석 중..."):
                    target_series = df.iloc[:, 0]
                    st.session_state["processed"] = preprocess_series(target_series)
                st.success("데이터 로드 및 전처리 완료!")

    # 1-2. 모델 설정 컨테이너 (사이드바 하단 배치)
    if st.session_state["processed"] is not None:
        with st.container(border=True):
            st.subheader("⚙️ 모델 상세 설정")
            # 4가지 옵션 일렬(세로) 배치
            model_type = st.selectbox("예측 모델 선택", ["SARIMA", "ARIMA", "STL", "HW", "ES", "MA"], key="sel_model")
            method = st.selectbox("평가 방식", ["Rolling", "Block"], key="sel_method")
            horizon = st.number_input("예측 기간(시평)", min_value=1, value=7, key="in_horizon")
            time_unit = st.selectbox("시간 단위 표시", ["일", "주", "월", "년"], key="sel_unit")
            
            st.divider()
            
            # 예측 실행 버튼
            if st.button("🚀 수요 예측 실행", use_container_width=True, type="primary"):
                ps = st.session_state["processed"]
                split_idx = int(len(ps) * 0.8)
                train_p, test_p = ps.iloc[:split_idx], ps.iloc[split_idx:]
                
                # 기존 함수 활용 성능 평가 및 예측
                y_pred = evaluate_forecast(train_p, test_p, model_type, horizon)
                metrics = evaluate_metrics(test_p[:len(y_pred)], y_pred, model_type, method)
                
                # 기록 업데이트
                st.session_state["perf_log"] = update_log(st.session_state["perf_log"], metrics)
                
                # 주기 분석 후 최종 예측
                time_info = analyze_time_index(ps.index)
                st.session_state["forecast_res"] = get_forecast(ps, horizon, model_type, time_info['suggested_periods'][0])
                st.session_state["current_y_pred"] = y_pred # 검증 시각화용
                st.toast(f"{model_type} 모델 예측 완료!")

            if st.button("🗑️ 로그 초기화", use_container_width=True):
                st.session_state["perf_log"] = pd.DataFrame()
                st.rerun()

# -----------------------------
# 2. 메인 화면 (분석 리포트)
# -----------------------------
st.title("📈 시계열 분석 Project1 수요 예측 리포트")
st.subheader("C321032 박하율")

if st.session_state["processed"] is not None:
    ps = st.session_state["processed"]
    raw = st.session_state["df"].iloc[:, 0]

    # [1행] 전처리 결과 비교 & 정상성 검정 (1:1 배치)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=450):
            st.subheader("####전처리 결과 비교")
            st.plotly_chart(plot_preprocessing(raw, ps), use_container_width=True)
    
    with col2:
        with st.container(border=True, height=450):
            st.subheader("####정상성 및 통계 진단")
            adf = run_stationarity_test(ps)
            lb = run_ljungbox_test(ps)
            
            c_stat1, c_stat2 = st.columns(2)
            c_stat1.metric("ADF p-value (정상성)", f"{adf['p_value']:.4f}", 
                           delta="정상" if adf['is_stationary'] else "비정상")
            
            is_white_noise = lb['p_value'] > 0.05
            c_stat2.metric("Ljung-Box p-value (백색잡음)", f"{lb['p_value']:.4f}", 
                           delta="패턴 없음" if is_white_noise else "자기상관 존재")
            
            if is_white_noise:
                st.warning("⚠️ 데이터가 무작위 노이즈(백색잡음)에 가깝습니다.")
            else:
                st.info("✅ 모델링에 적합한 패턴이 존재합니다.")
            
            st.write("**데이터 기술 통계**")
            st.dataframe(ps.describe().to_frame().T, use_container_width=True)

    st.divider()

    # [2행] 시계열 분해 (원본+추세 / 계절성+잔차 2단 구성)
    with st.container(border=True):
        st.subheader("####시계열 분해 결과(Decomposition)")
        time_info = analyze_time_index(ps.index)
        decomp_res = decompose_series(ps, time_info['suggested_periods'][0])
        
        # 수정된 2x1 레이아웃 적용
        fig_decomp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                                   subplot_titles=("📈 원본 및 추세 (Observed & Trend)", "🍂 계절성 및 잔차 (Seasonal & Residual)"))
        
        fig_decomp.add_trace(go.Scatter(y=decomp_res.observed, name="Original", opacity=0.4, line=dict(color="gray")), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(y=decomp_res.trend, name="Trend", line=dict(color="#1f77b4", width=3)), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(y=decomp_res.seasonal, name="Seasonal", line=dict(color="#2ca02c")), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(y=decomp_res.resid, name="Residual", mode='markers', marker=dict(size=4, color="#ff7f0e")), row=2, col=1)
        
        fig_decomp.update_layout(height=550, margin=dict(t=40, b=20))
        st.plotly_chart(fig_decomp, use_container_width=True)

    st.divider()

    # [3행] 수요 예측 결과 & 성능 평가 로그
   # -----------------------------
    # 4️⃣ 최종 수요 예측 결과 및 분석 리포트 (행 전체 사용)
    # -----------------------------
    if st.session_state["forecast_res"] is not None:
        st.divider()
        st.subheader("#### 최종 수요 예측 결과 및 분석 리포트")
        
        # [병렬 배치] 왼쪽: 메인 차트(1.5) | 오른쪽: 분석 리포트(1)
        res_row_col1, res_row_col2 = st.columns([1.5, 1])
        f_res = st.session_state["forecast_res"]
        ps = st.session_state["processed"]
        # 미래 날짜 생성
        future_dates = pd.date_range(start=ps.index[-1], periods=len(f_res['mean'])+1, freq=ps.index.freq)[1:]
    
        with res_row_col1:
            with st.container(border=True, height=600):
                st.write("**📈 수요 예측 시각화**")
                fig_all = go.Figure()
                # 과거 데이터 (파란색) [cite: 200, 265]
                fig_all.add_trace(go.Scatter(x=ps.index, y=ps.values, name="과거 실제값", line=dict(color="#1f77b4")))
                # 미래 예측 (빨간색, 굵게) [cite: 200, 212, 265]
                fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['mean'], name="미래 예측치", 
                                             line=dict(color="#ef553b", width=4), mode='lines+markers'))
                # 신뢰구간 밴드 [cite: 201, 266]
                fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['upper'], line=dict(width=0), showlegend=False))
                fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['lower'], fill='tonexty', 
                                             fillcolor='rgba(239, 85, 59, 0.1)', line=dict(width=0), name="신뢰구간(95%)"))
    
                fig_all.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_all, use_container_width=True)
    
        with res_row_col2:
            with st.container(border=True, height=600):
                st.write("**📑 예측 결과 분석 리포트**")
                # 1. 요약 지표 (summarize_forecast 활용) [cite: 253, 259, 273]
                summary = summarize_forecast(f_res)
                m1, m2, m3 = st.columns(3)
                m1.metric("평균 예측치", f"{summary['avg']:,.1f}")
                m2.metric("최대 수요", f"{summary['max']:,.1f}")
                m3.metric("최소 수요", f"{summary['min']:,.1f}")
                
                st.divider()
                # 2. 기간 단위 환산 (aggregate_forecast 활용) [cite: 254, 262, 274]
                # 사이드바에서 선택된 time_unit(일, 주, 월, 년) 연동
                freq_map = {"일": "D", "주": "W", "월": "M", "년": "Y"}
                selected_unit = st.session_state.get("unit_sel_box", "일")
                agg_val = aggregate_forecast(f_res, freq=freq_map.get(selected_unit, "D"))
                st.info(f"✨ 해당 기간 **{selected_unit} 단위** 환산 예측치: **{agg_val:,.2f}**")
                
                # 3. 상세 데이터 테이블 (forecast_table 활용) [cite: 253, 259, 273]
                st.write("**📅 일자별 상세 예측 데이터**")
                f_table = forecast_table(f_res, future_dates)
                st.dataframe(f_table, use_container_width=True, height=250)
    
        st.divider()
    
        # -----------------------------
        # 5️⃣ 성능 평가 및 모델 검증 (행 전체 사용)
        # -----------------------------
        st.subheader("📏 성능 평가 결과 및 모델 검증")
        # [병렬 배치] 왼쪽: 성능 로그(1) | 오른쪽: 검증 차트(1) [cite: 213, 261, 275]
        eval_row_col1, eval_row_col2 = st.columns([1, 1])
        
        with eval_row_col1:
            with st.container(border=True, height=520):
                st.write("**📊 성능 평가 로그 (History Log)**")
                if not st.session_state["perf_log"].empty:
                    st.dataframe(st.session_state["perf_log"], use_container_width=True, height=350)
                st.info("💡 MAE·RMSE(낮음 우수), MAPE(오차율 %), TS(±4 정상)") [cite: 166, 268]
    
        with eval_row_col2:
            with st.container(border=True, height=520):
                st.write("** 모델 시각화 (Actual vs Prediction)**")
                y_val_pred = st.session_state.get("current_val_pred") # 오타 수정: current_val_pred
                if y_val_pred is not None:
                    split_idx = int(len(ps) * 0.8)
                    test_p = ps.iloc[split_idx:]
                    # 실제 데이터와 예측치 대비 시각화 [cite: 164, 269]
                    fig_val = plot_forecast_vs_actual(test_p, {st.session_state.get("model_sel_box", "Model"): y_val_pred})
                    fig_val.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_val, use_container_width=True)
                else:
                    st.warning("⚠️ '예측 실행' 시 모델 검증 차트가 생성됩니다.")
