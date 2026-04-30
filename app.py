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
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=("원본", "추세", "계절성", "잔차")
    )

    fig.add_trace(go.Scatter(y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(height=800, showlegend=False)
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

# -----------------------------
# 0. 세션 상태 초기화 (누적 기록 포함)
# -----------------------------
if "df" not in st.session_state: st.session_state["df"] = None
if "processed" not in st.session_state: st.session_state["processed"] = None
if "perf_log" not in st.session_state: st.session_state["perf_log"] = pd.DataFrame()
if "forecast_res" not in st.session_state: st.session_state["forecast_res"] = None

st.set_page_config(layout="wide", page_title="수요 예측 앱")

# -----------------------------
# 1. 데이터 업로드 및 기본 설정
# -----------------------------
header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("C321032 박하율")

col1, col2 = st.columns([1, 2])

with col1:
    with st.container(border=True, height=400):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
        if file:
            df = load_data(file)
            # 날짜 컬럼 자동 인식 및 인덱스 설정
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            st.session_state["df"] = df
            st.success("데이터 로드 완료!")

        target_series = st.session_state["df"].iloc[:, 0]
        if st.session_state["processed"] is None:
            with st.spinner("데이터를 분석 중입니다..."):
                st.session_state["processed"] = preprocess_series(target_series)
            st.success("분석 완료!")
        
        st.write("**데이터 요약**")
        st.dataframe(df.describe().T, use_container_width=True)

with col2:
    if st.session_state["df"] is not None:
        with st.container(border=True, height=400):
            st.subheader("📈시계열 시각화")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("데이터 미리보기")
                st.dataframe(df.head(), use_container_width=True)
            with c2:
                st.plotly_chart(plot_preprocessing(raw, ps), use_container_width=True)

# -----------------------------
# 3~6. 분석 리포트 영역 (전처리 완료 시 노출)
# -----------------------------
if st.session_state["processed"] is not None:
ps = st.session_state["processed"]
raw = st.session_state["df"].iloc[:, 0]

# 3. 전처리 비교 시각화
st.subheader("시계열 분석 리포트")
col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    with st.container(border=True):
        st.subheader("ADF & Ljung-Box 검정")
        adf = run_stationarity_test(ps) [cite: 6]
        lb = run_ljungbox_test(ps) [cite: 6]
        
        # 1. ADF 검정 (정상성)
        st.metric(
            label="ADF p-value (정상성)", 
            value=f"{adf['p_value']:.4f}", 
            delta="정상 데이터" if adf['is_stationary'] else "비정상(차분 필요)",
            delta_color="normal" if adf['is_stationary'] else "inverse"
        )
        
        # 2. Ljung-Box 검정 (백색잡음 여부) [cite: 7]
        # p-value가 0.05보다 크면 백색잡음(유의미한 패턴 없음)으로 판단
        is_white_noise = lb['p_value'] > 0.05
        st.metric(
            label="Ljung-Box p-value (백색잡음)", 
            value=f"{lb['p_value']:.4f}",
            delta="백색잡음 (패턴 없음)" if is_white_noise else "자기상관 존재 (예측 가능)",
            delta_color="off" if is_white_noise else "normal"
        )
        
        if is_white_noise:
            st.warning("⚠️ 백색잡음")
        else:
            st.info("✅ 모델링 적합")

with row2_col2:
    with st.container(border=True):
        st.subheader("시계열 분해")
        st.plotly_chart(plot_decomposition(decomp_res), use_container_width=True)

with row2_col3:
    with st.container(border=True):
        st.subheader("〽️시계열 분해 요약")
        decomp_res = decompose_series(ps, selected_period)
        summary = summarize_decomposition(decomp_res)
        st.write(f"추세 강도: **{summary['trend_strength']}**")
        st.write(f"계절성 강도: **{summary['seasonal_strength']}**")

# 5. 시계열 분해 차트
st.subheader("5️⃣ 시계열 분해 결과")
st.plotly_chart(plot_decomposition(decomp_res), use_container_width=True)

st.divider()

# -----------------------------
# 7. 모델 선택 및 시평 설정
# -----------------------------
st.divider()
st.subheader("⚙️모델 선택 및 설정")

with st.container(border=True):
    # 상단 요약 정보 (이미지 상단 메타데이터 영역)
    time_info = analyze_time_index(st.session_state["processed"].index)
    r1, r2 = st.rows(2)
    with r1:
        st.markdown(f"📅 **Datetime 범위:** `{time_info['start'].strftime('%Y.%m.%d')} ~ {time_info['end'].strftime('%Y.%m.%d')}`")
    with r2:
        st.markdown(f"🔄 **평균 데이터 주기:** `1일 (총 {len(st.session_state['processed'])}개 샘플)`")
    
    st.divider()

    # 모델 설정 입력 영역 (2x2 그리드)
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        model_type = st.selectbox("예측 모델", ["SARIMA", "ARIMA", "STL", "HW", "ES", "MA"], key="model_sel_box")
        horizon = st.number_input("예측 길이(시평)", min_value=1, value=7, key="horizon_input")
    with col_input2:
        method = st.checktbox("평가 방식", ["Rolling", "Block"], key="method_sel_box")
        time_unit = st.checkbox("시간 단위", ["일", "주", "월", "년"], key="unit_sel_box")

    # 실행 버튼부
    if st.button("▶️ 예측 실행", use_container_width=True, type="primary"):
        ps = st.session_state["processed"]
        # 성능 평가용 Split 및 검증 실행
        split_idx = int(len(ps) * 0.8)
        train_p, test_p = ps.iloc[:split_idx], ps.iloc[split_idx:]
        
        y_pred = evaluate_forecast(train_p, test_p, model_type, horizon)
        metrics = evaluate_metrics(test_p[:len(y_pred)], y_pred, model_type, method)
        
        # 세션 상태 업데이트 
        st.session_state["perf_log"] = update_log(st.session_state["perf_log"], metrics)
        st.session_state["forecast_res"] = get_forecast(ps, horizon, model_type, time_info['suggested_periods'][0])
        st.toast(f"{model_type} 모델 예측 완료!")

    if st.button("🗑️ 로그 초기화", use_container_width=True):
        st.session_state["perf_log"] = pd.DataFrame()
        st.rerun()

# -----------------------------
# 8~9. 평가 결과 및 예측 리포트
# -----------------------------
if st.session_state["forecast_res"] is not None:
    st.divider()
    # 이미지 레이아웃에 맞춰 컬럼 배치 (평가 로그 / 수요 예측 결과)
    res_col1, res_col2 = st.columns([1, 1])

    # 📏 왼쪽: 평가 결과 및 로그 섹션
    with res_col1:
        st.markdown("### 📊 수요 예측 결과")
        with st.container(border=True, height=580):
            f_res = st.session_state["forecast_res"]
            ps = st.session_state["processed"]
            
            # 미래 날짜 생성 (인덱스 활용) 
            future_dates = pd.date_range(start=ps.index[-1], periods=len(f_res['mean'])+1, freq=ps.index.freq)[1:]
            
            # 요청하신 스타일의 메인 차트 구성
            fig_all = go.Figure()
            # 과거 데이터 (파란색)
            fig_all.add_trace(go.Scatter(x=ps.index, y=ps.values, name="과거 데이터", line=dict(color="#1f77b4")))
            # 미래 예측 (빨간색, 굵게)
            fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['mean'], name="미래 예측", 
                                         line=dict(color="#ef553b", width=4), mode='lines+markers'))
            
            # 신뢰구간 (이미지 스타일의 밴드 추가) 
            fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['upper'], line=dict(width=0), showlegend=False))
            fig_all.add_trace(go.Scatter(x=future_dates, y=f_res['lower'], fill='tonexty', 
                                         fillcolor='rgba(239, 85, 59, 0.1)', line=dict(width=0), name="신뢰구간"))

            fig_all.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_all, use_container_width=True)
            
            # 하단 결과 요약 텍스트 박스 (이미지 하단 스타일) [cite: 43, 54]
            st.markdown(f"""
            <div style="background-color: #e8f0fe; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;">
                ✨ <b>결과 요약:</b> 향후 {horizon}{time_unit}간 평균 예상 수요는 <b>{f_res['mean'].mean():,.1f}</b>입니다.
            </div>
            """, unsafe_allow_html=True)

    # 📊 오른쪽: 수요 예측 결과 섹션
    with res_col2:
        st.markdown("### 📏 평가 결과 및 로그")
        with st.container(border=True, height=580):
            # 1. 성능 지표 테이블 [cite: 34, 117]
            if not st.session_state["perf_log"].empty:
                st.dataframe(st.session_state["perf_log"], use_container_width=True)
            
            st.info("💡 **지표 참고 사항:** MAE(낮음 우수), MdRAE(<1 우수), TS(±4 정상)")
            
            # 2. 모델별 검증 예측 비교 (이미지 하단 점선 차트 영역) 
            st.markdown("**검증 데이터 예측 비교 (Actual vs Pred)**")
            
            # 검증용 데이터 준비 (evaluate_forecast 활용)
            split_idx = int(len(ps) * 0.8)
            test_p = ps.iloc[split_idx:]
            # 최신 모델의 검증 결과 시각화
            fig_val = plot_forecast_vs_actual(test_p, {model_type: y_pred})
            fig_val.update_layout(height=280, margin=dict(l=5, r=5, t=10, b=10), showlegend=True)
            st.plotly_chart(fig_val, use_container_width=True)
