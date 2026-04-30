import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots

from scipy.signal import savgol_filter

from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# 1. 분석 핵심 함수 정의
# -----------------------------

#1. 데이터 로드
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

#2. 데이터 전처리
#이상치 탐지 및 대체
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

#디노이징
def denoise_series(series, window=11, poly=2):
    return pd.Series(
        savgol_filter(series, window_length=window, polyorder=poly),
        index=series.index
    )

#최종 전처리 pipeline
def preprocess_series(series):
    s = series.astype(float)

    # 1. 결측치
    s = s.interpolate(limit_direction='both')

    # 2. 이상치
    s = hampel_filter(s, window=5, n=3)

    # 3. 디노이징 (택1)
    s = denoise_series(s)  # Savitzky-Golay 추천

    return s

#3. 전처리 시각화
def plot_preprocessing(raw, processed):
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=raw, name="원본", opacity=0.5))
    fig.add_trace(go.Scatter(y=processed, name="전처리", line=dict(color="royalblue")))

    return fig

#4. 정상성 검정
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


#5. 시계열 분해 및 분석
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

 #6. 데이터 기간 및 freq 계산
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

#7. 모형 선택 및 예측 수행 함수
#단순이동평균
def ma_forecast(train, horizon, window):
    val = train.rolling(window=window, min_periods=1).mean().iloc[-1]
    mean = np.repeat(val, horizon)
    std = train.std()

    return mean, mean - std, mean + std

#지수평활
def exp_forecast(train, horizon):
    model = ExponentialSmoothing(train).fit()
    forecast = model.forecast(horizon)

    std = train.std()
    return forecast.values, forecast.values - std, forecast.values + std

#holt winters
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

#STL 분해
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

#autoARIMA
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

#모델 통합 함수
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

#8. 성능 평가 및 누적 로그 & 누적 시각화
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

#8. 성능평가 지표
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

#누적 로그 테이블 생성
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

#9. 수요 예측 결과 시각화 및 분석 
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

# -----------------------------
# 2. 메인 앱 설정
# -----------------------------

# -----------------------------
# 상태 초기화
# -----------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None

if "processed" not in st.session_state:
    st.session_state["processed"] = None

if "forecast" not in st.session_state:
    st.session_state["forecast"] = None

if "view" not in st.session_state:
    st.session_state["view"] = "dashboard"

if "selected" not in st.session_state:
    st.session_state["selected"] = None

# -----------------------------
# 뷰 전환 함수
# -----------------------------
def go_detail(name):
    st.session_state.view = "detail"
    st.session_state.selected = name

def go_home():
    st.session_state.view = "dashboard"
    st.session_state.selected = None

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# 상태 초기화
# -----------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None

if "processed" not in st.session_state:
    st.session_state["processed"] = None

if "forecast" not in st.session_state:
    st.session_state["forecast"] = None

if "view" not in st.session_state:
    st.session_state["view"] = "dashboard"

if "selected" not in st.session_state:
    st.session_state["selected"] = None

# -----------------------------
# 대시보드
# -----------------------------
def render_dashboard():
    st.set_page_config(layout="wide")

    st.title("📊 📈 시계열 분석 Project1 수요 예측")
    st.subheader("C321032 박하율")


    df = st.session_state["df"]
    processed = st.session_state["processed"]

    col1, col2 = st.columns(2)

    # 1. 업로드
    with col1:
        st.subheader("데이터 업로드")
        file = st.file_uploader("CSV 업로드")

        if file:
            df = load_data(file)
            st.session_state["df"] = df

        if df is not None:
            st.dataframe(df.head())

    # 2. 전처리
    with col2:
        st.subheader("전처리")

        if df is not None:
            col = st.selectbox("컬럼 선택", df.columns)

            if st.button("전처리 실행"):
                processed = preprocess_series(df[col])
                st.session_state["processed"] = processed

        if processed is not None:
            st.plotly_chart(plot_preprocessing(df[col], processed))

            st.button("🔍 확대", on_click=lambda: go_detail("preprocess"))

    # 3. 정상성
    if processed is not None:
        st.subheader("정상성 검정")

        adf = run_stationarity_test(processed)
        lb = run_ljungbox_test(processed)

        st.write(f"ADF: {adf['p_value']:.4f}")
        st.write(f"Ljung-Box: {lb['p_value']:.4f}")

        st.button("🔍 확대", on_click=lambda: go_detail("stationarity"))

    # 4. 예측
    if processed is not None:
        st.subheader("예측")

        if st.button("예측 실행"):
            result = get_forecast(processed, 10)
            st.session_state["forecast"] = result

        if st.session_state["forecast"] is not None:
            st.plotly_chart(plot_forecast_result(processed, st.session_state["forecast"]))

            st.button("🔍 확대", on_click=lambda: go_detail("forecast"))

# -----------------------------
# 상세 보기
# -----------------------------
def render_detail():
    st.button("⬅️ 돌아가기", on_click=go_home)

    selected = st.session_state.selected
    df = st.session_state["df"]
    processed = st.session_state["processed"]

    if selected == "preprocess":
        st.title("전처리 상세")
        st.plotly_chart(plot_preprocessing(df.iloc[:,0], processed), use_container_width=True)

    elif selected == "stationarity":
        st.title("정상성 상세")
        adf = run_stationarity_test(processed)
        st.write(adf)

    elif selected == "forecast":
        st.title("예측 상세")
        st.plotly_chart(plot_forecast_result(processed, st.session_state["forecast"]), use_container_width=True)

# -----------------------------
# 실행
# -----------------------------
if st.session_state.view == "dashboard":
    render_dashboard()
else:
    render_detail()



#화면 컨테이너
def big_section(title):
    st.markdown(f"""
    <div style="
        padding: 30px;
        margin-bottom: 30px;
        border-radius: 15px;
        background-color: #f8f9fa;
    ">
    <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)

if step == "1. 데이터 업로드":
    big_section("📂 데이터 업로드")

    uploaded_file = st.file_uploader("파일 업로드", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.success("업로드 완료")

        st.dataframe(df.head(), use_container_width=True)

elif step == "2. 전처리":
    big_section("🧹 데이터 전처리")

    col = st.selectbox("타겟 컬럼", df.columns)

    if st.button("전처리 실행"):
        processed = preprocess_series(df[col])

        st.session_state["processed"] = processed
        st.success("전처리 완료")

elif step == "3. 전처리 시각화":
    big_section("📈 전처리 비교")

    fig = plot_preprocessing(df[col], st.session_state["processed"])

    st.plotly_chart(fig, use_container_width=True)

elif step == "4. 정상성 검정":
    big_section("📊 정상성 검정")

    s = st.session_state["processed"]

    adf = run_stationarity_test(s)
    lb = run_ljungbox_test(s)

    col1, col2 = st.columns(2)

    col1.metric("ADF p-value", f"{adf['p_value']:.4f}")
    col2.metric("Ljung-Box p-value", f"{lb['p_value']:.4f}")

elif step == "5. 시계열 분해":
    big_section("📉 시계열 분해")

    period = st.slider("주기 선택", 2, 30, 7)

    result = decompose_series(st.session_state["processed"], period)

    fig = plot_decomposition(result)

    st.plotly_chart(fig, use_container_width=True)

elif step == "6. 주기 분석":
    big_section("⏱️ 주기 분석")

    info = analyze_time_index(df.index)

    st.write(info)

elif step == "7. 모델링":
    big_section("🤖 모델 선택 및 예측")

    model_type = st.selectbox("모델 선택", ["MA(이동평균)", "ES(지수평활)", "HW(홀트윈터)", "STL(분해)", "ARIMA","SARIMA"])
    horizon = st.slider("예측 기간", 5, 60, 14)

    if st.button("예측 실행"):
        result = get_forecast(
            st.session_state["processed"],
            horizon,
            model_type
        )

        st.session_state["forecast"] = result

elif step == "8. 성능 평가":
    big_section("📊 모델 성능 평가")

    # block 방식 사용
    preds = evaluate_forecast(train, test, model_type, horizon)

    res = evaluate_metrics(test, preds, model_type, "Block")

    st.dataframe(pd.DataFrame([res]), use_container_width=True)

    fig = plot_forecast_vs_actual(test, {"Pred": preds})
    st.plotly_chart(fig, use_container_width=True)

elif step == "9. 예측 결과":
    big_section("📈 수요 예측 결과")

    result = st.session_state["forecast"]

    fig = plot_forecast_result(
        st.session_state["processed"],
        result,
        df.index
    )

    st.plotly_chart(fig, use_container_width=True)

    summary = summarize_forecast(result)

    col1, col2, col3 = st.columns(3)
    col1.metric("평균", f"{summary['avg']:.1f}")
    col2.metric("최소/최대", f"{summary['min']:.1f}~{summary['max']:.1f}")
    col3.metric("신뢰구간", f"{summary['ci_low']:.1f}~{summary['ci_high']:.1f}")

    st.dataframe(forecast_table(result, None), use_container_width=True)
