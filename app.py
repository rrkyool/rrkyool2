import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 이상치 처리 (Hampel Filter)
from sktime.transformations.series.outlier_detection import HampelFilter

# 2. 결측치 처리 (Imputer)
from sktime.transformations.series.impute import Imputer

# 3. 디노이징/평활화 (Exponential Smoothing) - 여기가 에러 난 부분
from sktime.transformations.series.exponent_smoothing import ExponentialSmoothingTransformer

# 4. 평가지표
from sktime.performance_metrics.forecasting import (
    mean_absolute_error, 
    median_relative_absolute_error,
    mean_absolute_scaled_error
)

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon

from sktime.forecasting.model_selection import forecasting_efficiency_stats, evaluate
from sktime.forecasting.model_selection import SlidingWindowSplitter, ExpandingWindowSplitter

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX # SARIMA 작동을 위해 추가
from pmdarima import auto_arima

# -----------------------------
# 기본 설정 및 헤더
# -----------------------------
st.set_page_config(layout="wide")

header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("C321032 박하율")

# 세션 상태 관리
if "results_df" not in st.session_state:
    # model_score 컬럼 추가
    st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 분석 함수
# -----------------------------
def load_data(file):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

#결측치 처리
def fill_missing_values(series, method='linear'):
    imputer = Imputer(method=method)
    filled_series = imputer.fit_transform(series)
    return filled_series

#이상치 처리
def hampel_filter(series, window=5, n=3):
    transformer = HampelFilter(window_length=2 * window + 1, n_sigma=n, return_inline=True)
    new_series = transformer.fit_transform(series)
    return new_series

#디노이징
def denoise_series(series, smoothing_level=0.5):
    """
    FFT 대신 sktime의 지수 평활법을 사용하여 노이즈를 제거합니다.
    이 방법은 데이터의 최근 추세를 더 잘 반영하며 말단 왜곡이 적습니다.
    """
    # smoothing_level (alpha): 0에 가까울수록 매끄러워지고(노이즈 제거 강함), 
    # 1에 가까울수록 원본 데이터에 가깝게 유지됩니다.
    transformer = ExponentialSmoothingTransformer(smoothing_level=smoothing_level)
    
    # 데이터를 학습하고 변환합니다.
    # sktime 트랜스포머는 입력 데이터의 인덱스와 형식을 보존합니다.
    denoised_series = transformer.fit_transform(series)
    
    return denoised_series

#평가지표
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mdrae(y_true, y_pred, y_train=None):
    return median_relative_absolute_error(y_true, y_pred)

def tracking_signal(y_true, y_pred):
    errors = y_true - y_pred
    mad = mean_absolute_error(y_true, y_pred)
    if mad == 0: return 0
    return errors.sum() / mad

#예측 모델 적용
def get_best_forecast(train, horizon, model_type):
    train = pd.Series(train).astype(float)
    m_val = 12
    fh = np.arange(1, horizon + 1) # 예측 구간 설정 (1부터 horizon까지)
    
    # 2. 모델 타입별 로직
    if model_type == "이동평균":
        # sktime의 NaiveForecaster는 이동평균(window)을 지원합니다.
        forecaster = NaiveForecaster(strategy="mean", window_length=12)
        
    elif model_type == "지수평활":
        # 기본적인 단순 지수평활
        forecaster = ExponentialSmoothing(trend=None, seasonal=None)
        
    elif model_type == "Holt-Winters":
        # 데이터 양에 따른 자동 대응
        if len(train) < 2 * m_val:
            st.warning("⚠️ 데이터 부족으로 계절성 제외, 추세만 반영합니다.")
            forecaster = ExponentialSmoothing(trend='add', seasonal=None)
        else:
            # sktime은 자동으로 최적 파라미터를 찾으려 시도합니다.
            forecaster = ExponentialSmoothing(trend='add', seasonal='add', sp=m_val)
            
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        if is_seasonal and len(train) < 2 * m_val:
            st.info("💡 데이터 부족으로 ARIMA(비계절성)로 자동 전환합니다.")
            is_seasonal = False
            
        # 최적화와 속도의 균형을 맞춘 AutoARIMA 설정
        forecaster = AutoARIMA(
            sp=m_val if is_seasonal else 1,
            suppress_warnings=True,
            error_action="ignore",
            # 속도 향상을 위한 파라미터 제한
            max_p=3, max_q=3, 
            seasonal=is_seasonal,
            stepwise=True  # 모든 조합을 다 검사하지 않고 효율적으로 탐색
        )

    # 3. 학습 및 예측
    try:
        forecaster.fit(train)
        y_pred = forecaster.predict(fh)
        return y_pred.values
    except Exception as e:
        st.error(f"모델 학습 오류: {e}")
        return np.repeat(train.iloc[-1], horizon)


#예측 평가 구간
def run_backtest(train, test, model_obj, strategy="rolling"):
    if strategy == "rolling":
        cv = SlidingWindowSplitter(window_length=len(train), step_length=1)
    else:
        cv = ExpandingWindowSplitter(initial_window=len(train), step_length=1)
    
    y_full = pd.concat([train, test])
    
    results = evaluate(
        forecaster=model_obj, 
        cv=cv, 
        y=y_full, 
        strategy="refit", # 매번 모델을 다시 최적화 (파라미터 고정 문제 해결)
        return_data=True
    )
    
    return results["y_pred"].apply(lambda x: x.iloc[0]).values

#정상성 검정
def run_statistical_tests(series):
    """
    ADF(정상성) 및 Ljung-Box(백색잡음) 검정을 수행하고 결과를 요약합니다.
    """
    results = []
    
    # 1. ADF Test (정상성 검정)
    # sktime 데이터는 Series 형태이므로 바로 statsmodels 함수에 전달 가능합니다.
    adf_result = adfuller(series.dropna())
    adf_p_value = adf_result[1]
    is_stationary = adf_p_value < 0.05
    
    results.append({
        "검정명": "ADF (정상성)",
        "귀무가설(H0)": "단위근 존재 (비정상)",
        "p-value": f"{adf_p_value:.4f}",
        "해석": "정상성 확보" if is_stationary else "비정상 (차분 필요)"
    })
    
    # 2. Ljung-Box Test (자기상관/백색잡음 검정)
    # lag=10 정도로 설정하여 전반적인 패턴 존재 여부를 확인합니다.
    lb_result = acorr_ljungbox(series.dropna(), lags=[1, 10], return_df=True)
    
    for lag in [1, 10]:
        lb_p_value = lb_result.loc[lag, 'lb_pvalue']
        is_pattern = lb_p_value < 0.05
        
        results.append({
            "검정명": f"Ljung-Box (lag={lag})",
            "귀무가설(H0)": "자기상관 없음 (백색잡음)",
            "p-value": f"{lb_p_value:.4f}",
            "해석": "패턴 존재 (모형 개선 가능)" if is_pattern else "백색잡음 (추가 모형 불필요)"
        })
        
    return pd.DataFrame(results)
    
# -----------------------------
# 상단 레이아웃
# -----------------------------
top_left, top_right = st.columns(2)

with top_left:
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 선택하세요", label_visibility="collapsed")
        
        if file:
            # 1. 데이터 로드 (기본적인 구조 유지)
            df_raw_data = pd.read_csv(file) 
            
            if df_raw_data is not None:
                date_col = df_raw_data.columns[0]
                value_col = df_raw_data.select_dtypes(include=np.number).columns[0]
                
                df_raw_data[date_col] = pd.to_datetime(df_raw_data[date_col])
                df_raw_data = df_raw_data.sort_values(date_col).set_index(date_col)
                
                # 원본 데이터 보존
                raw_values = df_raw_data[value_col].copy()
                
                # --- [수정 부분 1: sktime 기반 전처리 파이프라인] ---
                # 1) 결측치 보간 -> 2) 이상치 제거 -> 3) 지수평활 디노이징 (FFT 대체)
                proc_values = (
                    fill_missing_values(raw_values)
                    .pipe(apply_hampel_filter)
                    .pipe(denoise_series, smoothing_level=0.3) # 말단 왜곡을 방지하는 지수평활
                )
                
                # 전처리된 데이터를 데이터프레임에 반영
                df_raw_data["processed"] = proc_values
                
                # 2. 시각화 (원본 vs 전처리)
                fig_prep = go.Figure()
                fig_prep.add_trace(go.Scatter(
                    x=df_raw_data.index, y=raw_values, 
                    name="원본", line=dict(color="lightgray", width=1), opacity=0.6
                ))
                fig_prep.add_trace(go.Scatter(
                    x=df_raw_data.index, y=proc_values, 
                    name="전처리", line=dict(color="royalblue", width=2)
                ))
                fig_prep.update_layout(
                    height=250, # 정상성 표 공간 확보를 위해 약간 조정 가능
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_prep, use_container_width=True)
                
                # --- [정상성 검정 결과 출력] ---
                st.markdown("---")
                st.write("🔍 **데이터 통계적 특성 검정**")
                
                test_results = run_statistical_tests(proc_values)
                
                st.dataframe(
                    test_results, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "p-value": st.column_config.TextColumn("p-value", width="small"),
                        "해석": st.column_config.TextColumn("해석", width="medium")
                    }
                )

with top_right:
    with st.container(border=True):
        st.subheader("⚙️ 모델 선택 및 설정")

        # --- 데이터 정보 표시 로직 추가 ---
        if file and 'df_raw_data' in locals():
            # 날짜 범위 추출
            start_date = df_raw_data.index.min().strftime('%Y.%m.%d')
            end_date = df_raw_data.index.max().strftime('%Y.%m.%d')
            
            # 주기 계산 (인덱스 간 차이의 최빈값이나 평균 활용)
            if len(df_raw_data) > 1:
                diffs = df_raw_data.index.to_series().diff().dt.days.dropna()
                avg_cycle = int(diffs.mode()[0]) if not diffs.mode().empty else int(diffs.mean())
                
                # 정보 출력
                st.markdown(f"📅 **Datetime 범위**: `{start_date} ~ {end_date}`")
                st.markdown(f"🔄 **평균 데이터 주기**: `{avg_cycle}일` (총 {len(df_raw_data)}개 샘플)")
                st.divider() 
                
        c1, c2 = st.columns(2)
        with c1:
            m_type = st.selectbox("예측 모델", ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA"])
            h_len = st.number_input("예측 길이(시평)", 1, 100, 12)
            
        with c2:
            e_type = st.selectbox("평가 방식", ["Rolling", "Expanding"])
            u_type = st.selectbox("시간 단위", ["일", "월", "년"], index=2)
            
        st.write("") # 간격
        btn_run = st.button("🚀 예측 실행", use_container_width=True, type="primary")
        if st.button("🗑️ 로그 초기화", use_container_width=True):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
            st.session_state.eval_preds = {}
            st.rerun()

# -----------------------------
# 분석 실행 로직
# -----------------------------
if file and btn_run:
    df = df_raw_data.copy()
    split_idx = int(len(df) * 0.8)
    train_set, test_set = df[value_col].iloc[:split_idx], df[value_col].iloc[split_idx:]
    
    with st.spinner(f"🚀 {m_type} ({e_type}) 분석 중..."):
        # [핵심] 평가 방식 호출
        if e_type == "Rolling":
            test_preds = rolling_forecast(train_set, test_set, m_type)
        else:
            test_preds = expanding_forecast(train_set, test_set, m_type)
            
        # 결과 저장
        eval_key = f"{m_type}_{e_type}"
        st.session_state.eval_preds[eval_key] = test_preds
        
        # 미래 예측 수행
        forecast_vals = get_best_forecast(df[value_col], h_len, m_type)
        avg_f = round(float(np.mean(forecast_vals)), 1)
        
        # 지표 계산
        m_val = round(mae(test_set, test_preds), 4)
        r_val = round(mdrae(test_set, test_preds), 4)
        ts_val = round(tracking_signal(test_set, test_preds), 4)
        
        new_entry = pd.DataFrame([{
            "모델 종류": m_type, "평가 방법": e_type, 
            "MAE": m_val, "MdRAE": r_val, "TS": ts_val, "예측 평균": avg_f
        }])
        st.session_state.results_df = pd.concat([st.session_state.results_df, new_entry], ignore_index=True)

# -----------------------------
# 하단 레이아웃 (버튼 클릭 여부와 상관없이 결과가 있으면 표시)
# -----------------------------
if not st.session_state.results_df.empty:
    # 데이터 재정의 (그래프용)
    df = df_raw_data.copy()
    split_idx = int(len(df) * 0.8)
    
    bot_left, bot_right = st.columns([1, 1.2])
    
    with bot_left:
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
            st.markdown("**💡 지표 참고 사항**: MAE(낮음 우수), MdRAE(<1 우수), TS(±4 정상)")
            
            # 누적 시뮬레이션 그래프
            actual_y = df[value_col].iloc[split_idx:]
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=actual_y.index, y=actual_y, name="Actual", line=dict(color="green", dash='dot')))
            for name, p_vals in st.session_state.eval_preds.items():
                fig_eval.add_trace(go.Scatter(x=actual_y.index, y=p_vals, name=f"Pred({name})", line=dict(dash='dot')))
            
            fig_eval.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_eval, use_container_width=True)

    with bot_right:
        with st.container(border=True):
            st.subheader("📊 수요 예측 결과")
            
            # 미래 날짜 및 예측값 계산
            freq_map = {"일": "D", "월": "MS", "년": "YS"}
            future_dates = pd.date_range(df.index[-1], periods=h_len+1, freq=freq_map[u_type])[1:]
            
            # 마지막 로그에 기록된 모델로 다시 미래 예측 수행 (시각화 유지용)
            last_model = st.session_state.results_df["모델 종류"].iloc[-1]
            final_forecast = get_best_forecast(df[value_col], h_len, last_model)
            
            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(x=df.index, y=df[value_col], name="과거 데이터", line=dict(color="#1f77b4")))
            fig_all.add_trace(go.Scatter(x=future_dates, y=final_forecast, name="미래 예측", line=dict(color="#ef553b", width=3)))
            
            fig_all.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_all, use_container_width=True)
            
            last_avg = st.session_state.results_df['예측 평균'].iloc[-1]
            st.info(f"✨ **결과 요약**: 향후 {h_len}{u_type}간 평균 예상 수요는 **{last_avg}**입니다.")
            
elif file:
    st.info("👈 설정 후 '예측 실행' 버튼을 눌러 분석을 시작하세요.")
