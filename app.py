import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

def fft_denoise(signal, keep_ratio=0.3):
    fft = np.fft.fft(signal)
    n = len(fft)
    cutoff = int(n * keep_ratio)
    fft[cutoff:n-cutoff] = 0
    return np.fft.ifft(fft).real

def mae(y, yhat): return np.mean(np.abs(np.array(y) - np.array(yhat)))

def mdrae(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    naive = y[1:]; y_prev = y[:-1]
    denom = np.abs(naive - y_prev)
    num = np.abs(naive - yhat[1:])
    return np.median(num / (denom + 1e-8))

def tracking_signal(y, yhat):
    err = np.array(y) - np.array(yhat)
    mad = np.mean(np.abs(err))
    return np.sum(err) / (mad + 1e-8)

def get_best_forecast(train, horizon, model_type):
    train = pd.Series(train).astype(float)
    # 데이터 부족 여부 판단 (주기 12 기준, 최소 24개 이상 권장)
    m_val = 12
    is_data_insufficient = len(train) < 2 * m_val
    
    if model_type == "이동평균":
        val = train.rolling(window=12, min_periods=1).mean().iloc[-1]
        return np.repeat(val, horizon)
    
    elif model_type == "지수평활":
        model = ExponentialSmoothing(train).fit()
        return model.forecast(horizon).values
    
    elif model_type == "Holt-Winters":
        try: 
            model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
        except: 
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
        return model.forecast(horizon).values
    
    elif model_type in ["ARIMA", "SARIMA"]:
        is_seasonal = (model_type == "SARIMA")
        
        # SARIMA 선택 시 데이터가 부족한 경우 처리
        if is_seasonal and is_data_insufficient:
            st.warning(f"⚠️ 계절성 분석을 위한 데이터가 부족합니다. (현재 데이터: {len(train)}개, 최소 필요: {2*m_val}개)")
            st.info("안정적인 분석을 위해 일반 ARIMA 모델로 자동 전환하여 결과를 생성합니다.")
            is_seasonal = False # 계절성 비활성화
            
        try:
            step_m = auto_arima(
                train, 
                seasonal=is_seasonal, 
                m=m_val if is_seasonal else 1, 
                stepwise=True, 
                suppress_warnings=True, 
                error_action="ignore",
                max_p=3, max_q=3,
                trace=False
            )
            return step_m.predict(n_periods=horizon).values
        except Exception as e:
            # 최종 예외 처리
            step_m = auto_arima(train, seasonal=False, stepwise=True)
            return step_m.predict(n_periods=horizon).values
            
    return np.repeat(train.iloc[-1], horizon)

# -----------------------------
# [함수 수정] 내부 연산 속도 최적화
# -----------------------------

def rolling_forecast(train, test, model_type):
    history = list(train)
    preds = []
    window_size = len(train)
    
    for t in range(len(test)):
        current_train = history[-window_size:]
        
        if model_type == "ARIMA":
            # 속도 개선: order 고정 및 탐색 생략
            model = ARIMA(current_train, order=(1,1,1)).fit()
            yhat = model.forecast(steps=1)[0]
        elif model_type == "SARIMA":
            if len(current_train) < 24:
                model = ARIMA(current_train, order=(1,1,1)).fit()
            else:
                # 속도 개선: disp=False 및 반복 횟수 최적화
                model = SARIMAX(current_train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            yhat = model.forecast(steps=1)[0]
        else:
            yhat = get_best_forecast(current_train, 1, model_type)[0]
            
        preds.append(yhat)
        history.append(test.iloc[t])
    return np.array(preds)

def expanding_forecast(train, test, model_type):
    preds = []
    for i in range(len(test)):
        # Expanding: 데이터가 1개씩 계속 누적됨
        hist = pd.concat([train, test[:i]])
        
        if model_type == "ARIMA":
            model = ARIMA(hist, order=(1,1,1)).fit()
            yhat = model.forecast(steps=1)[0]
        elif model_type == "SARIMA":
            if len(hist) < 24:
                model = ARIMA(hist, order=(1,1,1)).fit()
            else:
                model = SARIMAX(hist, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            yhat = model.forecast(steps=1)[0]
        else:
            yhat = get_best_forecast(hist, 1, model_type)[0]
            
        preds.append(yhat)
    return np.array(preds)

# -----------------------------
# 상단 레이아웃
# -----------------------------
top_left, top_right = st.columns(2)

with top_left:
    with st.container(border=True):
        st.subheader("📂 데이터 업로드")
        file = st.file_uploader("CSV 파일을 선택하세요", label_visibility="collapsed")
        if file:
            df_raw_data = load_data(file)
            if df_raw_data is not None:
                date_col = df_raw_data.columns[0]
                value_col = df_raw_data.select_dtypes(include=np.number).columns[0]
                df_raw_data[date_col] = pd.to_datetime(df_raw_data[date_col])
                df_raw_data = df_raw_data.sort_values(date_col).set_index(date_col)
                
                raw_values = df_raw_data[value_col].copy()
                proc_values = raw_values.interpolate().pipe(hampel_filter).pipe(fft_denoise)
                df_raw_data[value_col] = proc_values
                
                fig_prep = go.Figure()
                fig_prep.add_trace(go.Scatter(x=df_raw_data.index, y=raw_values, name="원본", line=dict(color="gray", width=1), opacity=0.4))
                fig_prep.add_trace(go.Scatter(x=df_raw_data.index, y=proc_values, name="전처리", line=dict(color="royalblue")))
                fig_prep.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_prep, use_container_width=True)

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
