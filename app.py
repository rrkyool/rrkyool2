import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# -----------------------------
# 기본 설정 및 헤더
# -----------------------------
st.set_page_config(layout="wide", page_title="수요 예측 대시보드")

header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("📈 시계열 분석 Project1 수요 예측")
with header_right:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("C321032 박하율")

# 세션 상태 관리
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
if "eval_preds" not in st.session_state:
    st.session_state.eval_preds = {}

# -----------------------------
# 분석 함수들
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

def denoise_series(series, alpha=0.3):
    return series.ewm(alpha=alpha, adjust=False).mean()

def run_stationarity_test(series):
    res = adfuller(series.dropna())
    return {"p_value": res[1], "is_stationary": res[1] < 0.05}

def decompose_series(series, period=12):
    if len(series) < period * 2: return None
    try:
        return seasonal_decompose(series, model='additive', period=period)
    except: return None

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
    # 1. NaN 제거 (ARIMA 오류 방지 핵심)
    train = pd.Series(train).astype(float).dropna() 
    
    # 2. 결과 딕셔너리 초기화 (함수 시작 시점에 위치해야 함)
    results = {'mean': None, 'upper': None, 'lower': None, 'trend_slope': 0}
    
    if len(train) < 5: # 최소 데이터 확인
        last_val = train.iloc[-1] if not train.empty else 0
        results.update({'mean': np.repeat(last_val, horizon), 
                        'upper': np.repeat(last_val, horizon), 
                        'lower': np.repeat(last_val, horizon)})
        return results

    # 추세 계산
    x = np.arange(len(train))
    slope, _ = np.polyfit(x, train.values, 1)
    results['trend_slope'] = slope

    try:
        if model_type == "이동평균":
            val = train.rolling(window=12, min_periods=1).mean().iloc[-1]
            results['mean'] = np.repeat(val, horizon)
            results['upper'] = results['mean'] + (train.std() * 0.5)
            results['lower'] = results['mean'] - (train.std() * 0.5)
        
        elif model_type == "지수평활":
            model = ExponentialSmoothing(train).fit()
            results['mean'] = model.forecast(horizon).values
            results['upper'] = results['mean'] * 1.05
            results['lower'] = results['mean'] * 0.95

        elif model_type == "Holt-Winters":
            # 데이터가 부족할 경우 계절성 제외 로직 추가
            seasonal_p = 12 if len(train) >= 24 else None
            model = ExponentialSmoothing(train, trend="add", seasonal="add" if seasonal_p else None, 
                                         seasonal_periods=seasonal_p).fit()
            forecast = model.forecast(horizon)
            results['mean'] = forecast.values
            results['upper'] = forecast.values + (train.std() * 1.96 / np.sqrt(len(train)))
            results['lower'] = forecast.values - (train.std() * 1.96 / np.sqrt(len(train)))

        elif model_type in ["ARIMA", "SARIMA"]:
            # pmdarima의 auto_arima 사용
            model = auto_arima(train, seasonal=(model_type=="SARIMA"), m=12, 
                               stepwise=True, suppress_warnings=True, 
                               max_p=2, max_q=2, error_action='ignore')
            forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
            results['mean'] = forecast
            results['lower'] = conf_int[:, 0]
            results['upper'] = conf_int[:, 1]
            
    except Exception as e:
        # 에러 발생 시 로그 출력 및 기본값 반환
        print(f"Model Error: {e}") 
        last_val = train.iloc[-1]
        results['mean'] = np.repeat(last_val, horizon)
        results['upper'] = results['mean'] * 1.1
        results['lower'] = results['mean'] * 0.9

    return results

# -----------------------------
# 백테스트 함수들
# -----------------------------
def rolling_forecast(train, test, model_type):
    history = list(train)
    preds = []
    window_size = len(train)
    for t in range(len(test)):
        current_train = history[-window_size:]
        res = get_best_forecast(current_train, 1, model_type)
        preds.append(res['mean'][0])
        history.append(test.iloc[t])
    return np.array(preds)

def expanding_forecast(train, test, model_type):
    preds = []
    for i in range(len(test)):
        hist = pd.concat([train, test[:i]])
        res = get_best_forecast(hist, 1, model_type)
        preds.append(res['mean'][0])
    return np.array(preds)

# -----------------------------
# 메인 레이아웃 시작
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
                # 전처리 파이프라인
                proc_values = raw_values.interpolate().pipe(hampel_filter).pipe(denoise_series)
                df_raw_data['processed'] = proc_values
                
                # 시각화
                fig_prep = go.Figure()
                fig_prep.add_trace(go.Scatter(x=df_raw_data.index, y=raw_values, name="원본", line=dict(color="gray", width=1), opacity=0.4))
                fig_prep.add_trace(go.Scatter(x=df_raw_data.index, y=proc_values, name="전처리", line=dict(color="royalblue")))
                fig_prep.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_prep, use_container_width=True)
                
                # 정상성 검정
                col_stat1, col_stat2 = st.columns(2)
                stat_res = run_stationarity_test(proc_values)
                col_stat1.metric("ADF p-value", f"{stat_res['p_value']:.4f}")
                col_stat2.info("정상성 확보" if stat_res['is_stationary'] else "비정상(차분 권장)")
                
                # 시계열 분해
                dec = decompose_series(proc_values)
                if dec:
                    fig_dec = go.Figure()
                    # 확인: dec.trend는 부드러운 선, dec.seasonal은 반복되는 패턴이어야 합니다.
                    fig_dec.add_trace(go.Scatter(x=dec.trend.index, y=dec.trend, 
                                                 name="Trend(추세)", line=dict(color="royalblue")))
                    fig_dec.add_trace(go.Scatter(x=dec.seasonal.index, y=dec.seasonal, 
                                                 name="Seasonal(계절성)", line=dict(color="lightskyblue")))
                    
                    # 만약 차트에서 여전히 반대로 보인다면, 아래와 같이 데이터 자체를 확인해 보세요.
                    # st.write(dec.trend.head()) 
                    
                    fig_dec.update_layout(height=250, title="시계열 분해 요소", 
                                          margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_dec, use_container_width=True)

with top_right:
    with st.container(border=True):
        st.subheader("⚙️ 모델 선택 및 설정")
        if file and 'df_raw_data' in locals():
            start_date = df_raw_data.index.min().strftime('%Y.%m.%d')
            end_date = df_raw_data.index.max().strftime('%Y.%m.%d')
            st.markdown(f"📅 **범위**: `{start_date} ~ {end_date}` | 샘플: `{len(df_raw_data)}개`")
            st.divider() 
            
        c1, c2 = st.columns(2)
        with c1:
            m_type = st.selectbox("예측 모델", ["이동평균", "지수평활", "Holt-Winters", "ARIMA", "SARIMA"])
            h_len = st.number_input("예측 길이(시평)", 1, 100, 12)
        with c2:
            e_type = st.selectbox("평가 방식", ["Rolling", "Expanding"])
            u_type = st.selectbox("시간 단위", ["일", "월", "년"], index=1)
            
        btn_run = st.button("🚀 예측 실행", use_container_width=True, type="primary")
        if st.button("🗑️ 로그 초기화", use_container_width=True):
            st.session_state.results_df = pd.DataFrame(columns=["모델 종류", "평가 방법", "MAE", "MdRAE", "TS", "예측 평균"])
            st.session_state.eval_preds = {}
            st.rerun()

# -----------------------------
# 분석 실행
# -----------------------------
if file and btn_run:
    df = df_raw_data.copy()
    val_col = 'processed'
    split_idx = int(len(df) * 0.8)
    train_set, test_set = df[val_col].iloc[:split_idx], df[val_col].iloc[split_idx:]
    
    with st.spinner("분석 중..."):
        if e_type == "Rolling":
            test_preds = rolling_forecast(train_set, test_set, m_type)
        else:
            test_preds = expanding_forecast(train_set, test_set, m_type)
            
        eval_key = f"{m_type}_{e_type}"
        st.session_state.eval_preds[eval_key] = test_preds
        
        # 미래 예측
        res_future = get_best_forecast(df[val_col], h_len, m_type)
        avg_f = round(float(np.mean(res_future['mean'])), 1)
        
        new_entry = pd.DataFrame([{
            "모델 종류": m_type, "평가 방법": e_type, 
            "MAE": round(mae(test_set, test_preds), 4),
            "MdRAE": round(mdrae(test_set, test_preds), 4),
            "TS": round(tracking_signal(test_set, test_preds), 4),
            "예측 평균": avg_f
        }])
        st.session_state.results_df = pd.concat([st.session_state.results_df, new_entry], ignore_index=True)

# -----------------------------
# 하단 레이아웃
# -----------------------------
if not st.session_state.results_df.empty:
    df = df_raw_data.copy()
    val_col = 'processed'
    split_idx = int(len(df) * 0.8)
    bot_left, bot_right = st.columns([1, 1.2])
    
    with bot_left:
        with st.container(border=True):
            st.subheader("📏 평가 결과 및 로그")
            st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
            
            actual_y = df[val_col].iloc[split_idx:]
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=actual_y.index, y=actual_y, name="Actual", line=dict(color="green")))
            for name, p_vals in st.session_state.eval_preds.items():
                fig_eval.add_trace(go.Scatter(x=actual_y.index, y=p_vals, name=name, line=dict(dash='dot')))
            fig_eval.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_eval, use_container_width=True)

    with bot_right:
        with st.container(border=True):
            st.subheader("📊 수요 예측 결과")
            last_model = st.session_state.results_df["모델 종류"].iloc[-1]
            res = get_best_forecast(df[val_col], h_len, last_model)
            
            freq_map = {"일": "D", "월": "MS", "년": "YS"}
            future_dates = pd.date_range(df.index[-1], periods=h_len+1, freq=freq_map[u_type])[1:]
            
            fig_all = go.Figure()
            # 과거 데이터
            fig_all.add_trace(go.Scatter(x=df.index, y=df[val_col], name="과거 데이터", line=dict(color="#1f77b4")))
            # 신뢰구간 시각화
            fig_all.add_trace(go.Scatter(x=future_dates, y=res['upper'], fill=None, mode='lines', line_color='rgba(239, 85, 59, 0)', showlegend=False))
            fig_all.add_trace(go.Scatter(x=future_dates, y=res['lower'], fill='tonexty', mode='lines', line_color='rgba(239, 85, 59, 0)', fillcolor='rgba(239, 85, 59, 0.2)', name="95% 신뢰구간"))
            # 예측값
            fig_all.add_trace(go.Scatter(x=future_dates, y=res['mean'], name="미래 예측", line=dict(color="#ef553b", width=3)))
            
            fig_all.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_all, use_container_width=True)
            
            trend_txt = "상승" if res['trend_slope'] > 0 else "하락"
            st.success(f"""
            ### 📋 상세 예측 리포트 ({last_model})
            - **장기 추세**: 현재 데이터는 전반적으로 **{trend_txt}** 추세에 있습니다.
            - **신뢰 구간**: 예측값은 약 **{res['lower'].mean():.1f} ~ {res['upper'].mean():.1f}** 사이에서 변동할 가능성이 높습니다.
            """)
            
elif file:
    st.info("👈 설정 후 '예측 실행' 버튼을 눌러 분석을 시작하세요.")
