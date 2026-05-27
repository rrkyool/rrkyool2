# app.py
# ------------------------------------------------------------
# 다변량 시계열 이상탐지 Streamlit 웹앱
# - CSV 업로드 시 자동 분석/전처리/이상탐지 수행
# - 파일 변경 시 fingerprint 기반으로 세션 초기화 후 재탐지
# - sktime, pmdarima 없이 pandas/numpy/scipy/scikit-learn/statsmodels/plotly 사용
# ------------------------------------------------------------

import hashlib
import io
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0. 페이지 설정 / 공통 스타일
# ------------------------------------------------------------
st.set_page_config(
    page_title="시계열분석 Project2 이상 탐지",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #eeeeee;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.035);
    }
    .small-help {color:#666; font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

RANDOM_STATE = 42


# ------------------------------------------------------------
# 1. 데이터 로드 / 파일 변경 감지
# ------------------------------------------------------------
def get_file_fingerprint(uploaded_file) -> str:
    """업로드 파일의 내용 기반 fingerprint. 파일이 바뀌면 자동 재분석하기 위해 사용."""
    bytes_data = uploaded_file.getvalue()
    return hashlib.md5(bytes_data).hexdigest()


def load_csv(uploaded_file) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    last_error = None
    raw = uploaded_file.getvalue()

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            raise ValueError(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")

    raise ValueError(f"지원되지 않는 인코딩입니다. 마지막 오류: {last_error}")


def reset_state_for_new_file(file_hash: str):
    keep_keys = {"file_hash"}
    for key in list(st.session_state.keys()):
        if key not in keep_keys:
            del st.session_state[key]
    st.session_state["file_hash"] = file_hash


# ------------------------------------------------------------
# 2. 시계열 컬럼 자동 인식 / 전처리
# ------------------------------------------------------------
def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """날짜/시간 컬럼 자동 추정. 실패하면 None."""
    candidate_scores = []

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce")
        else:
            parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

        valid_ratio = parsed.notna().mean()
        unique_ratio = parsed.nunique(dropna=True) / max(1, len(parsed))
        name_bonus = 0.15 if any(k in str(col).lower() for k in ["date", "time", "일자", "날짜", "시각", "timestamp"]) else 0
        score = valid_ratio + unique_ratio + name_bonus
        candidate_scores.append((score, valid_ratio, col))

    candidate_scores.sort(reverse=True)
    if candidate_scores and candidate_scores[0][1] >= 0.7:
        return candidate_scores[0][2]
    return None


def prepare_time_dataframe(df: pd.DataFrame, time_col: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    """시계열 인덱스 설정, 숫자 컬럼 추출, 결측 보간."""
    meta = {}
    work = df.copy()

    if time_col and time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).sort_values(time_col)
        work = work.drop_duplicates(subset=[time_col], keep="first")
        work = work.set_index(time_col)
        meta["time_col"] = time_col
        meta["has_datetime_index"] = True
    else:
        work.index = pd.RangeIndex(start=0, stop=len(work), step=1, name="row")
        meta["time_col"] = "행 번호(row index)"
        meta["has_datetime_index"] = False

    numeric = work.select_dtypes(include=[np.number]).copy()
    # 숫자로 읽히지 않은 컬럼 중 변환 가능한 컬럼 복구
    for col in work.columns:
        if col not in numeric.columns:
            converted = pd.to_numeric(work[col], errors="coerce")
            if converted.notna().mean() >= 0.8:
                numeric[col] = converted

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.empty:
        raise ValueError("분석 가능한 숫자형 컬럼이 없습니다. CSV에 수치형 시계열 변수가 필요합니다.")

    # 결측치 처리: 시간 흐름 기준 보간 후 앞/뒤 채움
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    numeric = numeric.ffill().bfill()

    # 상수 컬럼 제거
    nunique = numeric.nunique(dropna=True)
    numeric = numeric.loc[:, nunique > 1]
    if numeric.empty:
        raise ValueError("모든 숫자형 컬럼이 상수입니다. 이상탐지를 수행할 변동성이 부족합니다.")

    if isinstance(numeric.index, pd.DatetimeIndex) and len(numeric.index) > 2:
        inferred = pd.infer_freq(numeric.index)
        if inferred:
            numeric = numeric.asfreq(inferred)
            numeric = numeric.interpolate(method="linear", limit_direction="both").ffill().bfill()
            meta["inferred_freq"] = inferred
        else:
            meta["inferred_freq"] = "불규칙/판단 불가"
    else:
        meta["inferred_freq"] = "행 단위"

    meta["n_rows"] = len(numeric)
    meta["n_features"] = numeric.shape[1]
    meta["start"] = numeric.index.min()
    meta["end"] = numeric.index.max()
    return numeric, meta


def robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler()
    values = scaler.fit_transform(df.values)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def make_feature_matrix(df: pd.DataFrame, rolling_window: int, include_rolling: bool) -> pd.DataFrame:
    """원본 변수 + 변화량 + rolling 통계 기반 feature matrix."""
    base = df.copy()
    pieces = [base.add_suffix("__level")]

    diff = df.diff().fillna(0)
    pieces.append(diff.add_suffix("__diff"))

    pct = df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    pieces.append(pct.add_suffix("__pct"))

    if include_rolling and rolling_window >= 3:
        roll_mean = df.rolling(rolling_window, min_periods=max(2, rolling_window // 3)).mean().bfill()
        roll_std = df.rolling(rolling_window, min_periods=max(2, rolling_window // 3)).std().fillna(0)
        residual = df - roll_mean
        pieces.extend([
            roll_mean.add_suffix("__roll_mean"),
            roll_std.add_suffix("__roll_std"),
            residual.add_suffix("__roll_resid"),
        ])

    feature_df = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    return feature_df


# ------------------------------------------------------------
# 3. 이상탐지 모델
# ------------------------------------------------------------
def score_isolation_forest(features: pd.DataFrame, contamination: float) -> np.ndarray:
    scaler = StandardScaler()
    x = scaler.fit_transform(features.values)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x)
    # decision_function: 클수록 정상. 반전해서 클수록 이상.
    return -model.decision_function(x)


def score_robust_z(df: pd.DataFrame) -> np.ndarray:
    scaled = robust_scale(df)
    # 다변량 관점: 각 시점에서 가장 튀는 변수의 robust z-score
    return scaled.abs().max(axis=1).values


def score_pca_reconstruction(features: pd.DataFrame, variance_keep: float = 0.90) -> np.ndarray:
    scaler = StandardScaler()
    x = scaler.fit_transform(features.values)
    max_components = min(x.shape[0], x.shape[1])
    if max_components < 2:
        return np.zeros(x.shape[0])

    pca_full = PCA(n_components=max_components, random_state=RANDOM_STATE)
    pca_full.fit(x)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumsum, variance_keep) + 1)
    n_components = min(max(1, n_components), max_components)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    z = pca.fit_transform(x)
    x_hat = pca.inverse_transform(z)
    return np.mean((x - x_hat) ** 2, axis=1)


def minmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)


def detect_anomalies(
    df: pd.DataFrame,
    method: str,
    contamination: float,
    rolling_window: int,
    include_rolling: bool,
    threshold_mode: str,
    manual_quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = make_feature_matrix(df, rolling_window, include_rolling)

    scores = {}
    if method in ["Isolation Forest", "Ensemble"]:
        scores["Isolation Forest"] = minmax(score_isolation_forest(features, contamination))
    if method in ["Robust Z-Score", "Ensemble"]:
        scores["Robust Z-Score"] = minmax(score_robust_z(df))
    if method in ["PCA Reconstruction", "Ensemble"]:
        scores["PCA Reconstruction"] = minmax(score_pca_reconstruction(features))

    if method == "Ensemble":
        final_score = np.mean(np.vstack(list(scores.values())), axis=0)
    else:
        final_score = list(scores.values())[0]

    if threshold_mode == "상위 비율(contamination) 자동":
        threshold = np.quantile(final_score, 1 - contamination)
    else:
        threshold = np.quantile(final_score, manual_quantile / 100)

    is_anomaly = final_score >= threshold

    result = pd.DataFrame({
        "anomaly_score": final_score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
    }, index=df.index)

    score_detail = pd.DataFrame(scores, index=df.index)
    score_detail["Final Score"] = final_score
    return result, score_detail


# ------------------------------------------------------------
# 4. 평가/진단 지표
# ------------------------------------------------------------
def calc_stationarity_summary(df: pd.DataFrame, max_cols: int = 12) -> pd.DataFrame:
    rows = []
    for col in df.columns[:max_cols]:
        s = df[col].dropna()
        if len(s) < 12:
            rows.append({"변수": col, "ADF p-value": np.nan, "정상성": "데이터 부족"})
            continue
        try:
            p = adfuller(s, autolag="AIC")[1]
            rows.append({"변수": col, "ADF p-value": p, "정상성": "정상" if p < 0.05 else "비정상"})
        except Exception:
            rows.append({"변수": col, "ADF p-value": np.nan, "정상성": "계산 실패"})
    return pd.DataFrame(rows)


def calc_ljungbox_summary(score: pd.Series) -> Dict:
    s = score.dropna()
    lag = min(12, max(1, len(s) // 5))
    try:
        p = acorr_ljungbox(s, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]
        return {"lag": lag, "p_value": p, "has_autocorr": p < 0.05}
    except Exception:
        return {"lag": lag, "p_value": np.nan, "has_autocorr": False}


def feature_contribution(df: pd.DataFrame, anomaly_mask: pd.Series) -> pd.DataFrame:
    scaled = robust_scale(df).abs()
    if anomaly_mask.sum() == 0:
        contrib = scaled.mean().sort_values(ascending=False)
    else:
        contrib = scaled.loc[anomaly_mask].mean().sort_values(ascending=False)
    return contrib.rename("평균 이상 기여도").reset_index().rename(columns={"index": "변수"})


def make_pseudo_labels(df: pd.DataFrame, top_ratio: float = 0.05) -> pd.Series:
    """정답 라벨이 없는 과제 상황에서 대시보드용 참고 라벨 생성.
    실제 성능지표가 아니라, 변수별 robust deviation 기준의 참고 벤치마크임.
    """
    base_score = score_robust_z(df)
    th = np.quantile(base_score, 1 - top_ratio)
    return pd.Series(base_score >= th, index=df.index)


def classification_metrics(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict:
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(y_true & pred))
    fp = int(np.sum(~y_true & pred))
    fn = int(np.sum(y_true & ~pred))
    tn = int(np.sum(~y_true & ~pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    try:
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = np.nan
    try:
        pr, rc, _ = precision_recall_curve(y_true, score)
        pr_auc = auc(rc, pr)
    except Exception:
        pr_auc = np.nan

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision, "Recall": recall, "F1": f1,
        "ROC-AUC": roc_auc, "PR-AUC": pr_auc,
    }


# ------------------------------------------------------------
# 5. 시각화 함수
# ------------------------------------------------------------
def plot_multivariate_series(df: pd.DataFrame, result: pd.DataFrame, selected_cols: List[str]) -> go.Figure:
    fig = go.Figure()
    for col in selected_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col, opacity=0.8))

    anomalies = result[result["is_anomaly"]]
    if len(anomalies) > 0 and selected_cols:
        y_top = df[selected_cols].max(axis=1)
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=y_top.loc[anomalies.index],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(size=9, symbol="x", color="#d62728"),
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def plot_score(result: pd.DataFrame, score_detail: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in score_detail.columns:
        width = 4 if col == "Final Score" else 1.5
        fig.add_trace(go.Scatter(x=score_detail.index, y=score_detail[col], name=col, mode="lines", line=dict(width=width)))
    fig.add_trace(go.Scatter(
        x=result.index,
        y=result["threshold"],
        name="Threshold",
        mode="lines",
        line=dict(dash="dash", color="#d62728", width=2),
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def plot_score_distribution(result: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result["anomaly_score"], nbinsx=40, name="Score Distribution", opacity=0.8))
    fig.add_vline(x=float(result["threshold"].iloc[0]), line_dash="dash", line_color="#d62728", annotation_text="threshold")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig


def plot_contribution(contrib_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    top = contrib_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["평균 이상 기여도"], y=top["변수"], orientation="h"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_confusion(metrics: Dict) -> go.Figure:
    z = np.array([[metrics["TP"], metrics["FN"]], [metrics["FP"], metrics["TN"]]])
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Pred Anomaly", "Pred Normal"],
        y=["Ref Anomaly", "Ref Normal"],
        text=z,
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------
# 6. 사이드바: 입력/설정
# ------------------------------------------------------------
if "file_hash" not in st.session_state:
    st.session_state["file_hash"] = None

with st.sidebar:
    st.title("분석 설정")

    with st.container(border=True):
        st.subheader("📂 CSV 업로드")
        uploaded_file = st.file_uploader("다변량 시계열 CSV 파일", type=["csv"])

    if uploaded_file is not None:
        current_hash = get_file_fingerprint(uploaded_file)
        if st.session_state.get("file_hash") != current_hash:
            reset_state_for_new_file(current_hash)

        raw_df = load_csv(uploaded_file)
        auto_time_col = find_datetime_column(raw_df)
        all_cols = ["자동 감지"] + list(raw_df.columns) + ["사용 안 함"]

        with st.container(border=True):
            st.subheader("🧭 시간 컬럼")
            default_idx = 0
            time_choice = st.selectbox("날짜/시간 컬럼 선택", all_cols, index=default_idx)
            if time_choice == "자동 감지":
                time_col = auto_time_col
                st.caption(f"자동 감지 결과: {auto_time_col if auto_time_col else '없음 → 행 번호 사용'}")
            elif time_choice == "사용 안 함":
                time_col = None
            else:
                time_col = time_choice

        ts_df, meta = prepare_time_dataframe(raw_df, time_col)

        with st.container(border=True):
            st.subheader("⚙️ 이상탐지 설정")
            method = st.selectbox(
                "탐지 알고리즘",
                ["Ensemble", "Isolation Forest", "Robust Z-Score", "PCA Reconstruction"],
                help="Ensemble은 Isolation Forest, Robust Z-Score, PCA 재구성오차를 평균하여 더 안정적으로 탐지합니다.",
            )
            contamination = st.slider("예상 이상 비율", 0.005, 0.30, 0.05, 0.005, format="%.3f")
            threshold_mode = st.radio("임계값 방식", ["상위 비율(contamination) 자동", "수동 분위수"], horizontal=False)
            manual_q = st.slider("수동 임계 분위수(%)", 80, 99, 95, 1, disabled=threshold_mode.startswith("상위"))
            rolling_window = st.number_input("Rolling window", min_value=3, max_value=max(3, min(200, len(ts_df)//2)), value=min(24, max(3, len(ts_df)//10)), step=1)
            include_rolling = st.checkbox("Rolling 통계 feature 포함", value=True)

        with st.container(border=True):
            st.subheader("📊 시각화 설정")
            selected_cols = st.multiselect(
                "그래프에 표시할 변수",
                list(ts_df.columns),
                default=list(ts_df.columns[: min(4, len(ts_df.columns))]),
            )
            top_n = st.slider("기여도 표시 변수 수", 5, min(30, len(ts_df.columns)), min(12, len(ts_df.columns)))

        if st.button("이상탐지 실행 / 새로고침", use_container_width=True, type="primary"):
            st.session_state["force_run"] = True
    else:
        raw_df = None
        ts_df = None
        meta = None


# ------------------------------------------------------------
# 7. 메인 화면
# ------------------------------------------------------------
st.title("📈 시계열분석 Project2 이상 탐지")
st.subheader("C321032 박하율")
st.divider()

if uploaded_file is None:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하면 자동으로 분석이 시작됩니다.")
    st.markdown(
        """
        **권장 CSV 형태**  
        - 첫 번째 또는 특정 컬럼에 날짜/시간 값이 있으면 자동으로 시간 인덱스로 사용합니다.  
        - 나머지 수치형 컬럼들은 모두 다변량 이상탐지 대상이 됩니다.  
        - 정답 라벨이 있으면 일반 변수로 인식될 수 있으므로, 필요 시 라벨 컬럼은 업로드 전 제외하는 것을 권장합니다.
        """
    )
    st.stop()

try:
    with st.spinner("데이터 전처리 및 이상탐지를 수행하는 중입니다..."):
        result, score_detail = detect_anomalies(
            ts_df,
            method=method,
            contamination=contamination,
            rolling_window=int(rolling_window),
            include_rolling=include_rolling,
            threshold_mode=threshold_mode,
            manual_quantile=manual_q,
        )
        contrib_df = feature_contribution(ts_df, result["is_anomaly"])
        lb = calc_ljungbox_summary(result["anomaly_score"])
        stat_df = calc_stationarity_summary(ts_df)
        pseudo_y = make_pseudo_labels(ts_df, top_ratio=contamination)
        metrics = classification_metrics(pseudo_y.values, result["anomaly_score"].values, result["is_anomaly"].values)
except Exception as e:
    st.error(f"분석 중 오류가 발생했습니다: {e}")
    st.stop()

# 1행: 핵심 요약
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("데이터 수", f"{meta['n_rows']:,}")
m2.metric("변수 수", f"{meta['n_features']:,}")
m3.metric("탐지 이상 수", f"{int(result['is_anomaly'].sum()):,}", f"{result['is_anomaly'].mean()*100:.2f}%")
m4.metric("평균 이상점수", f"{result['anomaly_score'].mean():.3f}")
m5.metric("임계값", f"{float(result['threshold'].iloc[0]):.3f}")

st.caption(
    f"시간 컬럼: {meta['time_col']} | 빈도: {meta['inferred_freq']} | 기간: {meta['start']} ~ {meta['end']}"
)
st.divider()

# 2행: 원시 시계열과 score
left, right = st.columns([1.45, 1])
with left:
    st.subheader("탐지 결과 시계열")
    with st.container(border=True):
        if selected_cols:
            st.plotly_chart(plot_multivariate_series(ts_df, result, selected_cols), use_container_width=True)
        else:
            st.warning("사이드바에서 표시할 변수를 1개 이상 선택해 주세요.")

with right:
    st.subheader("이상 점수 추이")
    with st.container(border=True):
        st.plotly_chart(plot_score(result, score_detail), use_container_width=True)

# 3행: 평가 대시보드
st.divider()
st.subheader("평가지표 기반 판단 대시보드")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision*", f"{metrics['Precision']:.3f}")
c2.metric("Recall*", f"{metrics['Recall']:.3f}")
c3.metric("F1*", f"{metrics['F1']:.3f}")
c4.metric("PR-AUC*", f"{metrics['PR-AUC']:.3f}" if not np.isnan(metrics['PR-AUC']) else "N/A")

st.markdown(
    "<div class='small-help'>* 정답 라벨이 없는 일반 CSV를 전제로 하므로, 위 평가지표는 변수별 Robust deviation 상위 구간을 참고 라벨로 둔 내부 벤치마크입니다. 실제 라벨이 있는 데이터라면 라벨 컬럼을 분리하여 같은 구조로 확장할 수 있습니다.</div>",
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns([1, 1, 1])
with p1:
    with st.container(border=True):
        st.markdown("#### 점수 분포와 임계값")
        st.plotly_chart(plot_score_distribution(result), use_container_width=True)
with p2:
    with st.container(border=True):
        st.markdown("#### 참고 혼동행렬")
        st.plotly_chart(plot_confusion(metrics), use_container_width=True)
with p3:
    with st.container(border=True):
        st.markdown("#### 점수 자기상관 진단")
        st.metric(
            f"Ljung-Box p-value(lag={lb['lag']})",
            f"{lb['p_value']:.4f}" if not np.isnan(lb["p_value"]) else "N/A",
            "점수 패턴 존재" if lb["has_autocorr"] else "독립적 탐지에 가까움",
        )
        st.write("이상 점수가 특정 구간에 몰려 반복적으로 높다면, 단발성 이상이 아니라 구조적 변화나 계절성 미반영일 수 있습니다.")

# 4행: 원인 변수/상관관계
st.divider()
col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("이상 발생 시 주요 기여 변수")
    with st.container(border=True):
        st.plotly_chart(plot_contribution(contrib_df, top_n=top_n), use_container_width=True)
        st.dataframe(contrib_df.head(top_n), use_container_width=True, hide_index=True)
with col_b:
    st.subheader("변수 간 상관관계")
    with st.container(border=True):
        corr_cols = ts_df.columns[: min(25, len(ts_df.columns))]
        st.plotly_chart(plot_correlation_heatmap(ts_df[corr_cols]), use_container_width=True)
        if len(ts_df.columns) > 25:
            st.caption("변수가 많아 상위 25개 컬럼만 표시했습니다.")

# 5행: 통계 진단 / 이상 목록 다운로드
st.divider()
st.subheader("통계 진단 및 이상 목록")

d1, d2 = st.columns([1, 1.4])
with d1:
    with st.container(border=True):
        st.markdown("#### 변수별 정상성 진단(ADF)")
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
        non_stationary = (stat_df["정상성"] == "비정상").sum()
        if non_stationary > 0:
            st.warning(f"비정상으로 판단된 변수가 {non_stationary}개 있습니다. 추세/계절성이 강한 데이터에서는 rolling feature 또는 PCA 기반 점수를 함께 보는 것이 좋습니다.")
        else:
            st.success("표시된 변수 기준으로 정상성이 비교적 양호합니다.")

with d2:
    with st.container(border=True):
        st.markdown("#### 탐지된 이상 시점")
        out = ts_df.copy()
        out.insert(0, "is_anomaly", result["is_anomaly"].values)
        out.insert(0, "anomaly_score", result["anomaly_score"].values)
        anomalies_only = out[out["is_anomaly"]].sort_values("anomaly_score", ascending=False)
        st.dataframe(anomalies_only, use_container_width=True, height=330)

        csv = out.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "전체 결과 CSV 다운로드",
            data=csv,
            file_name="anomaly_detection_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

# 6행: 해석 가이드
st.divider()
with st.expander("📌 결과 해석 가이드", expanded=True):
    st.markdown(
        """
        - **탐지 이상 수가 너무 많으면** 예상 이상 비율(contamination)을 낮추거나 임계값을 수동 분위수 97~99%로 올려 보세요.  
        - **이상 점수가 넓은 구간에서 계속 높으면** 단발 이상보다 구조 변화, 추세 변화, 계절성 미반영 가능성이 큽니다.  
        - **기여도 상위 변수가 특정 변수에 몰리면** 해당 센서/수요/지표의 급등락이 이상 판단을 주도한 것입니다.  
        - **Precision/Recall/F1은 참고용**입니다. 실제 라벨이 없는 CSV에서도 판단을 돕기 위해 내부 기준 라벨을 생성한 것이며, 과제 보고서에는 이 한계를 명시하는 것이 좋습니다.
        """
    )
