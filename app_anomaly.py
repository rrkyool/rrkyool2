# app.py
# ------------------------------------------------------------
# 다변량 시계열 이상탐지 Streamlit 웹앱
# 핵심 구조:
# 1. 데이터 기본 분석
# 2. 이상탐지 결과 분석 및 시각화
# 3. 평가지표 및 시각화
# ------------------------------------------------------------

import hashlib
import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# ------------------------------------------------------------
# 0. 페이지 설정 / 스타일
# ------------------------------------------------------------
st.set_page_config(
    page_title="다변량 시계열 이상탐지 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 13px 15px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.035);
    }
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.035);
    }
    .small-help {
        color: #666;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 1. 파일 로드 / 변경 감지
# ------------------------------------------------------------
def get_file_fingerprint(uploaded_file) -> str:
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def load_csv(uploaded_file) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    raw = uploaded_file.getvalue()
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            raise ValueError(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")

    raise ValueError(f"지원되지 않는 인코딩입니다. 마지막 오류: {last_error}")


def reset_state_for_new_file(file_hash: str):
    for key in list(st.session_state.keys()):
        if key != "file_hash":
            del st.session_state[key]
    st.session_state["file_hash"] = file_hash


# ------------------------------------------------------------
# 2. 시간 컬럼 자동 감지 / 전처리
# ------------------------------------------------------------
def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidate_scores = []

    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid_ratio = parsed.notna().mean()
        unique_ratio = parsed.nunique(dropna=True) / max(1, len(parsed))

        name_bonus = 0.15 if any(
            k in str(col).lower()
            for k in ["date", "time", "datetime", "timestamp", "일자", "날짜", "시간", "시각"]
        ) else 0

        score = valid_ratio + unique_ratio + name_bonus
        candidate_scores.append((score, valid_ratio, col))

    candidate_scores.sort(reverse=True)

    if candidate_scores and candidate_scores[0][1] >= 0.7:
        return candidate_scores[0][2]

    return None


def prepare_time_dataframe(df: pd.DataFrame, time_col: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    meta = {}
    work = df.copy()

    if time_col and time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col])
        work = work.sort_values(time_col)
        work = work.drop_duplicates(subset=[time_col], keep="first")
        work = work.set_index(time_col)
        meta["time_col"] = time_col
        meta["has_datetime_index"] = True
    else:
        work.index = pd.RangeIndex(start=0, stop=len(work), step=1, name="row")
        meta["time_col"] = "자동 감지 실패 → 행 번호 사용"
        meta["has_datetime_index"] = False

    numeric = work.select_dtypes(include=[np.number]).copy()

    for col in work.columns:
        if col not in numeric.columns:
            converted = pd.to_numeric(work[col], errors="coerce")
            if converted.notna().mean() >= 0.8:
                numeric[col] = converted

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.empty:
        raise ValueError("분석 가능한 숫자형 컬럼이 없습니다.")

    numeric = numeric.interpolate(method="linear", limit_direction="both")
    numeric = numeric.ffill().bfill()

    nunique = numeric.nunique(dropna=True)
    numeric = numeric.loc[:, nunique > 1]

    if numeric.empty:
        raise ValueError("모든 숫자형 컬럼이 상수입니다.")

    if isinstance(numeric.index, pd.DatetimeIndex) and len(numeric.index) > 2:
        inferred = pd.infer_freq(numeric.index)
        if inferred:
            numeric = numeric.asfreq(inferred)
            numeric = numeric.interpolate(method="linear", limit_direction="both").ffill().bfill()
            meta["inferred_freq"] = inferred
        else:
            meta["inferred_freq"] = "불규칙 / 판단 불가"
    else:
        meta["inferred_freq"] = "행 단위"

    meta["n_rows"] = len(numeric)
    meta["n_features"] = numeric.shape[1]
    meta["start"] = numeric.index.min()
    meta["end"] = numeric.index.max()

    return numeric, meta


# ------------------------------------------------------------
# 3. Feature 생성 / 이상탐지
# ------------------------------------------------------------
def robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler()
    values = scaler.fit_transform(df.values)
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def make_feature_matrix(df: pd.DataFrame, rolling_window: int, include_rolling: bool) -> pd.DataFrame:
    pieces = []

    pieces.append(df.add_suffix("__level"))

    diff = df.diff().fillna(0)
    pieces.append(diff.add_suffix("__diff"))

    pct = df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    pieces.append(pct.add_suffix("__pct"))

    if include_rolling and rolling_window >= 3:
        roll_mean = df.rolling(
            rolling_window,
            min_periods=max(2, rolling_window // 3)
        ).mean().bfill()

        roll_std = df.rolling(
            rolling_window,
            min_periods=max(2, rolling_window // 3)
        ).std().fillna(0)

        residual = df - roll_mean

        pieces.append(roll_mean.add_suffix("__roll_mean"))
        pieces.append(roll_std.add_suffix("__roll_std"))
        pieces.append(residual.add_suffix("__roll_resid"))

    feature_df = pd.concat(pieces, axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feature_df


def minmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)


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

    return -model.decision_function(x)


def score_robust_z(df: pd.DataFrame) -> np.ndarray:
    scaled = robust_scale(df)
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

    if threshold_mode == "자동":
        threshold = np.quantile(final_score, 1 - contamination)
    else:
        threshold = np.quantile(final_score, manual_quantile / 100)

    is_anomaly = final_score >= threshold

    result = pd.DataFrame(
        {
            "anomaly_score": final_score,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
        },
        index=df.index,
    )

    score_detail = pd.DataFrame(scores, index=df.index)
    score_detail["Final Score"] = final_score

    return result, score_detail


# ------------------------------------------------------------
# 4. 통계 진단 / 평가 지표
# ------------------------------------------------------------
def calc_stationarity_summary(df: pd.DataFrame, max_cols: int = 15) -> pd.DataFrame:
    rows = []

    for col in df.columns[:max_cols]:
        s = df[col].dropna()

        if len(s) < 12:
            rows.append(
                {
                    "변수": col,
                    "ADF p-value": np.nan,
                    "판정": "데이터 부족",
                }
            )
            continue

        try:
            p = adfuller(s, autolag="AIC")[1]
            rows.append(
                {
                    "변수": col,
                    "ADF p-value": p,
                    "판정": "정상성 있음" if p < 0.05 else "비정상 가능",
                }
            )
        except Exception:
            rows.append(
                {
                    "변수": col,
                    "ADF p-value": np.nan,
                    "판정": "계산 실패",
                }
            )

    return pd.DataFrame(rows)


def calc_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    pairs = []

    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]

            if abs(value) >= threshold:
                pairs.append(
                    {
                        "변수 1": cols[i],
                        "변수 2": cols[j],
                        "상관계수": value,
                        "절대값": abs(value),
                    }
                )

    out = pd.DataFrame(pairs)

    if out.empty:
        return pd.DataFrame(columns=["변수 1", "변수 2", "상관계수", "절대값"])

    return out.sort_values("절대값", ascending=False)


def calc_ljungbox_summary(score: pd.Series) -> Dict:
    s = score.dropna()
    lag = min(12, max(1, len(s) // 5))

    try:
        p = acorr_ljungbox(s, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]
        return {
            "lag": lag,
            "p_value": p,
            "has_autocorr": p < 0.05,
        }
    except Exception:
        return {
            "lag": lag,
            "p_value": np.nan,
            "has_autocorr": False,
        }


def feature_contribution(df: pd.DataFrame, anomaly_mask: pd.Series) -> pd.DataFrame:
    scaled = robust_scale(df).abs()

    if anomaly_mask.sum() == 0:
        contrib = scaled.mean().sort_values(ascending=False)
    else:
        contrib = scaled.loc[anomaly_mask].mean().sort_values(ascending=False)

    return (
        contrib.rename("평균 이상 기여도")
        .reset_index()
        .rename(columns={"index": "변수"})
    )


def make_pseudo_labels(df: pd.DataFrame, top_ratio: float = 0.05) -> pd.Series:
    base_score = score_robust_z(df)
    threshold = np.quantile(base_score, 1 - top_ratio)

    return pd.Series(base_score >= threshold, index=df.index)


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
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


# ------------------------------------------------------------
# 5. 시각화 함수
# ------------------------------------------------------------
def plot_multivariate_series(df: pd.DataFrame, result: pd.DataFrame, selected_cols: List[str]) -> go.Figure:
    fig = go.Figure()

    for col in selected_cols:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                opacity=0.8,
            )
        )

    anomalies = result[result["is_anomaly"]]

    if len(anomalies) > 0 and selected_cols:
        y_top = df[selected_cols].max(axis=1)

        fig.add_trace(
            go.Scatter(
                x=anomalies.index,
                y=y_top.loc[anomalies.index],
                mode="markers",
                name="Detected Anomaly",
                marker=dict(size=9, symbol="x", color="#d62728"),
            )
        )

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
        fig.add_trace(
            go.Scatter(
                x=score_detail.index,
                y=score_detail[col],
                name=col,
                mode="lines",
                line=dict(width=4 if col == "Final Score" else 1.5),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=result.index,
            y=result["threshold"],
            name="Threshold",
            mode="lines",
            line=dict(dash="dash", color="#d62728", width=2),
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    return fig


def plot_score_distribution(result: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=result["anomaly_score"],
            nbinsx=40,
            name="Score Distribution",
            opacity=0.8,
        )
    )

    fig.add_vline(
        x=float(result["threshold"].iloc[0]),
        line_dash="dash",
        line_color="#d62728",
        annotation_text="threshold",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )

    return fig


def plot_contribution(contrib_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    top = contrib_df.head(top_n).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=top["평균 이상 기여도"],
            y=top["변수"],
            orientation="h",
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


def plot_confusion(metrics: Dict) -> go.Figure:
    z = np.array(
        [
            [metrics["TP"], metrics["FN"]],
            [metrics["FP"], metrics["TN"]],
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred Anomaly", "Pred Normal"],
            y=["Ref Anomaly", "Ref Normal"],
            text=z,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


# ------------------------------------------------------------
# 7. 메인
# ------------------------------------------------------------

if "file_hash" not in st.session_state:
    st.session_state["file_hash"] = None


# 제목
st.title("다변량 시계열 이상탐지 대시보드")
st.markdown("#### C321032 박하율")

# ------------------------------------------------------------
# 상단 통합 Control Panel
# ------------------------------------------------------------
with st.container(border=True):

    st.markdown("### 데이터 업로드 & 분석 조건 설정")

    left_panel, right_panel = st.columns([1.15, 1])

    # ========================================================
    # LEFT : 1. 데이터 업로드 / 2. 분석 설정
    # ========================================================
    with left_panel:

        # ----------------------------------------------------
        # 1. 데이터 업로드
        # ----------------------------------------------------
        with st.container(border=True, height=200):
            uploaded_file = st.file_uploader(
                "CSV 파일 업로드",
                type=["csv"],
                help="파일이 변경되면 자동으로 새로운 이상탐지를 수행합니다.",
            )

            if uploaded_file is None:
                st.info("CSV 파일을 업로드하면 분석 설정과 세부 설정이 활성화됩니다.")
                st.stop()

        # ----------------------------------------------------
        # 파일 업로드 후 데이터 로드
        # ----------------------------------------------------
        try:
            current_hash = get_file_fingerprint(uploaded_file)

            if st.session_state.get("file_hash") != current_hash:
                reset_state_for_new_file(current_hash)

            raw_df = load_csv(uploaded_file)

            auto_time_col = find_datetime_column(raw_df)

            ts_df, meta = prepare_time_dataframe(raw_df, auto_time_col)

            max_window = max(3, min(200, len(ts_df) // 2))
            default_window = min(24, max(3, len(ts_df) // 10))

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
            st.stop()

        # ----------------------------------------------------
        # 2. 분석 설정
        # ----------------------------------------------------
        with st.container(border=True, height=240):
            st.markdown("#### 분석 설정")

            c1, c2 = st.columns(2)

            with c1:
                method = st.selectbox(
                    "탐지 알고리즘",
                    [
                        "Ensemble",
                        "Isolation Forest",
                        "Robust Z-Score",
                        "PCA Reconstruction",
                    ],
                )

                threshold_mode = st.radio(
                    "임계값 방식",
                    ["자동", "수동 분위수"],
                    horizontal=True,
                )

            with c2:
                contamination = st.number_input(
                    "예상 이상 비율",
                    min_value=0.001,
                    max_value=0.300,
                    value=0.050,
                    step=0.005,
                    format="%.3f",
                )

                manual_q = st.number_input(
                    "수동 분위수(%)",
                    min_value=50.0,
                    max_value=99.9,
                    value=95.0,
                    step=0.1,
                    disabled=(threshold_mode == "자동"),
                )

    # ========================================================
    # RIGHT : 3. 세부 설정
    # ========================================================
    with right_panel:

        with st.container(border=True, height=455):
            st.markdown("#### 세부 설정")

            c3, c4 = st.columns(2)

            with c3:
                rolling_window = st.number_input(
                    "Rolling window",
                    min_value=3,
                    max_value=max_window,
                    value=default_window,
                    step=1,
                )

            with c4:
                corr_threshold = st.number_input(
                    "상관관계 표시 기준",
                    min_value=0.1,
                    max_value=0.99,
                    value=0.7,
                    step=0.05,
                )

            include_rolling = st.checkbox(
                "Rolling 통계 feature 포함",
                value=True,
            )

            selected_cols = st.multiselect(
                "시계열 그래프 표시 변수",
                list(ts_df.columns),
                default=list(ts_df.columns[: min(4, len(ts_df.columns))]),
            )

            top_n = st.number_input(
                "기여도 변수 수",
                min_value=3,
                max_value=min(30, len(ts_df.columns)),
                value=min(12, len(ts_df.columns)),
                step=1,
            )

st.divider()

# ------------------------------------------------------------
# 이상탐지 실행
# ------------------------------------------------------------
try:
    with st.spinner("이상탐지 수행 중..."):
        result, score_detail = detect_anomalies(
            ts_df,
            method=method,
            contamination=contamination,
            rolling_window=int(rolling_window),
            include_rolling=include_rolling,
            threshold_mode=threshold_mode,
            manual_quantile=manual_q,
        )

        stat_df = calc_stationarity_summary(ts_df)
        high_corr_df = calc_high_corr_pairs(ts_df, threshold=corr_threshold)
        contrib_df = feature_contribution(ts_df, result["is_anomaly"])
        lb = calc_ljungbox_summary(result["anomaly_score"])

        pseudo_y = make_pseudo_labels(ts_df, top_ratio=contamination)
        metrics = classification_metrics(
            pseudo_y.values,
            result["anomaly_score"].values,
            result["is_anomaly"].values,
        )

except Exception as e:
    st.error(f"분석 중 오류가 발생했습니다: {e}")
    st.stop()


# ------------------------------------------------------------
# 8. 상단 요약 카드
# ------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("데이터 수", f"{meta['n_rows']:,}")
m2.metric("변수 수", f"{meta['n_features']:,}")
m3.metric("탐지 이상 수", f"{int(result['is_anomaly'].sum()):,}")
m4.metric("이상 비율", f"{result['is_anomaly'].mean() * 100:.2f}%")
m5.metric("임계값", f"{float(result['threshold'].iloc[0]):.3f}")

st.caption(
    f"시간 컬럼: {meta['time_col']} | 빈도: {meta['inferred_freq']} | 기간: {meta['start']} ~ {meta['end']}"
)

st.divider()

# ------------------------------------------------------------
# TAB 기반 대시보드
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "데이터 기본 분석",
    "이상탐지 결과 분석",
    "평가지표 및 시각화"
])


# ============================================================
# TAB 1 : 데이터 기본 분석
# ============================================================
with tab1:

    st.subheader("데이터 기본 분석")

    c1, c2 = st.columns([1, 1])

    with c1:
        with st.container(border=True):

            st.markdown("#### 데이터 미리보기")

            st.dataframe(
                raw_df.head(10),
                use_container_width=True,
            )

    with c2:
        with st.container(border=True):

            st.markdown("#### 변수별 정상성 검정")

            st.dataframe(
                stat_df,
                use_container_width=True,
                hide_index=True,
            )
    
    c3, c4 = st.columns([1, 1])

    with c3:
        with st.container(border=True):

            st.markdown("#### 높은 상관관계 변수쌍")

            st.dataframe(
                high_corr_df,
                use_container_width=True,
                hide_index=True,
            )

    with c4:
        with st.container(border=True):

            st.markdown("#### 상관관계 Heatmap")
    
            corr_cols = ts_df.columns[: min(25, len(ts_df.columns))]
    
            st.plotly_chart(
                plot_correlation_heatmap(ts_df[corr_cols]),
                use_container_width=True,
            )

# ============================================================
# TAB 2 : 이상탐지 결과 분석
# ============================================================
with tab2:

    st.subheader("시계열 이상탐지 결과 분석 및 시각화")

    c1, c2 = st.columns([1.4, 1])

    with c1:
        with st.container(border=True):

            st.markdown("#### 원본 시계열과 탐지된 이상 시점")

            st.plotly_chart(
                plot_multivariate_series(
                    ts_df,
                    result,
                    selected_cols
                ),
                use_container_width=True,
            )

    with c2:
        with st.container(border=True):

            st.markdown("#### 이상 점수 추이")

            st.plotly_chart(
                plot_score(
                    result,
                    score_detail
                ),
                use_container_width=True,
            )

    c3, c4 = st.columns([1, 1])

    with c3:
        with st.container(border=True):

            st.markdown("#### 이상 점수 분포")

            st.plotly_chart(
                plot_score_distribution(result),
                use_container_width=True,
            )

    with c4:
        with st.container(border=True):

            st.markdown("#### 주요 기여 변수")

            st.plotly_chart(
                plot_contribution(
                    contrib_df,
                    top_n=int(top_n)
                ),
                use_container_width=True,
            )

    with st.container(border=True):

        st.markdown("#### 탐지된 이상 시점")

        out = ts_df.copy()

        out.insert(
            0,
            "is_anomaly",
            result["is_anomaly"].values
        )

        out.insert(
            0,
            "anomaly_score",
            result["anomaly_score"].values
        )

        anomalies_only = out[
            out["is_anomaly"]
        ].sort_values(
            "anomaly_score",
            ascending=False,
        )

        st.dataframe(
            anomalies_only,
            use_container_width=True,
            height=350,
        )


# ============================================================
# TAB 3 : 평가지표 및 시각화
# ============================================================
with tab3:

    st.subheader("평가지표 및 시각화")

    st.markdown(
        """
        Precision, Recall, F1 등은
        실제 정답 라벨이 없는 환경을 고려하여
        내부 기준 기반 pseudo-label로 계산된 참고 지표입니다.
        """
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Precision",
        f"{metrics['Precision']:.3f}"
    )

    m2.metric(
        "Recall",
        f"{metrics['Recall']:.3f}"
    )

    m3.metric(
        "F1",
        f"{metrics['F1']:.3f}"
    )

    m4.metric(
        "ROC-AUC",
        f"{metrics['ROC-AUC']:.3f}"
        if not np.isnan(metrics["ROC-AUC"])
        else "N/A"
    )

    m5.metric(
        "PR-AUC",
        f"{metrics['PR-AUC']:.3f}"
        if not np.isnan(metrics["PR-AUC"])
        else "N/A"
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        with st.container(border=True):

            st.markdown("#### 참고 혼동행렬")

            st.plotly_chart(
                plot_confusion(metrics),
                use_container_width=True,
            )

    with c2:
        with st.container(border=True):

            st.markdown("#### 이상 점수 자기상관 진단")

            st.metric(
                f"Ljung-Box p-value (lag={lb['lag']})",
                f"{lb['p_value']:.4f}"
                if not np.isnan(lb["p_value"])
                else "N/A",
            )

            st.write(
                """
                이상 점수가 특정 구간에서
                연속적으로 높게 나타나면
                구조 변화나 계절성 가능성이 있습니다.
                """
            )
