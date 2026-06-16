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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

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


def rank_normalize(s: np.ndarray) -> np.ndarray:
    """순위 기반 [0,1] 정규화.

    이상 점수는 heavy-tail이라 minmax로 정규화하면 극단값 하나가 나머지를
    0 근처로 압축시킨다. 순위로 변환하면 각 모델 점수가 균등 분포가 되어,
    앙상블 평균이 특정 모델의 스케일이나 극단값에 휘둘리지 않는다.
    """
    s = np.asarray(s, dtype=float)
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return (ranks - 1.0) / (len(ranks) - 1.0 + 1e-12)


def score_isolation_forest(features: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    x = scaler.fit_transform(features.values)

    # contamination은 decision_function의 상수 offset만 바꾸며, 이후 순위
    # 정규화에서 상쇄되어 점수에 영향이 없으므로 "auto"로 둔다.
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
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
    target_ratio: float,
    rolling_window: int,
    include_rolling: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    features = make_feature_matrix(df, rolling_window, include_rolling)

    scores = {}

    if method in ["Isolation Forest", "Ensemble"]:
        scores["Isolation Forest"] = rank_normalize(score_isolation_forest(features))

    if method in ["Robust Z-Score", "Ensemble"]:
        scores["Robust Z-Score"] = rank_normalize(score_robust_z(df))

    if method in ["PCA Reconstruction", "Ensemble"]:
        scores["PCA Reconstruction"] = rank_normalize(score_pca_reconstruction(features))

    if method == "Ensemble":
        final_score = np.mean(np.vstack(list(scores.values())), axis=0)
    else:
        final_score = list(scores.values())[0]

    # 임계값은 "상위 target_ratio 비율"을 이상으로 보는 단일 분위수 컷.
    # 기존의 contamination(자동) / manual_quantile(수동)은 동일한 분위수 컷을
    # 두 이름으로 중복 노출한 것이라 하나로 통합했다.
    threshold = np.quantile(final_score, 1 - target_ratio)

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
def calc_missing_summary(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    total = len(df)

    for col in df.columns:

        missing_count = df[col].isna().sum()
        missing_ratio = (missing_count / total) * 100

        rows.append(
            {
                "변수": col,
                "결측 개수": int(missing_count),
                "결측 비율(%)": round(missing_ratio, 2),
                "상태": (
                    "정상"
                    if missing_ratio == 0
                    else "결측 존재"
                ),
            }
        )

    out = pd.DataFrame(rows)

    out = out.sort_values(
        "결측 비율(%)",
        ascending=False,
    )

    return out

def calc_acf_summary(
    df: pd.DataFrame,
    max_lag: int = 48,
    top_k: int = 10,
) -> pd.DataFrame:

    rows = []

    for col in df.columns[:top_k]:

        s = df[col].dropna()

        if len(s) < max_lag + 5:
            continue

        try:
            acf_values = acf(
                s,
                nlags=max_lag,
                fft=True,
            )

            # lag 0 제외
            lag_values = acf_values[1:]

            best_lag = int(np.argmax(np.abs(lag_values)) + 1)
            best_corr = float(lag_values[best_lag - 1])

            if abs(best_corr) >= 0.7:
                interpretation = "강한 주기/반복 패턴"
            elif abs(best_corr) >= 0.4:
                interpretation = "중간 수준 자기상관"
            else:
                interpretation = "약한 자기상관"

            rows.append(
                {
                    "변수": col,
                    "주요 Lag": best_lag,
                    "자기상관": round(best_corr, 3),
                    "추천 Rolling Window": best_lag,
                    "해석": interpretation,
                }
            )

        except Exception:
            continue

    if len(rows) == 0:
        return pd.DataFrame(
            {
                "변수": ["분석 실패"],
                "주요 Lag": ["-"],
                "자기상관": ["-"],
                "추천 Rolling Window": ["-"],
                "해석": ["데이터 부족"],
            }
        )

    out = pd.DataFrame(rows)

    out["정렬용"] = out["자기상관"].abs()

    out = out.sort_values(
        "정렬용",
        ascending=False,
    ).drop(columns=["정렬용"])

    return out

def calc_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    pairs = []

    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]

            if pd.notna(value) and np.isfinite(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "변수 1": str(cols[i]),
                        "변수 2": str(cols[j]),
                        "상관계수": round(float(value), 4),
                    }
                )

    if not pairs:
        return pd.DataFrame(
            {
                "변수 1": ["해당 없음"],
                "변수 2": ["해당 없음"],
                "상관계수": ["기준 이상 상관관계 없음"],
            }
        )

    out = pd.DataFrame(pairs)
    out["정렬용"] = out["상관계수"].abs()
    out = out.sort_values("정렬용", ascending=False).drop(columns=["정렬용"])
    out = out.head(50)

    return out

def suggest_redundant_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """상관이 높은 변수쌍에서 중복으로 볼 수 있는 컬럼을 골라낸다.

    그리디 휴리스틱(R caret::findCorrelation과 동일):
    임계값을 넘는 쌍 중 상관이 가장 높은 쌍을 찾고, 둘 중 '나머지 변수들과
    평균 상관이 더 높은'(즉 더 중복적인) 쪽을 제외 후보로 지정한다.
    남은 변수들 사이에 임계값 초과 쌍이 없어질 때까지 반복한다.
    """
    corr = df.corr(numeric_only=True).abs()
    cols = list(corr.columns)

    empty = pd.DataFrame(
        {
            "제외 후보": ["해당 없음"],
            "대표 변수(유지)": ["-"],
            "상관계수": ["기준 이상 중복 없음"],
        }
    )

    if len(cols) < 2:
        return empty

    corr_arr = corr.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_arr, 0.0)
    corr = pd.DataFrame(corr_arr, index=cols, columns=cols)

    remaining = list(cols)
    dropped = []

    while len(remaining) >= 2:
        sub = corr.loc[remaining, remaining]
        max_val = float(sub.values.max())

        if max_val < threshold:
            break

        i, j = np.unravel_index(np.argmax(sub.values), sub.values.shape)
        a, b = sub.index[i], sub.columns[j]

        others = [c for c in remaining if c not in (a, b)]
        mean_a = float(corr.loc[a, others].mean()) if others else 0.0
        mean_b = float(corr.loc[b, others].mean()) if others else 0.0

        drop, keep = (a, b) if mean_a >= mean_b else (b, a)

        dropped.append(
            {
                "제외 후보": str(drop),
                "대표 변수(유지)": str(keep),
                "상관계수": round(float(corr.loc[drop, keep]), 4),
                "타 변수 평균상관": round(max(mean_a, mean_b), 4),
            }
        )
        remaining.remove(drop)

    if not dropped:
        return empty

    return pd.DataFrame(dropped)

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

def anomaly_quality_summary(
    result: pd.DataFrame,
    score_detail: pd.DataFrame,
) -> Dict:

    anomaly_mask = result["is_anomaly"]

    anomaly_ratio = anomaly_mask.mean() * 100
    mean_score = result["anomaly_score"].mean()
    max_score = result["anomaly_score"].max()

    base_models = [
        col for col in score_detail.columns
        if col != "Final Score"
    ]

    if len(base_models) > 0:
        model_flags = pd.DataFrame(index=score_detail.index)

        for col in base_models:
            threshold = np.quantile(
                score_detail[col],
                1 - anomaly_mask.mean()
            )
            model_flags[col] = score_detail[col] >= threshold

        model_agreement = model_flags.mean(axis=1)

        avg_agreement_anomaly = (
            model_agreement[anomaly_mask].mean()
            if anomaly_mask.sum() > 0
            else np.nan
        )

        high_confidence_ratio = (
            (model_agreement[anomaly_mask] >= 0.67).mean() * 100
            if anomaly_mask.sum() > 0
            else np.nan
        )
    else:
        avg_agreement_anomaly = np.nan
        high_confidence_ratio = np.nan

    return {
        "anomaly_ratio": anomaly_ratio,
        "mean_score": mean_score,
        "max_score": max_score,
        "avg_agreement_anomaly": avg_agreement_anomaly,
        "high_confidence_ratio": high_confidence_ratio,
    }

def make_model_agreement_table(
    result: pd.DataFrame,
    score_detail: pd.DataFrame,
) -> pd.DataFrame:

    anomaly_mask = result["is_anomaly"]

    base_models = [
        col for col in score_detail.columns
        if col != "Final Score"
    ]

    rows = []

    for col in base_models:
        threshold = np.quantile(
            score_detail[col],
            1 - anomaly_mask.mean()
        )

        model_detected = score_detail[col] >= threshold

        rows.append(
            {
                "모델": col,
                "모델 단독 탐지 수": int(model_detected.sum()),
                "최종 이상과 일치 수": int((model_detected & anomaly_mask).sum()),
                "최종 이상 기준 일치율(%)": round(
                    ((model_detected & anomaly_mask).sum()
                     / max(1, anomaly_mask.sum())) * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4-b. 합성 주입 기반 정량 평가 (라벨이 없는 환경의 통제된 점검)
# ------------------------------------------------------------
def inject_synthetic_anomalies(
    df: pd.DataFrame,
    point_ratio: float = 0.01,
    n_segments: int = 3,
    segment_len: int = 8,
    magnitude: float = 4.0,
    seed: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """데이터 복사본에 알려진 이상을 주입해 의사-정답 라벨을 만든다.

    - point spike : 무작위 시점·무작위 변수 일부에 magnitude*std 크기의 충격
    - collective  : 연속 구간 전체를 magnitude*std 만큼 이동(레짐 변화 모사)

    주의: 원본에 이미 실제 이상이 섞여 있을 수 있어, 주입되지 않은 시점이
    모두 '정상'이라는 보장은 없다. 따라서 절대 성능이 아니라 동일 파이프라인의
    탐지 민감도를 보는 통제된 점검으로 해석해야 한다.
    """
    rng = np.random.RandomState(seed)
    values = df.to_numpy(dtype=float).copy()
    n, p = values.shape
    labels = np.zeros(n, dtype=bool)

    col_std = df.std(axis=0).replace(0, 1.0).to_numpy()

    # 1) point spikes
    n_points = max(1, int(n * point_ratio))
    point_idx = rng.choice(n, size=min(n_points, n), replace=False)
    for t in point_idx:
        k = rng.randint(1, max(2, p // 2) + 1)
        cols = rng.choice(p, size=k, replace=False)
        sign = rng.choice([-1.0, 1.0], size=k)
        values[t, cols] += sign * magnitude * col_std[cols]
        labels[t] = True

    # 2) collective segments
    if n > segment_len + 2:
        for _ in range(n_segments):
            start = rng.randint(0, n - segment_len)
            end = start + segment_len
            k = rng.randint(1, max(2, p // 2) + 1)
            cols = rng.choice(p, size=k, replace=False)
            shift = rng.choice([-1.0, 1.0]) * magnitude * col_std[cols]
            values[start:end, cols] += shift
            labels[start:end] = True

    contaminated = pd.DataFrame(values, index=df.index, columns=df.columns)
    return contaminated, labels


def point_adjust_predictions(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Xu et al.(2018) point-adjust 규칙.

    정답 이상 '구간' 안에서 한 시점이라도 탐지되면, 그 구간 전체를 탐지한
    것으로 간주한다. 시계열 이상탐지 평가의 표준 관행.
    """
    pred = np.asarray(pred, dtype=bool).copy()
    label = np.asarray(label, dtype=bool)
    n = len(label)
    i = 0
    while i < n:
        if label[i]:
            j = i
            while j < n and label[j]:
                j += 1
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1
    return pred


def evaluate_with_synthetic_injection(
    df: pd.DataFrame,
    method: str,
    target_ratio: float,
    rolling_window: int,
    include_rolling: bool,
) -> Optional[Dict]:
    """주입한 이상을 정답으로 두고 동일 파이프라인을 재실행해 정량 평가."""
    try:
        contaminated, labels = inject_synthetic_anomalies(df)

        if labels.sum() == 0 or labels.all():
            return None

        res, _ = detect_anomalies(
            contaminated,
            method=method,
            target_ratio=target_ratio,
            rolling_window=rolling_window,
            include_rolling=include_rolling,
        )

        score = res["anomaly_score"].to_numpy()
        pred = res["is_anomaly"].to_numpy()

        pr_auc = float(average_precision_score(labels, score))
        roc_auc = float(roc_auc_score(labels, score))

        precision_raw, recall_raw, f1_raw, _ = precision_recall_fscore_support(
            labels, pred, average="binary", zero_division=0
        )

        pa_pred = point_adjust_predictions(pred, labels)
        precision_pa, recall_pa, f1_pa, _ = precision_recall_fscore_support(
            labels, pa_pred, average="binary", zero_division=0
        )

        prec_curve, rec_curve, _ = precision_recall_curve(labels, score)

        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1_raw": float(f1_raw),
            "precision_raw": float(precision_raw),
            "recall_raw": float(recall_raw),
            "f1_pa": float(f1_pa),
            "precision_pa": float(precision_pa),
            "recall_pa": float(recall_pa),
            "n_injected": int(labels.sum()),
            "baseline": float(labels.mean()),
            "prec_curve": prec_curve,
            "rec_curve": rec_curve,
        }
    except Exception:
        return None

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
                line=dict(
                width=3 if col == "Final Score" else 1.5,
                color="navy" if col == "Final Score" else None),
            )
        )

    fig.add_trace( 
        go.Scatter( 
            x=result.index, y=result["threshold"], 
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

def plot_model_agreement(
    result: pd.DataFrame,
    score_detail: pd.DataFrame,
) -> go.Figure:

    anomaly_mask = result["is_anomaly"]

    base_models = [
        col for col in score_detail.columns
        if col != "Final Score"
    ]

    agreement_counts = []

    for idx in score_detail.index:
        count = 0

        for col in base_models:
            threshold = np.quantile(
                score_detail[col],
                1 - anomaly_mask.mean()
            )

            if score_detail.loc[idx, col] >= threshold:
                count += 1

        agreement_counts.append(count)

    agreement_df = pd.DataFrame(
        {
            "동의 모델 수": agreement_counts,
            "is_anomaly": anomaly_mask.values,
        },
        index=score_detail.index,
    )

    anomaly_agree = agreement_df[agreement_df["is_anomaly"]]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=anomaly_agree["동의 모델 수"],
            nbinsx=len(base_models),
            name="Detected Anomaly",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="동의한 개별 모델 수",
        yaxis_title="이상 시점 수",
        showlegend=False,
    )

    return fig

def plot_score_gap(result: pd.DataFrame) -> go.Figure:

    plot_df = pd.DataFrame(
        {
            "구분": np.where(
                result["is_anomaly"],
                "Anomaly",
                "Normal",
            ),
            "Anomaly Score": result["anomaly_score"],
        }
    )

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=plot_df["구분"],
            y=plot_df["Anomaly Score"],
            boxmean=True,
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="구분",
        yaxis_title="Anomaly Score",
    )

    return fig

def plot_pr_curve(eval_result: Dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=eval_result["rec_curve"],
            y=eval_result["prec_curve"],
            mode="lines",
            name=f"PR (AP={eval_result['pr_auc']:.3f})",
            line=dict(color="navy", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,0,128,0.08)",
        )
    )

    fig.add_hline(
        y=eval_result["baseline"],
        line_dash="dash",
        line_color="#999",
        annotation_text=f"무작위 기준선 ({eval_result['baseline']:.3f})",
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1.0]),
        yaxis=dict(range=[0, 1.02]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

# ------------------------------------------------------------
# 7. 메인
# ------------------------------------------------------------

if "file_hash" not in st.session_state:
    st.session_state["file_hash"] = None


# 제목
st.title("📈시계열분석 Project2 이상탐지")
st.markdown("#### C321032 박하율")
st.divider()

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

            method = "Ensemble"

            target_ratio = st.slider(
                "목표 이상 비율 (상위 분위수 컷)",
                min_value=0.005,
                max_value=0.300,
                value=0.050,
                step=0.005,
                format="%.3f",
                help="최종 이상 점수의 상위 이 비율을 이상으로 판정합니다. "
                     "예: 0.05 → 상위 5%(=95분위) 이상.",
            )

            st.caption(
                f"최종 이상 점수의 상위 **{target_ratio * 100:.1f}%** 시점을 "
                "이상으로 판정합니다. 기존 contamination(자동) / 수동 분위수 "
                "컨트롤은 동일한 분위수 컷을 중복 노출한 것이라 하나로 통합했습니다."
            )

    # ========================================================
    # RIGHT : 3. 세부 설정
    # ========================================================
    with right_panel:

        with st.container(border=True, height=455):
            st.markdown("#### 세부 설정")

            st.info(
                    "Isolation Forest, Robust Z-Score, PCA Reconstruction을 결합한 Ensemble 모델로 이상 탐지"
                )

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
            target_ratio=target_ratio,
            rolling_window=int(rolling_window),
            include_rolling=include_rolling,
        )

        eval_result = evaluate_with_synthetic_injection(
            ts_df,
            method=method,
            target_ratio=target_ratio,
            rolling_window=int(rolling_window),
            include_rolling=include_rolling,
        )

        missing_df = calc_missing_summary(raw_df)
        high_corr_df = calc_high_corr_pairs(ts_df, threshold=corr_threshold)
        redundant_df = suggest_redundant_columns(ts_df, threshold=corr_threshold)
        acf_df = calc_acf_summary(ts_df)
        contrib_df = feature_contribution(ts_df, result["is_anomaly"])
        lb = calc_ljungbox_summary(result["anomaly_score"])

        quality = anomaly_quality_summary(
            result,
            score_detail,
        )
        
        agreement_df = make_model_agreement_table(
            result,
            score_detail,
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
    "🧩 데이터 진단 및 EDA",
    "🚨 이상탐지 결과",
    "✅ 탐지 품질 진단"
])


# ============================================================
# TAB 1 : 데이터 기본 분석
# ============================================================
with tab1:

    st.subheader("데이터 기본 분석")

    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        st.markdown("#### 데이터 미리보기")
        st.dataframe(
            raw_df.head(10),
            use_container_width=True,
            height=400,
        )

    with c2:
        st.markdown("#### 변수별 결측 분석")

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    st.divider()

    c3, c4 = st.columns([1, 1], gap="medium")

    with c3:
        st.markdown("#### 높은 상관관계 변수쌍")
    
        high_corr_display = high_corr_df.copy()
    
        st.dataframe(
            high_corr_display,
            use_container_width=True,
            hide_index=True,
            height=250,
        )

        st.markdown("#### 제외 검토 대상(중복) 변수")

        st.dataframe(
            redundant_df,
            use_container_width=True,
            hide_index=True,
            height=200,
        )

        st.caption(
            "각 중복 그룹에서 대표 1개만 남기고 나머지를 제외 후보로 제시합니다 "
            f"(현재 기준 |상관| ≥ {corr_threshold:.2f}). 실제 제거 목적이라면 "
            "0.9 이상을 쓰는 것이 일반적입니다. 다만 이 목록은 **정보 제공용**으로, "
            "PCA Reconstruction은 변수 간 상관 구조의 붕괴 자체를 이상 신호로 쓰므로 "
            "상관 높은 변수를 무작정 제거하면 탐지 신호가 약해질 수 있습니다."
        )
        
    with c4:

        st.markdown("#### ACF 기반 Lag 분석")
    
        st.dataframe(
            acf_df,
            use_container_width=True,
            hide_index=True,
            #height=300,
        )
    
        st.caption(
            """
            자기상관이 강한 lag는
            반복 패턴이나 주기성을 의미.
            Rolling window 설정 시 참고 가능.
            """
        )

# ============================================================
# TAB 2 : 이상탐지 결과 분석
# ============================================================
with tab2:

    st.subheader("시계열 이상탐지 결과 분석 및 시각화")

    c1, c2 = st.columns([1.4, 1])

    with c1:
        with st.container(border=True):

            st.markdown("#### 탐지된 이상 시점")

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
    
            st.markdown("#### 모델별 이상 점수 및 최종 점수")
    
            st.plotly_chart(
                plot_score(
                    result,
                    score_detail
                ),
                use_container_width=True,
            )
    
            st.caption(
                """
                각 모델의 이상 점수를 순위 기반으로 [0,1] 정규화한 뒤,
                평균값을 Final Score로 사용합니다.
                Threshold 이상인 시점을 최종 이상으로 판단합니다.
                """
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

    st.subheader("탐지 품질 진단")

    st.markdown(
        """
        실제 정답 라벨이 없는 비지도 환경이므로 평가를 두 갈래로 나눕니다.
        **(1) 합성 이상 주입 기반 정량 평가** — 알려진 이상을 주입해 PR-AUC와
        point-adjusted F1로 탐지기의 민감도를 측정합니다.
        **(2) 구조적 진단** — 점수 분포, 모델 간 일치도 등 내부 일관성을 검토합니다.
        """
    )

    # ----------------------------------------------------
    # (1) 합성 이상 주입 기반 정량 평가
    # ----------------------------------------------------
    with st.container(border=True):

        st.markdown("#### (1) 합성 이상 주입 기반 정량 평가")

        if eval_result is None:
            st.info("데이터가 짧거나 주입에 실패하여 정량 평가를 건너뛰었습니다.")
        else:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("PR-AUC (AP)", f"{eval_result['pr_auc']:.3f}")
            e2.metric("ROC-AUC", f"{eval_result['roc_auc']:.3f}")
            e3.metric("Point-adjusted F1", f"{eval_result['f1_pa']:.3f}")
            e4.metric("Raw F1", f"{eval_result['f1_raw']:.3f}")

            cc1, cc2 = st.columns([1.3, 1])

            with cc1:
                st.plotly_chart(
                    plot_pr_curve(eval_result),
                    use_container_width=True,
                )

            with cc2:
                st.markdown(
                    f"""
                    - 주입한 이상 시점: **{eval_result['n_injected']}개**
                    - Point-adjusted P / R:
                      **{eval_result['precision_pa']:.3f} / {eval_result['recall_pa']:.3f}**
                    - Raw P / R:
                      **{eval_result['precision_raw']:.3f} / {eval_result['recall_raw']:.3f}**
                    """
                )

        st.caption(
            "원본에 이미 실제 이상이 섞여 있을 수 있어, 주입되지 않은 시점이 "
            "모두 정상이라는 보장은 없습니다. 따라서 이 수치는 절대 성능이 "
            "아니라 동일 파이프라인의 탐지 민감도를 보는 통제된 점검으로 "
            "해석합니다. 시계열 관행에 따라 point-adjusted F1을 함께 제시합니다."
        )

    # ----------------------------------------------------
    # (2) 구조적 진단
    # ----------------------------------------------------
    st.markdown("#### (2) 구조적 진단")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "탐지 이상 비율",
        f"{quality['anomaly_ratio']:.2f}%"
    )

    m2.metric(
        "평균 이상 점수",
        f"{quality['mean_score']:.3f}"
    )

    m3.metric(
        "최대 이상 점수",
        f"{quality['max_score']:.3f}"
    )

    m4.metric(
        "고신뢰 이상 비율",
        f"{quality['high_confidence_ratio']:.1f}%"
        if not np.isnan(quality["high_confidence_ratio"])
        else "N/A"
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        with st.container(border=True):

            st.markdown("#### 이상 점수 분리도")

            st.plotly_chart(
                plot_score_gap(result),
                use_container_width=True,
            )

            st.caption(
                """
                정상 시점과 이상 시점의 anomaly score 분포가 잘 분리될수록
                임계값 기준이 비교적 명확하다고 볼 수 있음
                """
            )

    with c2:
        with st.container(border=True):

            st.markdown("#### 모델 간 이상 판단 일치도")

            st.plotly_chart(
                plot_model_agreement(
                    result,
                    score_detail,
                ),
                use_container_width=True,
            )

            st.caption(
                """
                여러 개별 모델이 동시에 이상으로 판단한 시점일수록
                상대적으로 신뢰도가 높은 이상 후보
                """
            )

    c3, c4 = st.columns([1, 1])

    with c3:
        with st.container(border=True):

            st.markdown("#### 개별 모델별 최종 이상 일치율")

            st.dataframe(
                agreement_df,
                use_container_width=True,
                hide_index=True,
                height=170,
            )

    with c4:
        with st.container(border=True):

            st.markdown("#### 이상 점수 연속성 진단")

            st.metric(
                f"Ljung-Box p-value (lag={lb['lag']})",
                f"{lb['p_value']:.4f}"
                if not np.isnan(lb["p_value"])
                else "N/A",
            )

            if lb["has_autocorr"]:
                st.warning(
                    """
                    이상 점수에 자기상관이 존재 →
                    이상이 특정 구간에 연속적으로 발생했을 가능성을 의미
                    """
                )
            else:
                st.info(
                    """
                    이상 점수의 뚜렷한 자기상관은 확인되지 않음 →
                    이상 후보가 비교적 산발적으로 분포했을 가능성
                    """
                )

    st.divider()

    with st.container(border=True):

        st.markdown("#### 📑품질 진단 해석 가이드")

        st.markdown(
            """
            - **PR-AUC / point-adjusted F1**(합성 주입)은 탐지기의 민감도를 보는
              유일한 정량 지표이므로 가장 우선해서 참고합니다.
            - **탐지 이상 비율**이 너무 높으면 과탐지 가능성이 있습니다.
            - **고신뢰 이상 비율**은 여러 개별 모델이 동시에 이상으로 판단한
              비율로, 값이 높을수록 앙상블 내부 일관성이 큽니다.
            - **Ljung-Box p-value**는 보조 지표입니다. rolling feature 자체가
              자기상관을 주입하므로, 낮은 p-value를 곧바로 이상의 시간적 군집으로
              해석하지 않도록 주의합니다.
            - 정답 라벨이 없는 환경에서는 이 지표들을 종합해 탐지 결과의 타당성을 판단합니다.
            """
        )
