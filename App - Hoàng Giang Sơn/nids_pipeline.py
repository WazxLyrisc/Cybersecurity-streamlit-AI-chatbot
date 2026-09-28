"""
nids_pipeline.py - Huấn luyện & suy luận cho NIDS (UNSW-NB15)

Ý tưởng chính
- Người dùng chỉ nhập 8 thông số "thô": proto, service, state, dur, spkts, dpkts, sbytes, dbytes.
- Các đặc trưng còn lại (rate, smean, dmean, sload, dload) được TÍNH từ 8 thông số đó bằng
  build_features(). Hàm này dùng chung cho cả lúc train lẫn lúc chạy app => không bị lệch.
- Một mô hình đa lớp (Normal + 9 loại tấn công). "Mức độ rủi ro" = 1 - P(Normal).
- Toàn bộ (pipeline + LabelEncoder + danh sách lựa chọn cho form) lưu trong MỘT file joblib.

Chạy huấn luyện (trên Kaggle):
    python nids_pipeline.py \
        --train /kaggle/input/datasets/dhoogla/unswnb15/UNSW_NB15_training-set.parquet \
        --test  /kaggle/input/datasets/dhoogla/unswnb15/UNSW_NB15_testing-set.parquet

Dùng trong Streamlit:
    from nids_pipeline import load_bundle, predict_flow, risk_badge
    bundle = load_bundle("nids_bundle.joblib")
    result = predict_flow(bundle, {"proto": "tcp", "service": "-", "state": "FIN",
                                   "dur": 1.2, "spkts": 14, "dpkts": 8,
                                   "sbytes": 8834, "dbytes": 354})[0]
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------------------
CAT_COLS = ["proto", "service", "state"]
RAW_NUM_COLS = ["dur", "spkts", "dpkts", "sbytes", "dbytes"]
DERIVED_COLS = ["rate", "smean", "dmean", "sload", "dload"]
NUM_COLS = RAW_NUM_COLS + DERIVED_COLS
FEATURES = CAT_COLS + NUM_COLS          # thứ tự cột cố định cho model
USER_INPUT_COLS = CAT_COLS + RAW_NUM_COLS  # những gì người dùng phải nhập

TOP_N_PROTO = 6          # giữ N giao thức phổ biến nhất, phần còn lại gộp thành "other"
NORMAL = "Normal"
RISK_THRESHOLD = 0.5
RANDOM_STATE = 42


# ----------------------------------------------------------------------------------------
# Tạo đặc trưng (dùng chung cho train và app)
# ----------------------------------------------------------------------------------------
def build_features(df: pd.DataFrame, top_protos: list[str]) -> pd.DataFrame:
    """Từ 8 cột thô -> 13 đặc trưng. Không dùng bất kỳ cột nhãn nào."""
    out = pd.DataFrame(index=df.index)
    out["proto"] = df["proto"].astype(str).str.strip().str.lower()
    out["proto"] = out["proto"].where(out["proto"].isin(top_protos), "other")
    out["service"] = df["service"].astype(str).str.strip().str.lower()
    out["state"] = df["state"].astype(str).str.strip().str.upper()
    for c in RAW_NUM_COLS:
        out[c] = pd.to_numeric(df[c]).astype("float64")

    dur, spk, dpk = out["dur"], out["spkts"], out["dpkts"]
    sby, dby = out["sbytes"], out["dbytes"]
    has_dur = dur > 0
    safe_dur = dur.where(has_dur, 1.0)

    # rate khớp đúng công thức của dataset: (spkts + dpkts - 1) / dur
    out["rate"] = np.where(has_dur, (spk + dpk - 1).clip(lower=0) / safe_dur, 0.0)
    out["smean"] = np.where(spk > 0, sby / spk.where(spk > 0, 1.0), 0.0)
    out["dmean"] = np.where(dpk > 0, dby / dpk.where(dpk > 0, 1.0), 0.0)
    # sload/dload chỉ là XẤP XỈ của cột gốc (bits/giây); vì train cũng dùng đúng công thức
    # này nên model và app vẫn nhất quán.
    out["sload"] = np.where(has_dur, sby * 8 / safe_dur, 0.0)
    out["dload"] = np.where(has_dur, dby * 8 / safe_dur, 0.0)
    return out[FEATURES]


# ----------------------------------------------------------------------------------------
# Mô hình
# ----------------------------------------------------------------------------------------
def make_pipeline(model) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    # Không cần StandardScaler: RF/XGBoost dựa trên ngưỡng cắt nên không nhạy với thang đo.
    # (Nếu giảng viên yêu cầu, thêm vào ("num", StandardScaler(), NUM_COLS) cũng không hại.)
    return Pipeline([("prep", prep), ("clf", model)])


def candidate_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
        ),
    }


def _weights(y) -> np.ndarray:
    # Trọng số cân bằng lớp, làm mềm bằng căn bậc hai để lớp hiếm (Worms...) không bị nhấn quá mạnh.
    return compute_sample_weight("balanced", y) ** 0.5


def _fit(pipe: Pipeline, X, y) -> Pipeline:
    pipe.fit(X, y, clf__sample_weight=_weights(y))
    return pipe


def _decide(proba: np.ndarray, normal_idx: int):
    """Quy tắc quyết định chung: risk = 1 - P(Normal); nếu risk >= ngưỡng thì chọn loại
    tấn công có xác suất cao nhất, ngược lại là Normal."""
    risk = 1.0 - proba[:, normal_idx]
    masked = proba.copy()
    masked[:, normal_idx] = -1.0
    pred = np.where(risk >= RISK_THRESHOLD, masked.argmax(axis=1), normal_idx)
    return pred, risk


# ----------------------------------------------------------------------------------------
# Huấn luyện
# ----------------------------------------------------------------------------------------
def train(tr: pd.DataFrame, te: pd.DataFrame, out_path: str = "nids_bundle.joblib",
          dedup: bool = True) -> dict:
    top_protos = (
        tr["proto"].astype(str).str.strip().str.lower().value_counts().head(TOP_N_PROTO).index.tolist()
    )
    le = LabelEncoder().fit(tr["attack_cat"].astype(str))
    normal_idx = list(le.classes_).index(NORMAL)

    Xtr = build_features(tr, top_protos)
    ytr = le.transform(tr["attack_cat"].astype(str))
    Xte = build_features(te, top_protos)
    yte = le.transform(te["attack_cat"].astype(str))

    # Kiểm tra chồng lấn: bao nhiêu dòng test giống hệt một dòng train (trên các đặc trưng này)?
    h = lambda X: pd.util.hash_pandas_object(X, index=False)
    overlap = h(Xte).isin(set(h(Xtr))).mean()
    print(f"[i] {overlap:.1%} dòng test trùng hệt một dòng train (trên {len(FEATURES)} đặc trưng)")

    if dedup:
        keep = ~Xtr.assign(_y=ytr).duplicated().to_numpy()
        print(f"[i] Loại trùng lặp trong train: {len(Xtr)} -> {int(keep.sum())} dòng")
        Xtr, ytr = Xtr[keep].reset_index(drop=True), ytr[keep]

    # Validation tách từ train (KHÔNG đụng vào test khi chọn model)
    Xa, Xv, ya, yv = train_test_split(
        Xtr, ytr, test_size=0.2, stratify=ytr, random_state=RANDOM_STATE
    )
    scores = {}
    for name, model in candidate_models().items():
        pipe = _fit(make_pipeline(model), Xa, ya)
        scores[name] = f1_score(yv, pipe.predict(Xv), average="macro")
        print(f"[val] {name}: macro-F1 = {scores[name]:.4f}")
    best = max(scores, key=scores.get)
    print(f"[i] Chọn {best}")

    # Huấn luyện lại trên toàn bộ train rồi đánh giá test đúng MỘT lần
    final = _fit(make_pipeline(candidate_models()[best]), Xtr, ytr)
    proba = final.predict_proba(Xte)
    pred, risk = _decide(proba, normal_idx)

    print("\n=== TEST: đa lớp ===")
    print(classification_report(yte, pred, target_names=le.classes_, digits=3, zero_division=0))
    print("Confusion matrix (hàng = thực tế, cột = dự đoán):")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(pd.DataFrame(confusion_matrix(yte, pred), index=le.classes_, columns=le.classes_))

    true_att, pred_att = yte != normal_idx, risk >= RISK_THRESHOLD
    print("\n=== TEST: nhị phân (Normal vs Attack) ===")
    print(f"Recall Normal : {(~pred_att[~true_att]).mean():.3f}")
    print(f"Recall Attack : {pred_att[true_att].mean():.3f}")
    print(f"Precision Attack: {true_att[pred_att].mean():.3f}")
    print(f"Số Normal bị báo nhầm (FP): {int(pred_att[~true_att].sum())} / {int((~true_att).sum())}")

    bundle = {
        "pipeline": final,
        "label_encoder": le,
        "top_protos": top_protos,
        "features": FEATURES,
        "model_name": best,
        "choices": {  # dùng cho selectbox trong Streamlit
            "proto": top_protos + ["other"],
            "service": sorted(tr["service"].astype(str).str.lower().unique().tolist()),
            "state": sorted(tr["state"].astype(str).str.upper().unique().tolist()),
        },
    }
    joblib.dump(bundle, out_path, compress=3)
    print(f"\n[i] Đã lưu: {out_path}")
    return bundle


# ----------------------------------------------------------------------------------------
# Suy luận (dùng trong Streamlit)
# ----------------------------------------------------------------------------------------
def load_bundle(path: str = "nids_bundle.joblib") -> dict:
    return joblib.load(path)


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in USER_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    nums = df[RAW_NUM_COLS].apply(pd.to_numeric, errors="coerce")
    if nums.isna().any().any():
        raise ValueError("Có giá trị số bị trống hoặc không hợp lệ.")
    if (nums < 0).any().any():
        raise ValueError("Các chỉ số không được âm.")
    if (nums["spkts"] < 1).any():
        raise ValueError("spkts phải >= 1.")


def risk_badge(risk: float) -> tuple[str, str]:
    """(nhãn, màu) cho badge cảnh báo."""
    if risk < 0.3:
        return "THẤP", "green"
    if risk < 0.7:
        return "TRUNG BÌNH", "orange"
    return "CAO", "red"


def predict_flow(bundle: dict, flows) -> list[dict]:
    """flows: dict (1 luồng) hoặc DataFrame (nhiều luồng, ví dụ từ file CSV tải lên)."""
    df = pd.DataFrame([flows]) if isinstance(flows, dict) else pd.DataFrame(flows).reset_index(drop=True)
    _validate(df)
    X = build_features(df, bundle["top_protos"])
    proba = bundle["pipeline"].predict_proba(X)
    classes = list(bundle["label_encoder"].classes_)
    pred, risk = _decide(proba, classes.index(NORMAL))

    results = []
    for i in range(len(X)):
        badge, color = risk_badge(float(risk[i]))
        results.append(
            {
                "label": classes[int(pred[i])],
                "risk": float(risk[i]),       # 1 - P(Normal), thang 0..1
                "badge": badge,
                "color": color,
                "probabilities": {c: float(p) for c, p in zip(classes, proba[i])},
                "features": X.iloc[i].to_dict(),  # đưa vào context của chatbot
            }
        )
    return results


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="nids_bundle.joblib")
    ap.add_argument("--no-dedup", action="store_true", help="không loại dòng trùng trong train")
    args = ap.parse_args()
    b = train(pd.read_parquet(args.train), pd.read_parquet(args.test), args.out, not args.no_dedup)
    demo = pd.read_parquet(args.test).sample(3, random_state=1)
    for (_, row), res in zip(demo.iterrows(), predict_flow(b, demo)):
        print(f"thực tế={row['attack_cat']:<15} dự đoán={res['label']:<15} rủi ro={res['risk']:.0%} ({res['badge']})")
