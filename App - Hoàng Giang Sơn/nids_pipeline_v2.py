"""
nids_pipeline_v2.py - NIDS 2 tang (Binary Gatekeeper -> Multiclass Attack-type), UNSW-NB15

Kien truc (theo lua chon cua ban):
  Stage 1 (binary)     : Normal vs Attack, train tren TOAN BO train set.
  Stage 2 (multiclass) : loai tan cong cu the, train CHI tren cac dong Attack.
  Luc suy luan: chay Stage 1 truoc. Neu risk (=P(Attack)) < nguong -> "Normal".
  Neu >= nguong -> chay Stage 2 de ra loai tan cong.

  Luu y quan trong ve loi cong don: neu Stage 1 bao nham 1 flow Normal la Attack,
  Stage 2 buoc phai chon 1 loai tan cong (no khong co lua chon "Normal"). Ham
  predict_flow() danh dau nhung truong hop nay bang "gatekeeper_risk" thap gan
  nguong, de chatbot/UI co the canh bao do tin cay thap thay vi khang dinh chac chan.

Form dau vao (14 truong, mo rong so voi 7 truong ban dau theo yeu cau cua ban):
  proto, service, state,
  dur, spkts, dpkts, sbytes, dbytes,
  sloss, dloss, swin, dwin, ct_src_dport_ltm, ct_dst_sport_ltm

  Co tinh CHU Y khi chon mo rong: notebook cua ban da chung minh bang thuc nghiem
  rang tcprtt/synack/ackdat/stcpb/dtcpb la nhung dac trung LECH PHAN PHOI manh
  nhat giua train va test (KS-test), va bo chung giup Recall_Normal tang. Vi vay
  file nay KHONG dua 5 truong do tro lai form, du chung nam trong top feature
  importance. Cac truong duoc them (sloss, dloss, swin, dwin, ct_src_dport_ltm,
  ct_dst_sport_ltm) la cac chi so dem/kich thuoc it bi lech hon.

Chay huan luyen (tren Kaggle):
    python nids_pipeline_v2.py \
        --train /kaggle/input/datasets/dhoogla/unswnb15/UNSW_NB15_training-set.parquet \
        --test  /kaggle/input/datasets/dhoogla/unswnb15/UNSW_NB15_testing-set.parquet

Dung trong Streamlit:
    from nids_pipeline_v2 import load_bundle, predict_flow, risk_badge
    bundle = load_bundle("nids_bundle_v2.joblib")
    result = predict_flow(bundle, {...14 truong...})[0]
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------------------
# Cau hinh
# ----------------------------------------------------------------------------------------
CAT_COLS = ["proto", "service", "state"]
RAW_NUM_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "sloss", "dloss", "swin", "dwin",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
]
DERIVED_COLS = ["rate", "smean", "dmean", "sload", "dload"]
NUM_COLS = RAW_NUM_COLS + DERIVED_COLS
FEATURES = CAT_COLS + NUM_COLS              # dung chung cho ca 2 stage
USER_INPUT_COLS = CAT_COLS + RAW_NUM_COLS    # 14 truong nguoi dung nhap

TOP_N_PROTO = 6
NORMAL = "Normal"
GATE_THRESHOLD = 0.5     # nguong Stage 1: risk >= nguong -> coi la Attack
RANDOM_STATE = 42


# ----------------------------------------------------------------------------------------
# Dac trung (dung chung train/app)
# ----------------------------------------------------------------------------------------
def build_features(df: pd.DataFrame, top_protos: list[str]) -> pd.DataFrame:
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

    out["rate"] = np.where(has_dur, (spk + dpk - 1).clip(lower=0) / safe_dur, 0.0)
    out["smean"] = np.where(spk > 0, sby / spk.where(spk > 0, 1.0), 0.0)
    out["dmean"] = np.where(dpk > 0, dby / dpk.where(dpk > 0, 1.0), 0.0)
    out["sload"] = np.where(has_dur, sby * 8 / safe_dur, 0.0)
    out["dload"] = np.where(has_dur, dby * 8 / safe_dur, 0.0)
    return out[FEATURES]


# ----------------------------------------------------------------------------------------
# Pipeline / model
# ----------------------------------------------------------------------------------------
def make_pipeline(model) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    return Pipeline([("prep", prep), ("clf", model)])


def candidate_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
            random_state=RANDOM_STATE, eval_metric="logloss",
        ),
    }


def _weights(y) -> np.ndarray:
    return compute_sample_weight("balanced", y)


def _fit(pipe: Pipeline, X, y) -> Pipeline:
    pipe.fit(X, y, clf__sample_weight=_weights(y))
    return pipe


# ----------------------------------------------------------------------------------------
# Huan luyen
# ----------------------------------------------------------------------------------------
def train(tr: pd.DataFrame, te: pd.DataFrame, out_path: str = "nids_bundle_v2.joblib") -> dict:
    top_protos = (
        tr["proto"].astype(str).str.strip().str.lower().value_counts().head(TOP_N_PROTO).index.tolist()
    )
    Xtr_all = build_features(tr, top_protos)
    ytr_bin = tr["label"].astype(int).to_numpy()
    Xte_all = build_features(te, top_protos)
    yte_bin = te["label"].astype(int).to_numpy()

    # ============================== STAGE 1: BINARY ==============================
    Xa, Xv, ya, yv = train_test_split(
        Xtr_all, ytr_bin, test_size=0.2, stratify=ytr_bin, random_state=RANDOM_STATE
    )
    bin_scores = {}
    for name, model in candidate_models().items():
        pipe = _fit(make_pipeline(model), Xa, ya)
        pred_v = pipe.predict(Xv)
        # uu tien Recall_Normal vi day la metric bi anh huong nang nhat boi distribution shift
        bin_scores[name] = recall_score(yv, pred_v, pos_label=0)
        print(f"[val][Stage1] {name}: Recall_Normal={bin_scores[name]:.4f}")
    best_bin = max(bin_scores, key=bin_scores.get)
    print(f"[i] Stage1 chon: {best_bin}")

    stage1 = _fit(make_pipeline(candidate_models()[best_bin]), Xtr_all, ytr_bin)
    proba1 = stage1.predict_proba(Xte_all)
    normal_col = list(stage1.classes_).index(0)
    risk = 1.0 - proba1[:, normal_col]
    pred_bin = (risk >= GATE_THRESHOLD).astype(int)

    print("\n=== TEST Stage1 (Normal vs Attack) ===")
    print(classification_report(yte_bin, pred_bin, target_names=["Normal", "Attack"], digits=3))
    cm1 = confusion_matrix(yte_bin, pred_bin)
    print(pd.DataFrame(cm1, index=["Normal", "Attack"], columns=["pred_Normal", "pred_Attack"]))

    # ============================== STAGE 2: MULTICLASS (chi Attack) ==============================
    attack_mask_tr = tr["attack_cat"].astype(str) != NORMAL
    Xatk, yatk_raw = Xtr_all[attack_mask_tr], tr.loc[attack_mask_tr, "attack_cat"].astype(str)
    le = LabelEncoder().fit(yatk_raw)
    yatk = le.transform(yatk_raw)

    Xaa, Xav, yaa, yav = train_test_split(
        Xatk, yatk, test_size=0.2, stratify=yatk, random_state=RANDOM_STATE
    )
    mc_scores = {}
    for name, model in candidate_models().items():
        pipe = _fit(make_pipeline(model), Xaa, yaa)
        mc_scores[name] = f1_score(yav, pipe.predict(Xav), average="macro")
        print(f"[val][Stage2] {name}: macro-F1={mc_scores[name]:.4f}")
    best_mc = max(mc_scores, key=mc_scores.get)
    print(f"[i] Stage2 chon: {best_mc}")

    stage2 = _fit(make_pipeline(candidate_models()[best_mc]), Xatk, yatk)

    # Danh gia Stage2 CHI tren cac dong Attack that su cua test (dung nhu notebook cua ban)
    attack_mask_te = te["attack_cat"].astype(str) != NORMAL
    Xte_atk = Xte_all[attack_mask_te]
    yte_atk = le.transform(te.loc[attack_mask_te, "attack_cat"].astype(str))
    pred_atk = stage2.predict(Xte_atk)
    print("\n=== TEST Stage2 (chi tren Attack that) ===")
    print(classification_report(yte_atk, pred_atk, target_names=le.classes_, digits=3, zero_division=0))

    # Danh gia PIPELINE END-TO-END: chay Stage1 -> Stage2 tren TOAN BO test (ca Normal lan Attack)
    # de do loi cong don thuc te khi trien khai.
    final_label = np.where(pred_bin == 0, NORMAL, "")
    if pred_bin.sum() > 0:
        pred_atk_all = stage2.predict(Xte_all[pred_bin == 1])
        final_label[pred_bin == 1] = le.inverse_transform(pred_atk_all)
    true_label = te["attack_cat"].astype(str).to_numpy()
    e2e_acc = (final_label == true_label).mean()
    e2e_normal_recall = (final_label[true_label == NORMAL] == NORMAL).mean()
    print(f"\n=== END-TO-END (Stage1 -> Stage2, toan bo test) ===")
    print(f"Accuracy tong the (10 lop): {e2e_acc:.3f}")
    print(f"Recall Normal (khong bi Stage1 bao nham): {e2e_normal_recall:.3f}")

    bundle = {
        "stage1": stage1, "stage2": stage2, "label_encoder_stage2": le,
        "top_protos": top_protos, "features": FEATURES,
        "stage1_model": best_bin, "stage2_model": best_mc,
        "choices": {
            "proto": top_protos + ["other"],
            "service": sorted(tr["service"].astype(str).str.lower().unique().tolist()),
            "state": sorted(tr["state"].astype(str).str.upper().unique().tolist()),
        },
    }
    joblib.dump(bundle, out_path, compress=3)
    print(f"\n[i] Da luu: {out_path}")
    return bundle


# ----------------------------------------------------------------------------------------
# Suy luan (Streamlit)
# ----------------------------------------------------------------------------------------
def load_bundle(path: str = "nids_bundle_v2.joblib") -> dict:
    return joblib.load(path)


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in USER_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thieu cot: {missing}")
    nums = df[RAW_NUM_COLS].apply(pd.to_numeric, errors="coerce")
    if nums.isna().any().any():
        raise ValueError("Co gia tri so bi trong hoac khong hop le.")
    if (nums < 0).any().any():
        raise ValueError("Cac chi so khong duoc am.")
    if (nums["spkts"] < 1).any():
        raise ValueError("spkts phai >= 1.")


def risk_badge(risk: float) -> tuple[str, str]:
    if risk < 0.3:
        return "THAP", "green"
    if risk < 0.7:
        return "TRUNG BINH", "orange"
    return "CAO", "red"


def predict_flow(bundle: dict, flows) -> list[dict]:
    df = pd.DataFrame([flows]) if isinstance(flows, dict) else pd.DataFrame(flows).reset_index(drop=True)
    _validate(df)
    X = build_features(df, bundle["top_protos"])

    proba1 = bundle["stage1"].predict_proba(X)
    normal_col = list(bundle["stage1"].classes_).index(0)
    risk = 1.0 - proba1[:, normal_col]
    is_attack = risk >= GATE_THRESHOLD

    classes2 = list(bundle["label_encoder_stage2"].classes_)
    results = []
    for i in range(len(X)):
        badge, color = risk_badge(float(risk[i]))
        row = {
            "gatekeeper_risk": float(risk[i]),  # P(Attack) tu Stage 1
            "badge": badge,
            "color": color,
            "features": X.iloc[i].to_dict(),
        }
        if not is_attack[i]:
            row["label"] = NORMAL
            row["attack_type_confidence"] = None
            row["attack_probabilities"] = None
        else:
            p2 = bundle["stage2"].predict_proba(X.iloc[[i]])[0]
            k = int(np.argmax(p2))
            row["label"] = classes2[k]
            row["attack_type_confidence"] = float(p2[k])
            row["attack_probabilities"] = {c: float(p) for c, p in zip(classes2, p2)}
            # Canh bao: neu gatekeeper_risk chi vua qua nguong, day co the la 1 flow
            # Normal bi Stage1 bao nham, va nhan attack type o day khong dang tin cay.
            row["low_confidence_gate"] = bool(risk[i] < 0.65)
        results.append(row)
    return results


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="nids_bundle_v2.joblib")
    args = ap.parse_args()
    b = train(pd.read_parquet(args.train), pd.read_parquet(args.test), args.out)
    demo = pd.read_parquet(args.test).sample(5, random_state=1)
    for (_, row), res in zip(demo.iterrows(), predict_flow(b, demo)):
        print(f"thuc te={row['attack_cat']:<15} du doan={res['label']:<15} "
              f"gate_risk={res['gatekeeper_risk']:.0%} ({res['badge']})")
