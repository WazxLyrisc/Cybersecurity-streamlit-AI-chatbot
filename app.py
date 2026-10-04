from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ids_prediction_engine import load_bundle, predict_ids


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "unsw_nb15_ids_29feature_bundle.joblib"


st.set_page_config(
    page_title="UNSW-NB15 IDS",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang nạp model IDS...")
def get_bundle(path: str, modified_ns: int):
    # modified_ns is included in the cache key so replacing the model reloads it.
    del modified_ns
    return load_bundle(path)


st.title("🛡️ Network Intrusion Detection")
st.caption(
    "Prototype nghiên cứu dùng pipeline UNSW-NB15 hai stage: "
    "Normal/Attack → loại tấn công."
)

if not BUNDLE_PATH.is_file():
    st.error("Chưa tìm thấy model bundle.")
    st.markdown(
        "Đặt file **`unsw_nb15_ids_29feature_bundle.joblib`** cạnh `app.py` "
        "và `ids_prediction_engine.py`, rồi tải lại trang."
    )
    st.stop()

try:
    bundle = get_bundle(str(BUNDLE_PATH), BUNDLE_PATH.stat().st_mtime_ns)
except Exception as exc:
    st.error("Không nạp được model bundle.")
    st.exception(exc)
    st.stop()

metadata = bundle["metadata"]
required_features = metadata["required_input_features"]
numeric_features = metadata["binary_numeric_features"]
categorical_features = metadata["categorical_features"]

with st.sidebar:
    st.subheader("Model")
    st.write("UNSW-NB15 · 29 retained features")
    st.write("Stage 1: Normal / Attack")
    st.write("Stage 2: Attack category")
    st.caption(
        "Đây là prototype nghiên cứu. Confidence là điểm xác suất từ model "
        "và chưa được hiệu chuẩn."
    )

st.subheader("Tải dữ liệu flow")
st.write(
    "Tải CSV có đủ 29 feature đầu vào. Engine cũng nhận được CSV 34 cột gốc "
    "và tự bỏ qua 5 feature đã loại khi train."
)
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is None:
    st.info("Chọn một CSV để xem trước dữ liệu và chạy dự đoán.")
    st.stop()

raw_bytes = uploaded_file.getvalue()
upload_key = hashlib.sha256(raw_bytes).hexdigest()
if st.session_state.get("upload_key") != upload_key:
    st.session_state["upload_key"] = upload_key
    st.session_state.pop("prediction_result", None)

try:
    input_df = pd.read_csv(io.BytesIO(raw_bytes))
except (UnicodeDecodeError, pd.errors.ParserError) as exc:
    st.error(f"Không đọc được CSV: {exc}")
    st.stop()

if input_df.empty:
    st.warning("CSV không có dòng dữ liệu.")
    st.stop()

missing_features = [name for name in required_features if name not in input_df.columns]
if missing_features:
    st.error(f"CSV đang thiếu {len(missing_features)} feature bắt buộc:")
    st.code(", ".join(missing_features))
    st.stop()

st.caption(f"{len(input_df):,} dòng · {len(input_df.columns)} cột trong file")
with st.expander("Xem trước dữ liệu", expanded=False):
    st.dataframe(input_df.head(20), use_container_width=True)

if st.button("Phân tích dữ liệu", type="primary", use_container_width=True):
    work_df = input_df.copy()
    numeric = work_df.loc[:, numeric_features].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna()
    if invalid_numeric.any().any():
        bad_columns = invalid_numeric.columns[invalid_numeric.any()].tolist()
        st.error(
            "Có giá trị số trống hoặc không hợp lệ ở các cột: "
            + ", ".join(bad_columns)
        )
        st.stop()
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        st.error("Dữ liệu số có giá trị vô hạn; hãy sửa CSV rồi thử lại.")
        st.stop()
    work_df.loc[:, numeric_features] = numeric

    null_categories = work_df.loc[:, categorical_features].isna()
    if null_categories.any().any():
        bad_columns = null_categories.columns[null_categories.any()].tolist()
        st.error(
            "Có category trống ở các cột: " + ", ".join(bad_columns)
        )
        st.stop()

    try:
        with st.spinner("Đang chạy hai stage..."):
            predictions = predict_ids(work_df, bundle)
        results = input_df.copy()
        for column in predictions.columns:
            results[column] = predictions[column].to_numpy()
        st.session_state["prediction_result"] = results
    except Exception as exc:
        st.error("Không thể dự đoán từ CSV này.")
        st.exception(exc)

results = st.session_state.get("prediction_result")
if results is not None and st.session_state.get("upload_key") == upload_key:
    st.subheader("Kết quả")
    counts = results["prediction"].value_counts()
    normal_count = int(counts.get("Normal", 0))
    attack_count = int(counts.get("Attack", 0))
    col1, col2, col3 = st.columns(3)
    col1.metric("Flows đã phân tích", f"{len(results):,}")
    col2.metric("Cảnh báo Attack", f"{attack_count:,}")
    col3.metric("Dự đoán Normal", f"{normal_count:,}")

    st.bar_chart(counts.rename_axis("prediction"))
    st.dataframe(results, use_container_width=True, hide_index=True)

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Tải kết quả CSV",
        data=csv_bytes,
        file_name="ids_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Một flow có thể được cảnh báo là Attack nhưng phân loại sai attack_type. "
        "Các chỉ số confidence chưa được hiệu chuẩn; không xem chúng là bảo đảm "
        "model đúng."
    )
