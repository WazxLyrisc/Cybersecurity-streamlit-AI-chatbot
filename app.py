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


def make_demo_flows(required_features: list[str]) -> pd.DataFrame:
    """Create clearly labeled illustrative flow records for a one-click demo.

    These are synthetic examples for trying the interface, not measured traffic
    and not a substitute for representative, labeled evaluation data.
    """
    profiles = [
        {
            "proto": "tcp",
            "service": "ssh",
            "state": "CON",
            "dur": 1.2,
            "spkts": 8,
            "dpkts": 7,
            "sbytes": 640,
            "dbytes": 980,
            "rate": 12.5,
            "sload": 4266.7,
            "dload": 6533.3,
            "sinpkt": 0.15,
            "dinpkt": 0.17,
            "swin": 255,
            "dwin": 255,
            "smean": 80,
            "dmean": 140,
            "ct_src_dport_ltm": 1,
            "ct_dst_sport_ltm": 1,
        },
        {
            "proto": "udp",
            "service": "-",
            "state": "INT",
            "dur": 0.01,
            "spkts": 1,
            "dpkts": 0,
            "sbytes": 46,
            "dbytes": 0,
            "rate": 100.0,
            "sload": 36800.0,
            "dload": 0.0,
            "sinpkt": 0.01,
            "dinpkt": 0.0,
            "ct_src_dport_ltm": 10,
            "ct_dst_sport_ltm": 1,
        },
    ]
    return pd.DataFrame(
        [{feature: profile.get(feature, 0) for feature in required_features}
         for profile in profiles],
        columns=required_features,
    )


st.title("🛡️ Network Intrusion Detection")
st.caption(
    "Prototype nghiên cứu dùng pipeline UNSW-NB15 hai stage: "
    "Normal/Attack → loại tấn công. Có thể chạy thử mà chưa cần chuẩn bị CSV."
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

with st.expander("Mô hình này phân tích gì?", expanded=False):
    st.write(
        "29 đặc trưng là các thông tin mô tả một network flow, không phải 29 "
        "loại tấn công. Ví dụ: giao thức và dịch vụ, trạng thái kết nối, thời "
        "lượng, số gói tin/byte gửi và nhận, tốc độ, tải mạng, độ trễ và số "
        "kết nối gần đây. Mô hình dùng chúng để dự đoán flow là Normal hay "
        "Attack; nếu là Attack thì dự đoán thêm nhóm tấn công."
    )
    st.info(
        "App hiện chưa quét URL hay theo dõi mạng trực tiếp. CSV cần chứa các "
        "flow đã được chuyển thành feature theo định dạng UNSW-NB15."
    )

mode = st.radio(
    "Bạn muốn bắt đầu thế nào?",
    ["Chạy thử demo", "Phân tích CSV của tôi"],
    horizontal=True,
)

if mode == "Chạy thử demo":
    st.subheader("Chạy thử bằng dữ liệu mô phỏng")
    input_df = make_demo_flows(required_features)
    st.caption(
        "Hai flow bên dưới được tạo để minh họa thao tác của app; đây không "
        "phải traffic thật và kết quả không dùng để đánh giá mô hình."
    )
    with st.expander("Xem 2 flow demo", expanded=True):
        st.dataframe(input_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Tải CSV demo",
        data=input_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ids_demo_flows.csv",
        mime="text/csv",
    )
    source_key = "demo-v1"
else:
    st.subheader("Phân tích dữ liệu flow của bạn")
    st.write(
        "Tải CSV có đủ 29 feature đầu vào. CSV 34 cột gốc của UNSW-NB15 "
        "cũng được chấp nhận; 5 feature đã loại sẽ được bỏ qua."
    )
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is None:
        st.info(
            "Chưa có CSV? Chọn **Chạy thử demo** để xem app hoạt động, hoặc "
            "chuẩn bị một CSV flow theo định dạng UNSW-NB15."
        )
        st.stop()

    raw_bytes = uploaded_file.getvalue()
    source_key = "csv-" + hashlib.sha256(raw_bytes).hexdigest()
    try:
        input_df = pd.read_csv(io.BytesIO(raw_bytes))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Không đọc được CSV: {exc}")
        st.stop()

if st.session_state.get("source_key") != source_key:
    st.session_state["source_key"] = source_key
    st.session_state.pop("prediction_result", None)

if input_df.empty:
    st.warning("CSV không có dòng dữ liệu.")
    st.stop()

missing_features = [name for name in required_features if name not in input_df.columns]
if missing_features:
    st.error(f"CSV đang thiếu {len(missing_features)} feature bắt buộc:")
    st.code(", ".join(missing_features))
    st.stop()

st.caption(f"{len(input_df):,} dòng · {len(input_df.columns)} cột trong file")
if mode != "Chạy thử demo":
    with st.expander("Xem trước dữ liệu", expanded=False):
        st.dataframe(input_df.head(20), use_container_width=True)

button_label = "Chạy dự đoán demo" if mode == "Chạy thử demo" else "Phân tích CSV"
if st.button(button_label, type="primary", use_container_width=True):
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
if results is not None and st.session_state.get("source_key") == source_key:
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
