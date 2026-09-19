import streamlit as st
import joblib
import pandas as pd

# Load package
pkg = joblib.load('nids_xgb_model.pkl')

st.title("Hệ thống phát hiện xâm nhập mạng (NIDS)")

# Form nhập thông số từ người dùng
proto = st.selectbox("Protocol", pkg['encoders']['proto'].classes_)
service = st.selectbox("Service", pkg['encoders']['service'].classes_)
state = st.selectbox("State", pkg['encoders']['state'].classes_)
dur = st.number_input("Duration", value=0.0)
sbytes = st.number_input("Source Bytes", value=0)
dbytes = st.number_input("Destination Bytes", value=0)
spkts = st.number_input("Source Packets", value=0)
sttl = st.number_input("Source TTL", value=64)

if st.button("Dự đoán"):
    # Tạo DataFrame từ Input
    input_df = pd.DataFrame([[proto, service, state, dur, sbytes, dbytes, spkts, sttl]], 
                            columns=pkg['selected_features'])
    
    # Encode & Scale dữ liệu nhập vào
    for col in pkg['categorical_cols']:
        input_df[col] = pkg['encoders'][col].transform(input_df[col])
    input_df[pkg['numerical_cols']] = pkg['scaler'].transform(input_df[pkg['numerical_cols']])
    
    # Dự đoán
    pred = pkg['model'].predict(input_df)[0]
    if pred == 1:
        st.error("⚠️ Cảnh báo: Phát hiện lưu lượng MẠNG BẤT THƯỜNG (Attack)!")
    else:
        st.success("✅ Trạng thái: Lưu lượng MẠNG BÌNH THƯỜNG (Normal).")