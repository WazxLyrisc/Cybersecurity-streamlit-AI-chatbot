"""
app.py - Mini SOC Dashboard (Streamlit)

Chay:
    pip install streamlit joblib pandas numpy scikit-learn xgboost google-genai
    streamlit run app.py

Yeu cau:
- File nids_pipeline_v2.py va nids_bundle_v2.joblib phai nam CUNG THU MUC voi app.py.
- Bien moi truong GEMINI_API_KEY (hoac nhap truc tiep o sidebar khi chay thu).
  Lay API key tai: https://aistudio.google.com/apikey
"""
import os

import pandas as pd
import streamlit as st

from nids_pipeline_v2 import (
    CAT_COLS,
    RAW_NUM_COLS,
    load_bundle,
    predict_flow,
    risk_badge,
)

BUNDLE_PATH = "nids_bundle_v2.joblib"

st.set_page_config(page_title="Mini SOC Dashboard", layout="wide")


# ==========================================================================================
# Load model (cache de khong load lai moi lan tuong tac)
# ==========================================================================================
@st.cache_resource
def get_bundle():
    return load_bundle(BUNDLE_PATH)


try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error(
        f"Khong tim thay {BUNDLE_PATH}. Hay dat file nay cung thu muc voi app.py "
        "(xem huong dan trong docstring dau file)."
    )
    st.stop()

CHOICES = bundle["choices"]


# ==========================================================================================
# Sidebar: API key cho Gemini
# ==========================================================================================
with st.sidebar:
    st.header("Cau hinh")
    api_key = st.text_input(
        "Gemini API key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Lay tai https://aistudio.google.com/apikey. "
        "Co the dat san bien moi truong GEMINI_API_KEY thay vi nhap moi lan.",
    )
    st.caption(
        f"Model dang dung — Stage 1: {bundle.get('stage1_model', '?')} | "
        f"Stage 2: {bundle.get('stage2_model', '?')}"
    )


# ==========================================================================================
# Ham goi Gemini
# ==========================================================================================
SYSTEM_PROMPT = """Ban la mot chuyen gia phan tich an ninh mang (SOC Analyst) day kinh nghiem,
dang ho tro cho mot quan tri vien dang xem Mini SOC Dashboard.

QUY TAC BAT BUOC:
- CHI duoc phan tich dua tren cac chi so duoc cung cap trong NGU CANH ben duoi. KHONG duoc
  bia them so lieu, KHONG doan mo rong ngoai nhung gi da cho.
- Neu mot chi so nao do khong duoc cung cap, hay noi ro la khong co du lieu ve chi so do,
  dung tu suy dien.
- Tra loi bang tieng Viet, ngan gon, ro rang, dung dinh dang gach dau dong khi liet ke.
- Vai tro cua ban la GIAI THICH ket qua va TU VAN quy trinh xu ly (vi du: co nen chan IP,
  gioi han rate limit, cach ly may chu, thu thap them log...). Ban KHONG phai la nguoi
  ra quyet dinh cuoi cung — luon nhac quan tri vien tu xac minh truoc khi hanh dong,
  vi day la du doan cua mot mo hinh Machine Learning, co the sai."""


def build_context(input_values: dict, result: dict) -> str:
    lines = ["=== Thong so luong mang (nguoi dung nhap) ==="]
    for k, v in input_values.items():
        lines.append(f"- {k}: {v}")

    lines.append("\n=== Ket qua tu Model ML (2 tang) ===")
    lines.append(f"- Nhan du doan: {result['label']}")
    lines.append(f"- Muc do rui ro (P(Attack) tu Stage 1 - binary gatekeeper): {result['gatekeeper_risk']:.1%}")
    lines.append(f"- Badge canh bao: {result['badge']}")

    if result["label"] != "Normal":
        lines.append(f"- Do tin cay vao loai tan cong cu the (Stage 2): {result['attack_type_confidence']:.1%}")
        if result.get("low_confidence_gate"):
            lines.append(
                "- CANH BAO NOI BO: risk cua Stage 1 chi vua qua nguong, co kha nang day la "
                "mot flow Normal bi bao nham thanh Attack. Hay de cap muc do khong chac chan nay "
                "trong cau tra loi thay vi khang dinh chac chan la tan cong."
            )
        top3 = sorted(result["attack_probabilities"].items(), key=lambda x: -x[1])[:3]
        lines.append("- Top 3 loai tan cong co xac suat cao nhat: " + ", ".join(f"{c} ({p:.1%})" for c, p in top3))

    return "\n".join(lines)


def call_gemini(api_key: str, context: str, chat_history: list, user_message: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents = []
    for turn in chat_history:
        contents.append(
            types.Content(role=turn["role"], parts=[types.Part.from_text(text=turn["text"])])
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"NGU CANH HIEN TAI:\n{context}\n\nCAU HOI: {user_message}")],
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.3),
    )
    return response.text


# ==========================================================================================
# Layout chinh: 2 cot
# ==========================================================================================
left, right = st.columns([1.1, 0.9])

# ------------------------------------------------------------------------------------------
# COT TRAI: Input + Model Output
# ------------------------------------------------------------------------------------------
with left:
    st.subheader("Input & Model Output")
    tab_form, tab_csv = st.tabs(["Nhap tay", "Tai file CSV"])

    with tab_form:
        with st.form("flow_form"):
            c1, c2, c3 = st.columns(3)
            proto = c1.selectbox("proto", CHOICES["proto"])
            service = c2.selectbox("service", CHOICES["service"])
            state = c3.selectbox("state", CHOICES["state"])

            c1, c2, c3, c4 = st.columns(4)
            dur = c1.number_input("dur (giay)", min_value=0.0, value=1.0, format="%.6f")
            spkts = c2.number_input("spkts", min_value=1, value=10, step=1)
            dpkts = c3.number_input("dpkts", min_value=0, value=8, step=1)
            sbytes = c4.number_input("sbytes", min_value=0, value=800, step=1)

            c1, c2, c3, c4 = st.columns(4)
            dbytes = c1.number_input("dbytes", min_value=0, value=400, step=1)
            sloss = c2.number_input("sloss", min_value=0, value=0, step=1)
            dloss = c3.number_input("dloss", min_value=0, value=0, step=1)
            swin = c4.number_input("swin", min_value=0, value=255, step=1)

            c1, c2, c3 = st.columns(3)
            dwin = c1.number_input("dwin", min_value=0, value=255, step=1)
            ct_src = c2.number_input("ct_src_dport_ltm", min_value=0, value=1, step=1)
            ct_dst = c3.number_input("ct_dst_sport_ltm", min_value=0, value=1, step=1)

            submitted = st.form_submit_button("Phan tich luong mang", type="primary")

        if submitted:
            values = {
                "proto": proto, "service": service, "state": state,
                "dur": dur, "spkts": spkts, "dpkts": dpkts,
                "sbytes": sbytes, "dbytes": dbytes,
                "sloss": sloss, "dloss": dloss, "swin": swin, "dwin": dwin,
                "ct_src_dport_ltm": ct_src, "ct_dst_sport_ltm": ct_dst,
            }
            try:
                result = predict_flow(bundle, values)[0]
                st.session_state["last_input"] = values
                st.session_state["last_result"] = result
                st.session_state.setdefault("chat_history", [])
                st.session_state["chat_history"] = []  # reset hoi thoai khi co phan tich moi
            except ValueError as e:
                st.error(f"Du lieu khong hop le: {e}")

    with tab_csv:
        st.caption(
            "File CSV can co du cac cot: " + ", ".join(CAT_COLS + RAW_NUM_COLS)
        )
        up = st.file_uploader("Tai file .csv", type=["csv"])
        if up is not None:
            try:
                df_in = pd.read_csv(up)
                results = predict_flow(bundle, df_in)
                out_df = df_in.copy()
                out_df["prediction"] = [r["label"] for r in results]
                out_df["risk_%"] = [round(r["gatekeeper_risk"] * 100, 1) for r in results]
                out_df["badge"] = [r["badge"] for r in results]
                st.dataframe(out_df, use_container_width=True)
                st.download_button(
                    "Tai ket qua (.csv)",
                    out_df.to_csv(index=False).encode("utf-8"),
                    file_name="ket_qua_phan_tich.csv",
                    mime="text/csv",
                )
                # Cho batch mode: dung dong dau tien lam context cho chatbot
                st.session_state["last_input"] = df_in.iloc[0].to_dict()
                st.session_state["last_result"] = results[0]
                st.session_state.setdefault("chat_history", [])
            except ValueError as e:
                st.error(f"Du lieu khong hop le: {e}")
            except Exception as e:  # loi doc file / thieu cot
                st.error(f"Khong the xu ly file: {e}")

    # Hien thi ket qua gan nhat
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        badge_colors = {"green": "#2e7d32", "orange": "#e65100", "red": "#c62828"}
        color = badge_colors.get(r["color"], "#616161")

        st.markdown("---")
        st.markdown(
            f"""
            <div style="padding:16px;border-radius:10px;border:2px solid {color};">
                <span style="font-size:1.4em;font-weight:700;color:{color};">{r['badge']}</span>
                &nbsp;&nbsp;<b>Nhan:</b> {r['label']}
                &nbsp;&nbsp;<b>Muc do rui ro:</b> {r['gatekeeper_risk']:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if r["label"] != "Normal":
            if r.get("low_confidence_gate"):
                st.warning(
                    "Risk vua qua nguong — co kha nang day la Normal bi bao nham. "
                    "Nen kiem tra them truoc khi hanh dong."
                )
            st.caption(f"Do tin cay vao loai tan cong: {r['attack_type_confidence']:.1%}")
            probs = pd.Series(r["attack_probabilities"]).sort_values(ascending=False)
            st.bar_chart(probs)

# ------------------------------------------------------------------------------------------
# COT PHAI: AI Chatbot
# ------------------------------------------------------------------------------------------
with right:
    st.subheader("Tro ly SOC (AI)")

    if "last_result" not in st.session_state:
        st.info("Hay phan tich mot luong mang o cot ben trai truoc, chatbot se dua vao ket qua do de tu van.")
    else:
        st.session_state.setdefault("chat_history", [])

        for turn in st.session_state["chat_history"]:
            role = "user" if turn["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(turn["text"])

        user_msg = st.chat_input("Hoi ve ket qua canh bao nay...")
        if user_msg:
            if not api_key:
                st.error("Vui long nhap Gemini API key o sidebar.")
            else:
                with st.chat_message("user"):
                    st.markdown(user_msg)
                context = build_context(st.session_state["last_input"], st.session_state["last_result"])
                with st.chat_message("assistant"):
                    with st.spinner("Dang phan tich..."):
                        try:
                            reply = call_gemini(api_key, context, st.session_state["chat_history"], user_msg)
                        except Exception as e:
                            reply = f"Loi khi goi Gemini API: {e}"
                    st.markdown(reply)
                st.session_state["chat_history"].append({"role": "user", "text": user_msg})
                st.session_state["chat_history"].append({"role": "model", "text": reply})