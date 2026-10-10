from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ids_prediction_engine import load_bundle, predict_ids


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "unsw_nb15_ids_29feature_bundle.joblib"

# Các nhóm tấn công có độ chính xác thấp trong đánh giá nội bộ (precision/recall/F1
# đều dưới ~0.3 trên tập test), nên chatbot cần nhắc thêm mức độ không chắc chắn khi
# gặp các nhãn này.
LOW_RELIABILITY_CLASSES = {"Analysis", "Backdoor", "DoS"}

# Giải thích bằng ngôn ngữ phổ thông cho 9 nhóm tấn công của UNSW-NB15 — dùng cả
# cho mục tra cứu chung và để "ghim" chatbot vào đúng định nghĩa, tránh việc
# Gemini tự suy diễn hoặc giải thích sai một thuật ngữ chuyên ngành.
ATTACK_TYPE_INFO = {
    "Analysis": {
        "label": "Analysis · Dò xét ứng dụng web",
        "what": "Kẻ tấn công rà soát một trang/ứng dụng web để tìm điểm yếu — ví dụ "
        "thử chèn mã vào ô nhập liệu, hoặc dò các cổng dịch vụ đang mở.",
        "danger": "Thường là bước thăm dò ban đầu, chưa gây hại trực tiếp nhưng là "
        "dấu hiệu cảnh báo sớm cần theo dõi.",
    },
    "Backdoor": {
        "label": "Backdoor · Cửa hậu",
        "what": "Kẻ tấn công cài một lối vào bí mật trên máy/hệ thống, để sau này "
        "quay lại truy cập mà không cần đăng nhập bình thường.",
        "danger": "Mức độ cao — nếu đúng là cửa hậu, kẻ tấn công có thể ra vào hệ "
        "thống bất cứ lúc nào mà quản trị viên không hay biết.",
    },
    "DoS": {
        "label": "DoS · Làm gián đoạn dịch vụ",
        "what": "Gửi dồn dập yêu cầu/giao tiếp khiến hệ thống quá tải, không còn "
        "phục vụ được người dùng thật (Denial of Service).",
        "danger": "Mức độ cao đối với tính sẵn sàng của dịch vụ — người dùng hợp "
        "lệ không truy cập được trong lúc bị tấn công.",
    },
    "Exploits": {
        "label": "Exploits · Khai thác lỗ hổng",
        "what": "Kẻ tấn công tận dụng một lỗ hổng bảo mật cụ thể, đã biết trước, "
        "trong phần mềm hoặc hệ điều hành để chiếm quyền kiểm soát.",
        "danger": "Mức độ cao — có thể dẫn tới chiếm quyền điều khiển một phần "
        "hoặc toàn bộ hệ thống.",
    },
    "Fuzzers": {
        "label": "Fuzzers · Thử dữ liệu bất thường",
        "what": "Gửi dữ liệu ngẫu nhiên hoặc dị dạng vào hệ thống để thử tìm lỗi "
        "hoặc làm treo chương trình.",
        "danger": "Thường ở mức thăm dò/thử nghiệm, nhưng có thể là bước mở đầu "
        "cho một cuộc tấn công nghiêm trọng hơn nếu tìm được lỗ hổng.",
    },
    "Generic": {
        "label": "Generic · Tấn công mã hoá tổng quát",
        "what": "Một kỹ thuật tấn công tổng quát nhắm vào thuật toán mã hoá, "
        "không phụ thuộc vào cấu hình cụ thể của hệ thống đang chạy.",
        "danger": "Mức độ tuỳ ngữ cảnh sử dụng — đây là nhóm có số lượng mẫu lớn "
        "nhất trong dữ liệu huấn luyện nên mô hình nhận diện khá tốt.",
    },
    "Reconnaissance": {
        "label": "Reconnaissance · Do thám",
        "what": "Thu thập thông tin về hệ thống — quét cổng, dò tìm dịch vụ đang "
        "chạy — để chuẩn bị cho một cuộc tấn công sau này.",
        "danger": "Mức độ thấp trực tiếp, nhưng là tín hiệu cảnh báo có khả năng "
        "sắp xảy ra tấn công nghiêm trọng hơn.",
    },
    "Shellcode": {
        "label": "Shellcode · Mã khai thác",
        "what": "Một đoạn mã nhỏ được 'bơm' vào hệ thống ngay sau khi một lỗ hổng "
        "đã bị khai thác thành công, dùng để chiếm quyền điều khiển.",
        "danger": "Mức độ cao — thường xuất hiện ngay sau bước Exploits, cho thấy "
        "cuộc tấn công đã tiến khá xa.",
    },
    "Worms": {
        "label": "Worms · Sâu máy tính",
        "what": "Phần mềm độc hại tự nhân bản và lây lan sang các máy khác trong "
        "mạng mà không cần người dùng thao tác gì thêm.",
        "danger": "Mức độ cao — có khả năng lan rộng nhanh trong toàn bộ mạng nếu "
        "không được cách ly kịp thời.",
    },
}

CHATBOT_SYSTEM_PROMPT = """Ban la chuyen gia phan tich an ninh mang (SOC Analyst) ho tro
quan tri vien dang xem ket qua tu he thong IDS hai tang (Binary Normal/Attack, sau do
Multiclass phan loai nhom tan cong) huan luyen tren bo du lieu nghien cuu UNSW-NB15.

QUY TAC BAT BUOC:
- CHI duoc phan tich dua tren 29 dac trung va ket qua mo hinh duoc cung cap trong
  NGU CANH ben duoi. KHONG bia them so lieu, KHONG suy doan ngoai nhung gi da cho.
- Day la "diem mo hinh" (model score) CHUA duoc hieu chuan (uncalibrated), khong phai
  xac suat dam bao dung. Luon nhac dieu nay khi nguoi dung hoi ve "do chac chan".
- Neu nhan "Nhom co do tin cay thap trong danh gia noi bo" xuat hien trong ngu canh,
  PHAI nhac ro rang ket qua nay can duoc con nguoi xac minh ky hon, vi nhom tan cong
  nay thuong bi mo hinh nham lan voi cac nhom khac trong qua trinh danh gia.
- Vai tro cua ban la GIAI THICH tai sao luong du lieu nay bi gan nhan nhu vay (dua
  tren cac dac trung bat thuong) va TU VAN cac buoc kiem tra/xu ly tiep theo ở muc
  do SOC (vi du: doi chieu log goc, kiem tra IP nguon, co nen tam thoi gioi han toc
  do hay cach ly khong). Ban KHONG phai nguoi ra quyet dinh cuoi cung.
- Tra loi bang tieng Viet, ngan gon, ro rang, dung gach dau dong khi liet ke."""


def build_chat_context(row: pd.Series, required_features: list[str]) -> str:
    lines = ["=== 29 dac trung cua luong du lieu duoc chon ==="]
    for feature in required_features:
        lines.append(f"- {feature}: {row[feature]}")

    lines.append("\n=== Ket qua tu he thong IDS 2 tang ===")
    lines.append(f"- Nhan tong quat (Stage 1): {row['prediction']}")
    lines.append(f"- Diem mo hinh ve kha nang la Attack: {row['attack_probability']:.1%}")
    lines.append(f"- Do tin cay vao nhan du doan (Stage 1): {row['binary_confidence']:.1%}")

    if row["prediction"] == "Attack":
        attack_type = row["attack_type"]
        lines.append(f"- Nhom tan cong du doan (Stage 2): {attack_type}")
        if pd.notna(row.get("attack_confidence")):
            lines.append(f"- Do tin cay vao nhom tan cong (Stage 2): {row['attack_confidence']:.1%}")
        info = ATTACK_TYPE_INFO.get(attack_type)
        if info:
            lines.append(
                f"- Dinh nghia nhom '{attack_type}' (dung DUNG dinh nghia nay, khong "
                f"tu suy dien them): {info['what']} Muc do nguy hiem thuong gap: {info['danger']}"
            )
        if attack_type in LOW_RELIABILITY_CLASSES:
            lines.append(
                "- CANH BAO NOI BO: day la nhom co do tin cay thap trong danh gia noi bo "
                "(precision/recall deu thap, de bi nham lan voi cac nhom khac nhu DoS/"
                "Backdoor/Exploits). Hay nhac nguoi dung xac minh them."
            )

    return "\n".join(lines)


def get_shared_api_key() -> str:
    """Key do người triển khai app (dev) cấu hình sẵn — không yêu cầu người xem
    tự có/tự nhập API key. Ưu tiên Streamlit `secrets.toml` (dùng khi deploy lên
    Streamlit Community Cloud: Settings → Secrets), sau đó mới tới biến môi
    trường GEMINI_API_KEY (dùng khi tự host bằng Docker/server riêng)."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        # st.secrets ném lỗi nếu chưa có file secrets.toml nào cả — bỏ qua, coi
        # như chưa cấu hình, rồi thử biến môi trường.
        key = ""
    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")
    return key


def call_gemini(api_key: str, context: str, chat_history: list, user_message: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(role=turn["role"], parts=[types.Part.from_text(text=turn["text"])])
        for turn in chat_history
    ]
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"NGU CANH HIEN TAI:\n{context}\n\nCAU HOI: {user_message}")],
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=CHATBOT_SYSTEM_PROMPT, temperature=0.3),
    )
    return response.text


st.set_page_config(
    page_title="UNSW-NB15 IDS",
    page_icon="🛡️",
    layout="wide",
)

# ==============================================================================
# Giao diện "SOC console": font + CSS tùy chỉnh. Màu theme nền/chữ chính nằm ở
# .streamlit/config.toml — khối này chỉ bổ sung những gì config.toml không làm
# được (font chữ ngoài, style card cho st.metric, badge rủi ro, khung chat).
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Tiêu đề chính đậm hơn, đúng tinh thần "console" */
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }

    /* 3 st.metric đầu kết quả -> dạng card có viền, số liệu dùng font monospace
       cho cảm giác "terminal" và dễ đọc số */
    div[data-testid="stMetric"] {
        background-color: #141A21;
        border: 1px solid #263140;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* Khung chat: viền riêng để tách bạch rõ với bảng kết quả phía trên */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    /* Badge mức độ rủi ro — 3 mức, nền đặc + chữ tối màu để luôn đạt tương phản
       AA bất kể theme sáng/tối, không chỉ dựa vào màu (có icon + chữ kèm theo) */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge.low    { background-color: #86EFAC; color: #052e16; }
    .risk-badge.medium { background-color: #FDE68A; color: #451a03; }
    .risk-badge.high   { background-color: #FCA5A5; color: #450a0a; }
    </style>
    """,
    unsafe_allow_html=True,
)


def risk_tier(attack_probability: float) -> tuple[str, str, str]:
    """3 mức rủi ro dùng chung cho cả bảng kết quả và badge trong khung chat."""
    if attack_probability < 0.3:
        return "low", "✅", "THẤP"
    if attack_probability < 0.7:
        return "medium", "⚠️", "TRUNG BÌNH"
    return "high", "🚨", "CAO"


def risk_badge_html(attack_probability: float) -> str:
    level, icon, label = risk_tier(attack_probability)
    label_full = {"low": "RỦI RO THẤP", "medium": "RỦI RO TRUNG BÌNH", "high": "RỦI RO CAO"}[level]
    return f'<span class="risk-badge {level}">{icon} {label_full} · {attack_probability:.0%}</span>'


def risk_badge_text(attack_probability: float) -> str:
    """Phiên bản văn bản thuần (icon + chữ + %) để dùng trong st.dataframe,
    nơi không render được HTML tuỳ ý."""
    _, icon, label = risk_tier(attack_probability)
    return f"{attack_probability:.0%} · {icon} {label}"


RISK_BG = {"low": "#86EFAC", "medium": "#FDE68A", "high": "#FCA5A5"}
RISK_FG = {"low": "#052e16", "medium": "#451a03", "high": "#450a0a"}


def style_risk_column(styler, column: str, probabilities: pd.Series):
    """Tô màu nền cho đúng 1 cột 'Điểm mô hình' trong bảng kết quả, theo 3 mức
    rủi ro — màu chỉ là lớp bổ sung, chữ + icon ở trên đã truyền đạt đủ ý nghĩa."""
    def _color(_value, prob):
        level, _, _ = risk_tier(prob)
        return f"background-color: {RISK_BG[level]}; color: {RISK_FG[level]}; font-weight: 700;"

    styles = pd.DataFrame("", index=styler.data.index, columns=styler.data.columns)
    styles[column] = [
        f"background-color: {RISK_BG[risk_tier(p)[0]]}; color: {RISK_FG[risk_tier(p)[0]]}; font-weight: 700;"
        for p in probabilities
    ]
    return styler.apply(lambda _df: styles, axis=None)


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


st.title("🛡️ Kiểm tra dữ liệu kết nối mạng")
st.caption(
    "Bản minh họa nghiên cứu: thử ví dụ có sẵn hoặc phân tích tệp dữ liệu mạng. "
    "Chưa có tệp? Hãy bắt đầu bằng bản dùng thử."
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

SHARED_API_KEY = get_shared_api_key()

with st.sidebar:
    st.header("Trợ lý AI")

    if SHARED_API_KEY:
        # Trường hợp bình thường: người triển khai app đã cấu hình sẵn key dùng
        # chung, người xem không cần biết "API key" là gì.
        st.success("Trợ lý AI đã sẵn sàng — không cần cấu hình gì thêm.")
        with st.expander("Dùng API key riêng của bạn (tuỳ chọn, nâng cao)"):
            st.caption(
                "Chỉ cần nếu bạn muốn dùng hạn mức Gemini của riêng mình thay vì "
                "key dùng chung của app này."
            )
            custom_api_key = st.text_input(
                "Gemini API key của bạn",
                type="password",
                help="Lấy miễn phí tại https://aistudio.google.com/apikey",
            )
        gemini_api_key = custom_api_key or SHARED_API_KEY
    else:
        # App chưa được cấu hình key dùng chung — đây là việc của người triển
        # khai (đặt trong Secrets của Streamlit Cloud hoặc biến môi trường
        # GEMINI_API_KEY), không phải việc người xem thông thường phải lo.
        st.warning(
            "Trợ lý AI chưa được cấu hình key dùng chung cho app này.",
            icon="⚠️",
        )
        with st.expander("Dành cho người triển khai / nâng cao"):
            st.caption(
                "Cách khuyến nghị: thêm `GEMINI_API_KEY = \"...\"` vào "
                "**Settings → Secrets** trên Streamlit Community Cloud (hoặc biến "
                "môi trường `GEMINI_API_KEY` khi tự host), rồi reload app — mọi "
                "người xem sẽ dùng chung key đó, không ai phải tự nhập."
            )
            gemini_api_key = st.text_input(
                "Hoặc nhập tạm một Gemini API key để thử ngay",
                type="password",
                help="Lấy tại https://aistudio.google.com/apikey",
            )

    with st.expander("Thông tin kỹ thuật", expanded=False):
        st.write("Dữ liệu huấn luyện: UNSW-NB15")
        st.write("Mô hình dùng 29 đặc trưng của mỗi network flow.")
        st.caption(
            "Đây là prototype nghiên cứu. Điểm mô hình chưa được hiệu chuẩn "
            "và không phải xác suất bảo đảm."
        )

with st.expander("Mô hình này phân tích gì?", expanded=False):
    st.write(
        "Mô hình xem thông tin tóm tắt của một lượt giao tiếp giữa các thiết bị "
        "trên mạng: dùng giao thức/dịch vụ nào, kéo dài bao lâu, gửi nhận bao "
        "nhiêu gói tin và dữ liệu, cùng một số dấu hiệu về nhịp độ kết nối. "
        "Đó là 29 đặc trưng đầu vào, không phải 29 loại tấn công."
    )
    st.info(
        "Ứng dụng hiện không quét địa chỉ website và không theo dõi mạng trực "
        "tiếp. Để phân tích dữ liệu của bạn, cần tệp CSV đã được chuẩn bị theo "
        "định dạng UNSW-NB15."
    )

with st.expander("📖 9 nhóm tấn công là gì? (dành cho người mới)", expanded=False):
    st.caption(
        "Mô hình chỉ xếp một kết nối vào 1 trong 9 nhóm dưới đây (hoặc Normal). "
        "Tên gọi là thuật ngữ chuyên ngành trong bộ dữ liệu nghiên cứu, không "
        "phải tên gọi phổ biến hàng ngày — bấm vào từng nhóm để xem giải thích."
    )
    for attack_type, info in ATTACK_TYPE_INFO.items():
        reliability_note = (
            " ⚠️ *Nhóm này mô hình hay nhận diện nhầm, nên cần xác minh thêm.*"
            if attack_type in LOW_RELIABILITY_CLASSES else ""
        )
        with st.expander(info["label"]):
            st.markdown(f"**Hoạt động như thế nào:** {info['what']}")
            st.markdown(f"**Mức độ nguy hiểm:** {info['danger']}{reliability_note}")

mode = st.radio(
    "Chọn cách bắt đầu",
    ["Xem bản dùng thử", "Tôi có tệp CSV"],
    horizontal=True,
)

if mode == "Xem bản dùng thử":
    st.subheader("Xem ví dụ minh họa")
    input_df = make_demo_flows(required_features)
    st.caption(
        "Bạn không cần tải tệp lên. Hai ví dụ này được tạo để minh họa thao tác; "
        "chúng không phải dữ liệu mạng thật."
    )
    st.dataframe(
        pd.DataFrame({
            "Ví dụ": ["Flow minh họa A", "Flow minh họa B"],
            "Mô tả": [
                "Một lượt trao đổi dữ liệu qua TCP.",
                "Một lượt UDP ngắn, được dựng để minh họa dữ liệu đầu vào.",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Xem các giá trị kỹ thuật trong ví dụ"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Tải tệp CSV của ví dụ",
        data=input_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ids_demo_flows.csv",
        mime="text/csv",
    )
    source_key = "demo-v1"
else:
    st.subheader("Phân tích tệp dữ liệu của bạn")
    st.write(
        "Chọn tệp CSV chứa các lượt kết nối mạng đã chuyển thành bảng. Tệp cần "
        "đủ 29 cột đặc trưng của mô hình; tệp 34 cột gốc của UNSW-NB15 cũng "
        "được chấp nhận."
    )
    uploaded_file = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if uploaded_file is None:
        st.info(
            "Chưa có tệp phù hợp? Chọn **Xem bản dùng thử** để làm quen với ứng dụng."
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
if mode != "Xem bản dùng thử":
    with st.expander("Xem trước dữ liệu", expanded=False):
        st.dataframe(input_df.head(20), use_container_width=True)

button_label = "Phân tích ví dụ" if mode == "Xem bản dùng thử" else "Phân tích tệp"
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
        st.error("Không thể chạy dự đoán cho dữ liệu này.")
        st.exception(exc)

results = st.session_state.get("prediction_result")
if results is not None and st.session_state.get("source_key") == source_key:
    st.subheader("Kết quả")
    counts = results["prediction"].value_counts()
    normal_count = int(counts.get("Normal", 0))
    attack_count = int(counts.get("Attack", 0))
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượt kết nối đã xem", f"{len(results):,}")
    col2.metric("Được đánh dấu cần kiểm tra", f"{attack_count:,}")
    col3.metric("Chưa bị đánh dấu", f"{normal_count:,}")

    st.warning(
        "“Cần kiểm tra” nghĩa là mô hình thấy điểm giống dữ liệu tấn công đã "
        "học. Đây chưa phải bằng chứng có tấn công. “Chưa bị đánh dấu” cũng "
        "không đảm bảo kết nối an toàn."
    )
    display_results = results[
        ["prediction", "attack_type", "attack_probability"]
    ].copy()
    display_results["prediction"] = display_results["prediction"].map(
        {"Attack": "Cần kiểm tra", "Normal": "Chưa bị đánh dấu"}
    )
    display_results["attack_type"] = display_results["attack_type"].map(
        lambda value: ATTACK_TYPE_INFO.get(value, {}).get("label", value)
    )
    risk_probabilities = results["attack_probability"]
    display_results["attack_probability"] = risk_probabilities.map(
        lambda value: "—" if pd.isna(value) else risk_badge_text(value)
    )
    display_results = display_results.rename(columns={
        "prediction": "Nhận định của mô hình",
        "attack_type": "Nhóm được dự đoán",
        "attack_probability": "Điểm mô hình & mức rủi ro",
    })
    styled_results = style_risk_column(
        display_results.style, "Điểm mô hình & mức rủi ro", risk_probabilities
    )
    st.dataframe(styled_results, use_container_width=True, hide_index=True)
    st.caption(
        "Điểm mô hình chưa được hiệu chuẩn; không nên hiểu đây là phần trăm "
        "chắc chắn đúng. Nhóm tấn công là tên nhóm trong bộ dữ liệu nghiên cứu."
    )
    with st.expander("Xem toàn bộ dữ liệu và chi tiết kỹ thuật"):
        st.dataframe(results, use_container_width=True, hide_index=True)

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Tải kết quả đầy đủ dưới dạng CSV",
        data=csv_bytes,
        file_name="ids_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Kết quả này là gợi ý từ mô hình nghiên cứu, có thể sai hoặc bỏ sót. "
        "Hãy nhờ người có chuyên môn kiểm tra trước khi đưa ra quyết định. "
        "Các nhóm tấn công hiếm có thể khó nhận diện hơn."
    )

    # ======================================================================
    # Trợ lý AI: giải thích và tư vấn cho một dòng kết quả cụ thể
    # ======================================================================
    st.divider()

    chat_box = st.container(border=True)
    with chat_box:
        st.subheader("💬 Hỏi trợ lý AI về một kết nối cụ thể")

        def _row_label(pos: int) -> str:
            row = results.iloc[pos]
            tag = "Cần kiểm tra" if row["prediction"] == "Attack" else "Chưa bị đánh dấu"
            extra = f" · {row['attack_type']}" if pd.notna(row.get("attack_type")) else ""
            return f"Dòng {pos + 1} — {tag}{extra}"

        # Mặc định chọn sẵn dòng "Cần kiểm tra" đầu tiên (nếu có) để người dùng
        # không phải tự tìm trong bảng lớn.
        attack_rows = results.index[results["prediction"] == "Attack"].tolist()
        default_pos = results.index.get_loc(attack_rows[0]) if attack_rows else 0

        selected_pos = st.selectbox(
            "Chọn dòng muốn thảo luận",
            options=list(range(len(results))),
            index=default_pos,
            format_func=_row_label,
            key="chat_row_selector",
        )

        selected_row = results.iloc[selected_pos]
        st.markdown(risk_badge_html(selected_row["attack_probability"]), unsafe_allow_html=True)
        selected_attack_type = selected_row.get("attack_type")
        if pd.notna(selected_attack_type):
            info = ATTACK_TYPE_INFO.get(selected_attack_type)
            if info:
                st.caption(f"**{info['label']}** — {info['what']}")
            else:
                st.caption(f"Nhóm được dự đoán: {selected_attack_type}")

        if st.session_state.get("chat_selected_pos") != selected_pos or \
                st.session_state.get("chat_source_key") != source_key:
            st.session_state["chat_selected_pos"] = selected_pos
            st.session_state["chat_source_key"] = source_key
            st.session_state["chat_history"] = []

        for turn in st.session_state.get("chat_history", []):
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.markdown(turn["text"])

    user_msg = st.chat_input("Ví dụ: Vì sao dòng này bị đánh dấu? Nên xử lý thế nào?")
    if user_msg:
        if not gemini_api_key:
            st.error(
                "Trợ lý AI hiện chưa khả dụng vì app chưa được cấu hình "
                "Gemini API key. Vui lòng liên hệ người quản trị/triển khai "
                "app này để bật tính năng, hoặc mở mục “Dành cho người triển "
                "khai / nâng cao” ở thanh bên trái nếu đây là app bạn tự host."
            )
        else:
            with st.chat_message("user"):
                st.markdown(user_msg)
            context = build_chat_context(results.iloc[selected_pos], required_features)
            with st.chat_message("assistant"):
                with st.spinner("Đang phân tích..."):
                    try:
                        reply = call_gemini(
                            gemini_api_key, context, st.session_state["chat_history"], user_msg
                        )
                    except Exception as exc:
                        reply = f"Lỗi khi gọi Gemini API: {exc}"
                st.markdown(reply)
            st.session_state["chat_history"].append({"role": "user", "text": user_msg})
            st.session_state["chat_history"].append({"role": "model", "text": reply})

    st.caption(
        "Trợ lý AI chỉ diễn giải dựa trên 29 đặc trưng và kết quả mô hình của dòng "
        "đang chọn — không tự tra cứu thêm dữ liệu nào khác."
    )