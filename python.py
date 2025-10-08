import streamlit as st
import requests
import json
import io
import math
import numpy as np
import pandas as pd
from docx import Document
import time
from typing import Dict, Any, List

# --- Cấu hình Firebase & LLM API (MANDATORY BOILERPLATE) ---
# Sử dụng biến môi trường Canvas
try:
    API_KEY = "" # Sẽ được cung cấp tự động khi chạy trong môi trường Canvas
    APP_ID = __app_id
    FIREBASE_CONFIG = json.loads(__firebase_config)
    INITIAL_AUTH_TOKEN = __initial_auth_token
except NameError:
    API_KEY = ""
    APP_ID = "default-app-id"
    FIREBASE_CONFIG = {}
    INITIAL_AUTH_TOKEN = ""

GEMINI_MODEL_FLASH = "gemini-2.5-flash-preview-05-20"
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# --- HELPER FUNCTIONS ---

def format_currency(amount: float) -> str:
    """Định dạng tiền tệ sang tỷ VND"""
    if math.isnan(amount):
        return "N/A"
        
    abs_amount = abs(amount)
    
    if abs_amount >= 1e9:
        return f"{amount / 1e9:,.2f} tỷ VND"
    elif abs_amount >= 1e6:
        return f"{amount / 1e6:,.2f} triệu VND"
    return f"{amount:,.0f} VND"

def read_docx(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Đọc toàn bộ nội dung văn bản từ file Word đã tải lên."""
    try:
        # Sử dụng io.BytesIO để đọc file trong bộ nhớ
        document = Document(io.BytesIO(uploaded_file.getvalue()))
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        st.error(f"Lỗi khi đọc file Word: {e}")
        return ""

def clean_and_convert_to_float(value: str) -> float:
    """Làm sạch chuỗi và chuyển thành float. Xử lý các đơn vị 'tỷ', 'triệu', '%'."""
    if not value:
        return 0.0
    
    # Chuẩn hóa đơn vị và hệ số nhân
    multiplier = 1.0
    if 'tỷ' in value.lower() or 'ty' in value.lower() or 'b' in value.lower():
        multiplier = 1e9
    elif 'triệu' in value.lower() or 'tr' in value.lower() or 'm' in value.lower():
        multiplier = 1e6
        
    is_percentage = '%' in value

    # Loại bỏ các ký tự đơn vị và ký tự không cần thiết
    cleaned = value.lower().replace('%', '').replace('vnd', '').replace('tỷ', '').replace('ty', '').replace('tr', '').replace('m', '').replace('b', '').strip()
    
    # Xử lý định dạng thập phân (xóa tất cả dấu chấm/phẩy trừ dấu cuối cùng làm thập phân)
    if cleaned.count('.') > 1 and cleaned.count(',') == 0: # Ví dụ: 10.000.000
        cleaned = cleaned.replace('.', '')
    elif cleaned.count(',') > 1 and cleaned.count('.') == 0: # Ví dụ: 10,000,000
        cleaned = cleaned.replace(',', '')
    
    cleaned = cleaned.replace(',', '.') # Chuẩn hóa dấu phẩy thành dấu chấm
    
    try:
        result = float(cleaned) * multiplier
        
        # Nếu là tỷ lệ (WACC, Thuế), chia cho 100 nếu nó được trích xuất dưới dạng số nguyên lớn (ví dụ: 20 thay vì 0.2)
        if is_percentage and result > 1:
            return result / 100.0
            
        return result
    except ValueError:
        return 0.0

def call_gemini_api(prompt: str, system_instruction: str, is_json: bool, schema: Dict[str, Any] = None) -> Any:
    """Gọi API Gemini với chính sách exponential backoff."""
    url = f"{API_BASE_URL}/{GEMINI_MODEL_FLASH}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "config": {
            "temperature": 0.1,
            # Tăng cường khả năng trích xuất chính xác hơn
            "systemInstruction": system_instruction 
        }
    }

    if is_json and schema:
        payload["config"]["responseMimeType"] = "application/json"
        payload["config"]["responseSchema"] = schema
        
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() 
            result = response.json()
            
            if result and result.get('candidates') and result['candidates'][0].get('content'):
                text = result['candidates'][0]['content']['parts'][0]['text']
                if is_json:
                    # Rất quan trọng: Xử lý lỗi JSON Decode nếu AI trả về chuỗi text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        st.error(f"Lỗi giải mã JSON. AI đã trả về văn bản không phải JSON: {text[:100]}...")
                        return None
                return text
            else:
                st.error(f"Lỗi: AI trả về không có nội dung. Phản hồi: {result}")
                return None

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error(f"Lỗi gọi API Gemini sau {MAX_RETRIES} lần thử: {e}")
                return None
        except Exception as e:
            st.error(f"Lỗi không xác định khi gọi API: {e}")
            return None
    return None

def extract_financial_data(doc_text: str) -> Dict[str, float]:
    """Sử dụng AI để trích xuất dữ liệu tài chính từ văn bản."""
    
    extraction_schema = {
        "type": "OBJECT",
        "properties": {
            "Vốn_đầu_tư": {"type": "STRING", "description": "Giá trị vốn đầu tư ban đầu, bao gồm cả đơn vị (ví dụ: 30 tỷ VND)"},
            "Vòng_đời_dự_án": {"type": "STRING", "description": "Số năm vòng đời dự án (ví dụ: 10 năm)"},
            "Doanh_thu": {"type": "STRING", "description": "Doanh thu hàng năm (ví dụ: 3.5 tỷ)"},
            "Chi_phí": {"type": "STRING", "description": "Chi phí hoạt động hàng năm (ví dụ: 2 tỷ)"},
            "WACC": {"type": "STRING", "description": "Tỷ lệ WACC của doanh nghiệp (ví dụ: 13%)"},
            "Thuế": {"type": "STRING", "description": "Thuế suất TNDN (ví dụ: 20%)"}
        }
    }
    
    system_prompt = (
        "Bạn là một chuyên gia phân tích tài chính. Hãy trích xuất các thông số tài chính chính xác "
        "từ văn bản dự án kinh doanh được cung cấp, bao gồm cả đơn vị (nếu có). "
        "Đảm bảo kết quả trả về là một đối tượng JSON tuân thủ schema đã cho. "
        "Giả định dự án có dòng tiền đều hàng năm. Chỉ trả về JSON."
    )
    
    prompt = f"Trích xuất các thông số tài chính từ văn bản sau:\n\n---\n{doc_text}\n---"
    
    raw_data = None
    with st.spinner("1. AI đang trích xuất dữ liệu tài chính..."):
        raw_data = call_gemini_api(prompt, system_prompt, is_json=True, schema=extraction_schema)

    if raw_data and isinstance(raw_data, dict):
        # Chuyển đổi các giá trị thô đã trích xuất sang định dạng số
        processed_data = {
            'I0': clean_and_convert_to_float(raw_data.get('Vốn_đầu_tư', '0')),
            'N': int(clean_and_convert_to_float(raw_data.get('Vòng_đời_dự_án', '0'))),
            'R': clean_and_convert_to_float(raw_data.get('Doanh_thu', '0')),
            'C': clean_and_convert_to_float(raw_data.get('Chi_phí', '0')),
            'WACC': clean_and_convert_to_float(raw_data.get('WACC', '0')), 
            'Tax': clean_and_convert_to_float(raw_data.get('Thuế', '0'))
        }
        
        # Đảm bảo WACC và Tax ở dạng thập phân (ví dụ: 0.13 và 0.2)
        if processed_data['WACC'] > 1.0: processed_data['WACC'] /= 100.0
        if processed_data['Tax'] > 1.0: processed_data['Tax'] /= 100.0

        return processed_data
    return {}

@st.cache_data(show_spinner=False)
def calculate_financial_metrics(data: Dict[str, float]) -> Dict[str, Any]:
    """Tính toán các chỉ số NPV, IRR, PP, DPP."""
    I0 = data.get('I0', 0)
    N = int(data.get('N', 0))
    R = data.get('R', 0)
    C = data.get('C', 0)
    WACC = data.get('WACC', 0.1) 
    Tax = data.get('Tax', 0.2)
    
    # Kiểm tra dữ liệu đầu vào cơ bản
    if N <= 0 or I0 <= 0:
        return {"NPV": np.nan, "IRR": np.nan, "PP": np.nan, "DPP": np.nan, "CF_Table": []}
    
    # 1. Tính Dòng tiền thuần hàng năm (NCF)
    PBT = R - C
    PAT = PBT * (1 - Tax)
    NCF = PAT # Dòng tiền thuần hàng năm (Giả định NCF đều hàng năm và không có Khấu hao)
    data['NCF'] = NCF # Cập nhật NCF vào dict gốc

    # Kiểm tra NCF để tránh chia cho 0
    if NCF <= 0 and I0 > 0:
        return {"NPV": -I0, "IRR": np.nan, "PP": np.nan, "DPP": np.nan, "CF_Table": []}

    # Bảng Dòng tiền
    cf_data = []
    cf_data.append({"Năm": 0, "NCF": -I0, "PV_NCF": -I0, "CF_Cum": -I0, "PV_CF_Cum": -I0})

    CF_Cum = -I0
    PV_CF_Cum = -I0
    
    # Tính toán cho các năm 1 đến N
    for t in range(1, N + 1):
        # Giá trị hiện tại của NCF
        PV_NCF = NCF / (1 + WACC) ** t
        
        # Dòng tiền tích lũy và dòng tiền tích lũy chiết khấu
        CF_Cum += NCF
        PV_CF_Cum += PV_NCF
        
        cf_data.append({
            "Năm": t, 
            "NCF": NCF, 
            "PV_NCF": PV_NCF, 
            "CF_Cum": CF_Cum, 
            "PV_CF_Cum": PV_CF_Cum
        })
        
    cf_df = pd.DataFrame(cf_data)
    
    # 2. NPV (Giá trị hiện tại thuần)
    NPV = cf_df['PV_NCF'].sum()
    
    # 3. IRR (Tỷ suất sinh lời nội bộ)
    cash_flows = np.array([-I0] + [NCF] * N)
    try:
        IRR = np.irr(cash_flows)
    except ValueError:
        IRR = np.nan
        
    # 4. PP (Thời gian hoàn vốn) và 5. DPP (Thời gian hoàn vốn có chiết khấu)
    PP = np.nan
    DPP = np.nan
    
    # Tính PP từ CF_Cum
    pp_row_idx = cf_df[cf_df['CF_Cum'] >= 0].index
    if len(pp_row_idx) > 0 and pp_row_idx[0] > 0:
        k = pp_row_idx[0] 
        CF_k_minus_1 = cf_df.loc[k-1, 'CF_Cum']
        
        # Chỉ tính nếu NCF > 0 để tránh lỗi chia cho 0
        if NCF > 0:
             PP = (k - 1) + abs(CF_k_minus_1) / NCF
    
    # Tính DPP từ PV_CF_Cum
    dpp_row_idx = cf_df[cf_df['PV_CF_Cum'] >= 0].index
    if len(dpp_row_idx) > 0 and dpp_row_idx[0] > 0:
        k_dpp = dpp_row_idx[0] 
        PV_CF_k_minus_1 = cf_df.loc[k_dpp-1, 'PV_CF_Cum'] 
        PV_NCF_k = cf_df.loc[k_dpp, 'PV_NCF'] 
        
        # Chỉ tính nếu PV_NCF_k > 0
        if PV_NCF_k > 0:
            DPP = (k_dpp - 1) + abs(PV_CF_k_minus_1) / PV_NCF_k
        
    
    return {
        "NPV": NPV, 
        "IRR": IRR, 
        "PP": PP, 
        "DPP": DPP,
        "CF_Table": cf_df
    }

def get_ai_analysis_report(metrics: Dict[str, Any], initial_data: Dict[str, float]) -> str:
    """Yêu cầu AI phân tích chuyên sâu các chỉ số tài chính."""
    
    NPV = metrics.get('NPV', np.nan)
    IRR = metrics.get('IRR', np.nan)
    PP = metrics.get('PP', np.nan)
    DPP = metrics.get('DPP', np.nan)
    WACC = initial_data.get('WACC', 0.1) * 100
    
    # Chuẩn bị dữ liệu cho AI
    data_for_ai = (
        f"NPV (Giá trị hiện tại thuần): {format_currency(NPV)}\n"
        f"IRR (Tỷ suất sinh lời nội bộ): {IRR*100:.2f}% (WACC của doanh nghiệp là {WACC:.2f}%)\n"
        f"PP (Thời gian hoàn vốn): {PP:.2f} năm\n"
        f"DPP (Thời gian hoàn vốn có chiết khấu): {DPP:.2f} năm\n"
        f"Vốn đầu tư ban đầu: {format_currency(initial_data.get('I0'))}\n"
        f"Vòng đời dự án: {initial_data.get('N')} năm\n"
        f"Dòng tiền thuần (NCF) hàng năm: {format_currency(initial_data.get('NCF'))}\n"
    )
    
    # Thêm điều kiện để AI đưa ra quyết định chấp nhận/từ chối rõ ràng hơn
    if not math.isnan(NPV) and NPV > 0 and not math.isnan(IRR) and IRR > WACC/100:
        feasibility_status = "Dự án có khả năng chấp nhận (NPV > 0 và IRR > WACC)."
    elif not math.isnan(NPV) and NPV < 0:
        feasibility_status = "Dự án có khả năng bị từ chối (NPV < 0)."
    else:
        feasibility_status = "Dữ liệu không đủ hoặc dự án biên độ thấp (cần phân tích sâu)."

    system_prompt = (
        "Bạn là một chuyên gia thẩm định và phân tích dự án đầu tư. "
        "Dựa trên các chỉ số hiệu quả tài chính được cung cấp, hãy đưa ra báo cáo phân tích chuyên sâu (ít nhất 3 đoạn):"
        "1. Đánh giá tính khả thi và chấp nhận/từ chối dự án (dựa vào NPV và so sánh IRR với WACC)."
        "2. Nhận xét về rủi ro và tính thanh khoản của dự án (dựa vào PP và DPP)."
        "3. Đề xuất kiến nghị để cải thiện hiệu quả tài chính (ví dụ: giảm vốn, tăng doanh thu, kéo dài vòng đời). "
        "Viết bằng tiếng Việt và sử dụng ngôn ngữ chuyên nghiệp."
    )
    
    prompt = (
        f"Phân tích báo cáo hiệu quả tài chính dự án với các chỉ số sau:\n\n---\n{data_for_ai}\n---"
        f"Hãy bắt đầu báo cáo bằng việc khẳng định trạng thái dự án: {feasibility_status}"
    )
    
    with st.spinner("4. AI đang tạo báo cáo phân tích chuyên sâu..."):
        analysis_report = call_gemini_api(prompt, system_prompt, is_json=False)
        return analysis_report


# --- STREAMLIT APP ---
st.set_page_config(
    page_title="App Phân Tích Dự Án Đầu Tư (Word to Metrics)",
    layout="wide"
)

st.title("💰 Phân Tích Hiệu Quả Dự Án Đầu Tư Tự Động")
st.markdown("Sử dụng AI để trích xuất dữ liệu từ file Word và tính toán các chỉ số NPV, IRR, PP, DPP.")

# --- Session State Initialization ---
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = ""

# --- 1. Tải File và Lọc Dữ liệu ---
uploaded_file = st.file_uploader(
    "Tải lên file Word (.docx) chứa phương án kinh doanh",
    type=['docx']
)

if uploaded_file is not None:
    st.session_state.doc_text = read_docx(uploaded_file)
    
    if st.button("1. Lọc Dữ liệu Tài chính (AI)"):
        if st.session_state.doc_text:
            st.session_state.project_data = extract_financial_data(st.session_state.doc_text)
            st.session_state.metrics = {} # Reset metrics khi lọc lại

    if st.session_state.project_data and st.session_state.project_data.get('I0', 0) > 0:
        
        data = st.session_state.project_data
        
        st.subheader("✅ Dữ liệu Dự án đã Lọc")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        # Tính toán NCF để hiển thị
        PBT = data.get('R', 0) - data.get('C', 0)
        PAT = PBT * (1 - data.get('Tax', 0.2))
        NCF = PAT
        st.session_state.project_data['NCF'] = NCF # Lưu NCF vào state

        col1.metric("Vốn Đầu tư ($I_0$)", format_currency(data.get('I0')))
        col2.metric("Vòng đời Dự án ($N$)", f"{data.get('N')} năm")
        col3.metric("WACC", f"{data.get('WACC', 0.0) * 100:.2f}%")
        col4.metric("Doanh thu ($R$)", format_currency(data.get('R')))
        col5.metric("Chi phí ($C$)", format_currency(data.get('C')))
        col6.metric("Thuế suất", f"{data.get('Tax', 0.0) * 100:.0f}%")
        
        st.metric("Dòng tiền thuần (NCF) hàng năm", format_currency(NCF))
        st.markdown("---")
        
        # --- 2. Xây dựng Bảng Dòng tiền và 3. Tính Chỉ số ---
        st.subheader("2 & 3. Bảng Dòng tiền và Chỉ số Đánh giá Hiệu quả")
        
        if st.button("Tính toán Chỉ số Tài chính (NPV, IRR, PP, DPP)"):
            st.session_state.metrics = calculate_financial_metrics(st.session_state.project_data)
            
        # Kiểm tra nếu tính toán đã được thực hiện và thành công
        if st.session_state.metrics and st.session_state.metrics.get("CF_Table") is not None and not st.session_state.metrics.get("CF_Table").empty:
            metrics = st.session_state.metrics
            
            # Hiển thị Chỉ số Đánh giá
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            col_m1.metric("NPV (Giá trị hiện tại thuần)", format_currency(metrics['NPV']))
            col_m2.metric("IRR (Tỷ suất sinh lời nội bộ)", f"{metrics['IRR'] * 100:.2f}%" if not math.isnan(metrics['IRR']) else "N/A")
            col_m3.metric("PP (Hoàn vốn)", f"{metrics['PP']:.2f} năm" if not math.isnan(metrics['PP']) else "N/A")
            col_m4.metric("DPP (Hoàn vốn chiết khấu)", f"{metrics['DPP']:.2f} năm" if not math.isnan(metrics['DPP']) else "N/A")

            st.markdown("#### Bảng Dòng tiền Dự án")
            cf_df = metrics['CF_Table'].copy()
            
            # Định dạng DataFrame để hiển thị đẹp hơn
            cf_df['Năm'] = cf_df['Năm'].astype(int)
            st.dataframe(
                cf_df.style.format({
                    'NCF': lambda x: format_currency(x),
                    'PV_NCF': lambda x: format_currency(x),
                    'CF_Cum': lambda x: format_currency(x),
                    'PV_CF_Cum': lambda x: format_currency(x),
                }),
                use_container_width=True,
                hide_index=True
            )
            st.markdown("---")
            
            # --- 4. Yêu cầu AI Phân tích ---
            st.subheader("4. Báo cáo Phân tích Chuyên sâu (AI)")
            
            if st.button("Yêu cầu AI Phân tích Hiệu quả Dự án"):
                report = get_ai_analysis_report(metrics, st.session_state.project_data)
                st.markdown("#### Kết quả Phân tích từ Gemini AI")
                st.info(report)
        
        else:
             st.info("Nhấn nút 'Tính toán Chỉ số Tài chính' để tạo bảng dòng tiền và các chỉ số.")

    else:
        if uploaded_file and st.session_state.doc_text and not st.session_state.project_data:
             st.warning("Vui lòng nhấn nút 'Lọc Dữ liệu Tài chính (AI)' để trích xuất thông số.")
        elif uploaded_file and st.session_state.project_data and st.session_state.project_data.get('I0', 0) <= 0:
             st.error("Không thể tính toán: Vốn Đầu tư ($I_0$) hoặc Vòng đời Dự án ($N$) được trích xuất bằng 0 hoặc không phải số dương.")

else:
    st.info("Vui lòng tải lên file Word để bắt đầu phân tích.")
