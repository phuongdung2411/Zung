# investment_app.py 

import streamlit as st
import pandas as pd
import numpy as np
# Cần import numpy_financial vì các hàm npv/irr đã bị loại khỏi numpy
import numpy_financial as npf 
from google import genai
from google.genai.errors import APIError
from docx import Document
import io
import re

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Phương án Kinh doanh 📈")

# --- Hàm đọc file Word ---
def read_docx_file(uploaded_file):
    """Đọc nội dung văn bản từ file Word."""
    try:
        doc = Document(io.BytesIO(uploaded_file.read()))
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        return f"Lỗi đọc file Word: {e}"

# --- Hàm gọi API Gemini để trích xuất thông tin (Yêu cầu 1) ---
@st.cache_data
def extract_financial_data(doc_text, api_key):
    """Sử dụng Gemini để trích xuất các thông số tài chính từ văn bản."""
    
    if not api_key:
        raise ValueError("Khóa API không được cung cấp.")
        
    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'
    
    # Prompt yêu cầu JSON nguyên mẫu để dễ dàng parse
    prompt = f"""
    Bạn là một chuyên gia tài chính và phân tích dự án. Nhiệm vụ của bạn là trích xuất các thông số sau từ nội dung văn bản kinh doanh bên dưới. 
    Các thông số này phải là GIÁ TRỊ SỐ, không có đơn vị (ví dụ: 1000000). 
    
    Vốn đầu tư (Initial Investment - C0): Giá trị tuyệt đối của vốn ban đầu cần bỏ ra.
    Dòng đời dự án (Project Life - N): Số năm hoạt động của dự án.
    WACC (Cost of Capital - k): Tỷ lệ chiết khấu (dạng thập phân, ví dụ: 0.10 cho 10%).
    Thuế suất (Tax Rate - t): Tỷ lệ thuế thu nhập doanh nghiệp (dạng thập phân, ví dụ: 0.20 cho 20%).
    
    Doanh thu hàng năm (Annual Revenue - R): Nếu không có thông tin chi tiết từng năm, hãy ước tính một con số đại diện cho doanh thu hàng năm.
    Chi phí hoạt động hàng năm (Annual Operating Cost - C): Nếu không có thông tin chi tiết từng năm, hãy ước tính một con số đại diện cho chi phí hoạt động hàng năm (chưa bao gồm Khấu hao).
    
    Nếu không tìm thấy thông tin cụ thể, hãy trả về 0 cho giá trị số (trừ WACC và Thuế suất nên là 0.2 nếu không tìm thấy).

    Định dạng đầu ra **bắt buộc** là JSON nguyên mẫu (RAW JSON), không có bất kỳ giải thích hay văn bản nào khác.
    
    {{
      "Vốn đầu tư": <Giá trị số>,
      "Dòng đời dự án": <Giá trị số năm>,
      "Doanh thu hàng năm": <Giá trị số>,
      "Chi phí hoạt động hàng năm": <Giá trị số>,
      "WACC": <Giá trị số thập phân>,
"Thuế suất": <Giá trị số thập phân>
    }}

    Nội dung file Word:
    ---
    {doc_text}
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    
    # Xử lý chuỗi JSON trả về
    json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
    return pd.read_json(io.StringIO(json_str), typ='series')

# --- Hàm tính toán Chỉ số Tài chính (Yêu cầu 3) ---
def calculate_project_metrics(df_cashflow, initial_investment, wacc):
    """Tính toán NPV, IRR, PP, DPP."""
    
    cash_flows = df_cashflow['Dòng tiền thuần (CF)'].values
    
    # 1. NPV
    # Thêm vốn đầu tư ban đầu vào đầu dòng tiền
    full_cash_flows = np.insert(cash_flows, 0, -initial_investment) 
    
    # Dùng npf.npv
    npv_value = npf.npv(wacc, full_cash_flows)
    
    # 2. IRR
    try:
        # Dùng npf.irr
        irr_value = npf.irr(full_cash_flows)
    except ValueError:
        irr_value = np.nan 

    # 3. PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(full_cash_flows)
    pp_year = np.where(cumulative_cf >= 0)[0]
    if pp_year.size > 0:
        pp_year = pp_year[0]
        if pp_year == 0: 
             pp = 0 
        else:
             capital_remaining = abs(cumulative_cf[pp_year-1])
             cf_of_payback_year = cash_flows[pp_year-1]
             pp = pp_year - 1 + (capital_remaining / cf_of_payback_year) if cf_of_payback_year != 0 else pp_year 
    else:
        pp = 'Không hoàn vốn'

    # 4. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discount_factors = 1 / ((1 + wacc) ** np.arange(0, len(full_cash_flows)))
    discounted_cf = full_cash_flows * discount_factors
    cumulative_dcf = np.cumsum(discounted_cf)
    
    dpp_year = np.where(cumulative_dcf >= 0)[0]
    if dpp_year.size > 0:
        dpp_year = dpp_year[0]
        if dpp_year == 0:
             dpp = 0
        else:
             capital_remaining_d = abs(cumulative_dcf[dpp_year-1])
             dcf_of_payback_year = discounted_cf[dpp_year] 
             dpp = dpp_year - 1 + (capital_remaining_d / dcf_of_payback_year) if dcf_of_payback_year != 0 else dpp_year
    else:
        dpp = 'Không hoàn vốn'
        
    return npv_value, irr_value, pp, dpp

# --- Hàm gọi AI phân tích chỉ số (Yêu cầu 4) ---
def get_ai_evaluation(metrics_data, wacc_rate, api_key):
    """Gửi các chỉ số đánh giá dự án đến Gemini API và nhận phân tích."""
    
    if not api_key:
        return "Lỗi: Khóa API không được cung cấp."

    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'  

        prompt = f"""
Bạn là một chuyên gia phân tích dự án đầu tư có kinh nghiệm. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra nhận xét ngắn gọn, khách quan (khoảng 3-4 đoạn) về khả năng chấp nhận và rủi ro của dự án. 
        
        Các chỉ số cần phân tích:
        - NPV: {metrics_data['NPV']:.2f}
        - IRR: {metrics_data['IRR']:.2%}
        - WACC (Tỷ lệ chiết khấu): {wacc_rate:.2%}
        - PP (Thời gian hoàn vốn): {metrics_data['PP']} năm
        - DPP (Thời gian hoàn vốn có chiết khấu): {metrics_data['DPP']} năm
        
        Chú ý:
        1. Đánh giá tính khả thi (NPV > 0 và IRR > WACC).
        2. Nhận xét về tốc độ hoàn vốn (PP và DPP).
        3. Kết luận tổng thể về việc chấp nhận hay từ chối dự án.
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Giao diện và Luồng chính ---

# Lấy API Key
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
     st.error("⚠️ Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng chức năng AI.")

uploaded_file = st.file_uploader(
    "1. Tải file Word (.docx) chứa Phương án Kinh doanh:",
    type=['docx']
)

# Khởi tạo state để lưu trữ dữ liệu đã trích xuất
if 'extracted_data' not in st.session_state:
    st.session_state['extracted_data'] = None

# --- Chức năng 1: Lọc dữ liệu bằng AI ---
if uploaded_file is not None:
    doc_text = read_docx_file(uploaded_file)
    
    if st.button("Trích xuất Dữ liệu Tài chính bằng AI 🤖"):
        if api_key:
            with st.spinner('Đang đọc và trích xuất thông số tài chính bằng Gemini...'):
                try:
                    st.session_state['extracted_data'] = extract_financial_data(doc_text, api_key)
                    st.success("Trích xuất dữ liệu thành công!")
                except APIError:
                    st.error("Lỗi API: Không thể kết nối hoặc xác thực API Key.")
                except Exception as e:
                    st.error(f"Lỗi trích xuất: {e}")
        else:
            st.error("Vui lòng cung cấp Khóa API.")

# --- Hiển thị và Tính toán (Yêu cầu 2 & 3) ---
if st.session_state['extracted_data'] is not None:
    data = st.session_state['extracted_data']
    
    # ****************** Lọc các giá trị số và xử lý ngoại lệ ******************
    try:
        initial_investment = float(data.get('Vốn đầu tư', 0))
        project_life = int(data.get('Dòng đời dự án', 0))
        annual_revenue = float(data.get('Doanh thu hàng năm', 0))
annual_cost = float(data.get('Chi phí hoạt động hàng năm', 0))
        wacc = float(data.get('WACC', 0.1)) # Giả định WACC 10% nếu không trích xuất được
        tax_rate = float(data.get('Thuế suất', 0.2)) # Giả định Thuế 20% nếu không trích xuất được
        
        # Đảm bảo WACC và Thuế suất ở dạng thập phân (0 < value < 1)
        if wacc > 1: wacc /= 100
        if tax_rate > 1: tax_rate /= 100
        
    except Exception as e:
        st.error(f"Lỗi chuyển đổi dữ liệu trích xuất thành số: {e}. Vui lòng kiểm tra lại nội dung file Word.")
        initial_investment, project_life, wacc, tax_rate = 0, 0, 0.1, 0.2

    # ****************** Hiển thị Thông số ******************
    st.subheader("2. Các Thông số Dự án đã Trích xuất")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Vốn Đầu tư (C₀)", f"{initial_investment:,.0f} VNĐ")
    col2.metric("Dòng đời dự án (N)", f"{project_life:.0f} năm")
    col3.metric("WACC (k)", f"{wacc:.2%}")
    col1.metric("Doanh thu Hàng năm (R)", f"{annual_revenue:,.0f} VNĐ")
    col2.metric("Chi phí HĐ Hàng năm (C)", f"{annual_cost:,.0f} VNĐ")
    col3.metric("Thuế suất (t)", f"{tax_rate:.2%}")

    st.markdown("---")
    
    # ****************** Bảng Dòng tiền (Yêu cầu 2) ******************
    st.subheader("3. Bảng Dòng tiền (Cash Flow)")
    
    if project_life > 0 and initial_investment >= 0:
        try:
            depreciation = initial_investment / project_life 
        except ZeroDivisionError:
            depreciation = 0

        years = np.arange(1, project_life + 1)
        
        # Tính toán dòng tiền hàng năm (Giả định đơn giản: dòng tiền đều)
        EBT = annual_revenue - annual_cost - depreciation
        Tax = EBT * tax_rate if EBT > 0 else 0
        EAT = EBT - Tax
        # Dòng tiền thuần = Lợi nhuận sau thuế + Khấu hao
        CF = EAT + depreciation
        
        cashflow_data = {
            'Năm': years,
            'Doanh thu (R)': [annual_revenue] * project_life,
            'Chi phí HĐ (C)': [annual_cost] * project_life,
            'Khấu hao (D)': [depreciation] * project_life,
            'Lợi nhuận trước thuế (EBT)': [EBT] * project_life,
            'Thuế (Tax)': [Tax] * project_life,
            'Lợi nhuận sau thuế (EAT)': [EAT] * project_life,
            'Dòng tiền thuần (CF)': [CF] * project_life
        }
        
        df_cashflow = pd.DataFrame(cashflow_data)
        
        st.dataframe(
            df_cashflow.style.format({
                col: '{:,.0f}' for col in df_cashflow.columns if col not in ['Năm']
            }), 
            use_container_width=True
        )

        st.markdown("---")
        
        # ****************** Tính toán Chỉ số (Yêu cầu 3) ******************
st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
        
        if wacc > 0:
            try:
                npv, irr, pp, dpp = calculate_project_metrics(df_cashflow, initial_investment, wacc)
                
                metrics_data = {
                    'NPV': npv,
                    'IRR': irr if not np.isnan(irr) else 0, # Dùng 0 để tránh lỗi format
                    'PP': pp,
                    'DPP': dpp
                }
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("NPV (Giá trị hiện tại thuần)", f"{npv:,.0f} VNĐ", delta=("Dự án có lời" if npv > 0 else "Dự án lỗ"))
                col2.metric("IRR (Tỷ suất sinh lời nội tại)", f"{irr:.2%}" if not np.isnan(irr) else "Không tính được")
                col3.metric("PP (Thời gian hoàn vốn)", f"{pp:.2f} năm" if isinstance(pp, float) or isinstance(pp, np.float64) else pp)
                col4.metric("DPP (Hoàn vốn có chiết khấu)", f"{dpp:.2f} năm" if isinstance(dpp, float) or isinstance(dpp, np.float64) else dpp)

                # ****************** Phân tích AI (Yêu cầu 4) ******************
                st.markdown("---")
                st.subheader("5. Phân tích Hiệu quả Dự án (AI)")
                
                if st.button("Yêu cầu AI Phân tích Chỉ số 🧠"):
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_evaluation(metrics_data, wacc, api_key)
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                         st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng kiểm tra cấu hình Secrets.")

            except Exception as e:
                st.error(f"Có lỗi xảy ra khi tính toán chỉ số: {e}. Vui lòng kiểm tra các thông số đầu vào.")
        else:
            st.warning("WACC (Tỷ lệ chiết khấu) phải lớn hơn 0 để tính toán NPV và DPP.")

    else:
        st.warning("Vui lòng đảm bảo Dòng đời Dự án và Vốn Đầu tư đã được trích xuất thành công và có giá trị lớn hơn 0.")

else:
    st.info("Vui lòng tải lên file Word và nhấn nút 'Trích xuất Dữ liệu Tài chính bằng AI' để bắt đầu.")
