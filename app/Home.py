import streamlit as st

st.set_page_config(
    page_title="Đồ án CSDL ĐPT - Nhóm 8",
    page_icon="📚",
    layout="wide"
)

st.title("HỆ THỐNG TÌM KIẾM VĂN BẢN & PHÂN TÍCH HÌNH HỌC TDV SALTON")
st.subheader("Đồ án môn học: Cơ sở dữ liệu Đa phương tiện")

st.markdown("""
### Thành viên thực hiện:
* **M1:** Xử lý Ngôn ngữ Tự nhiên & Lọc Hư từ Ngữ pháp
* **M2:** Động cơ TDV, Lập Chỉ mục & Truy xuất Thông tin
* **M3:** Trưởng nhóm & Kiến trúc sư ứng dụng

---
### Các chức năng chính:
1. **1_Tokenizer:** Mô phỏng thuật toán tách từ và lọc hư từ Tầng 1.
2. **2_Analytics:** Phân tích hình học không gian vector và đường cong TDV Salton.
3. **3_Search:** Cổng tìm kiếm trực tuyến & Bảng giải thích điểm số (Explainability Panel).
""")