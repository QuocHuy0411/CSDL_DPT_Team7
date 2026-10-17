import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phân tích TDV", layout="wide")
st.title("2. Phân tích giá trị phân biệt từ TDV")
directory = Path(__file__).resolve().parents[2] / "data/indexes/C4"
if not (directory / "tdv_scores.json").is_file():
    st.info("Chưa có bảng TDV. Chạy: python -m scripts.build_indexes")
    st.stop()

scores = json.loads((directory / "tdv_scores.json").read_text(encoding="utf-8"))
metadata = json.loads((directory / "build_metadata.json").read_text(encoding="utf-8"))
frequencies = json.loads((directory / "tdv_df.json").read_text(encoding="utf-8"))
stopwords = set(json.loads((directory / "tdv_stopwords.json").read_text(encoding="utf-8")))
if metadata["preprocessing_status"] != "official":
    st.warning("Số liệu phát triển; đang chờ Maximum Matching và tầng 1 chính thức từ M1.")
st.caption("TDV tính một lần trên toàn bộ 2.500 bài thuộc 10 lĩnh vực. "
           "D là trung bình Cosine giữa các cặp tài liệu, không gồm self; TDV = D sau xóa − D gốc.")
frame = pd.DataFrame([{"term": term, "df": frequencies[term], "tdv": score,
                       "filtered": term in stopwords} for term, score in scores.items()])
st.metric("Term bị lọc trong C4", len(stopwords))
st.scatter_chart(frame, x="df", y="tdv")
st.caption("Đây là phân bố đo được; không giả định đường cong phải có hình chữ U.")
st.caption("Term có TDV âm không đồng nghĩa không liên quan về nội dung. "
           "Chưa có MAP/NDCG chính thức cho tới khi qrels được người đọc xác nhận.")
needle = st.text_input("Tra term", value="")
selected = frame[frame["term"].str.contains(needle, regex=False)] if needle else frame
st.dataframe(selected.sort_values("tdv").head(1000), width="stretch")
