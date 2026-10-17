# Nguồn học thuật và quyết định triển khai

1. Tài liệu địa phương: `../Bai tap nhom - D23CQCNHT01.pdf`, trang 2, đề tài 8; yêu cầu tách từ, loại stopword, TF-IDF, Cosine và đánh giá. Các phần lý thuyết/cài đặt/thực nghiệm phải hiện diện trong báo cáo.
2. `../Chuong 3 - He thong tuy van thong tin van ban - 1.pdf`: trang 6–9 (Document/Query/Index/Relevance), 12–13 (đặc trưng), 20–24 (chỉ mục đảo), 26–28 (tokenization/stopword/normalization). PDF này có 33 trang; không coi các đoạn trích vector/TDV trong DOCX là code đã được kiểm chứng.
3. G. Salton, C. S. Yang, C. T. Yu, “A theory of term importance in automatic text analysis,” *Journal of the American Society for Information Science*, 26(1), 33–44, 1975. Nguồn nhà xuất bản: https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.4630260106 . Khái niệm discrimination value đánh giá sự thay đổi độ tách biệt của tài liệu khi gán/bỏ term. Bản nghiên cứu tại Cornell (1974): https://ecommons.cornell.edu/entities/publication/0f17b8d0-4e8d-43b7-81c7-363fe12a51b9 .
4. G. Salton, A. Wong, C. S. Yang, “A vector space model for automatic indexing,” *Communications of the ACM*, 18(11), 613–620, 1975, DOI: https://doi.org/10.1145/361219.361220 .
5. Tài liệu chính thức scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html . Dùng cấu hình `sublinear_tf=True`, `smooth_idf=True`, `norm='l2'`, vocabulary/token fixture giống nhau để đối soát M2.
6. Stoplist C2: [bản chụp, revision, số mục và MIT notice](../resources/static_stopwords_source.md).

Mật độ pairwise không gồm self, quy ước zero vector, IDF giữ nguyên trong phép đo xóa một chiều và dung sai `1e-12` là các lựa chọn triển khai được chốt tại [api_contracts.md](api_contracts.md). Không tuyên bố mọi chi tiết hiện thực này là nguyên văn bài báo 1975. Có phiên bản tham chiếu trực tiếp để kiểm chứng đường tính nhanh. Dấu TDV không tự chứng minh term vô nghĩa hay chất lượng C4 vượt các cấu hình khác.
