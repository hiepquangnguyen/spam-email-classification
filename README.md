# 📧 Phân Loại Email Rác (Spam Email Classification)

Dự án nghiên cứu, thực nghiệm và so sánh các phương pháp Học máy (Machine Learning) và Học sâu (Deep Learning / Transformers) trong bài toán phân loại email rác (**Spam**) và email thông thường (**Ham**).

---

## 📂 Cấu trúc dự án

```text
spam_email_classification/
├── data/                               # Dữ liệu huấn luyện & embeddings
│   ├── spam_ham_dataset.csv            # Bộ dữ liệu gốc (5,171 emails)
│   ├── email_spam.pkl                  # Dữ liệu tiền xử lý
│   └── embedding_spam_ham_datset_cleaned.pkl # Vector embeddings (SentenceTransformer)
├── checkpoints/                        # Trọng số và mô hình đã huấn luyện
│   ├── distilbert_spam_20250625_163743/ # Trọng số & Tokenizer DistilBERT (~268MB)
│   ├── ff_spam_classifier_model_spam_ham_dataset_cleaned.pth # Trọng số PyTorch FFNN
│   ├── logistic_regression_best.pkl    # Pipeline Scikit-Learn
│   └── README.md                       # Link tải checkpoints đầy đủ (Google Drive)
├── notebooks/                          # Các Jupyter Notebook thực nghiệm
│   ├── 01_logistic_spam_classification.ipynb   # Mô hình 1: Logistic Regression (Baseline)
│   ├── 02_ffnn_spam_classification.ipynb       # Mô hình 2: Feedforward Neural Network (PyTorch)
│   └── 03_distilbert_spam_classification.ipynb # Mô hình 3: Fine-tuning DistilBERT (Transformer)
├── docs/                               # Tài liệu & Báo cáo
│   └── report_btl.pdf                  # Báo cáo chi tiết đề tài
├── .gitignore                          # Cấu hình bỏ qua file lớn/tạm khi commit
├── requirements.txt                    # Danh sách thư viện cần thiết
└── README.md                           # Giới thiệu dự án
```

---

## 📊 So sánh hiệu năng các mô hình

| Thứ tự | Mô hình | Kỹ thuật trích xuất đặc trưng | Accuracy | F1-Score (Spam) | Ưu / Nhược điểm |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **01** | **Logistic Regression** | BoW / CountVectorizer | ~97.96% | 0.97 | Tốc độ train và inference cực nhanh, tài nguyên thấp, baseline mạnh. |
| **02** | **Feedforward NN (FFNN)** | `all-MiniLM-L6-v2` Embeddings | **98.93%** | 0.98 | Nắm bắt tốt ngữ nghĩa văn bản qua Dense Embeddings, mô hình nhẹ. |
| **03** | **DistilBERT** | Pre-trained Contextual Tokenizer | **98.81%** (CV) | 0.98 | Hiểu sâu ngữ cảnh đa chiều, tổng quát hóa tốt nhất trên văn bản phức tạp. |

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install -r requirements.txt
```

### 2. Thực thi Notebooks
Khởi chạy Jupyter Notebook hoặc mở trực tiếp trên VS Code:
```bash
jupyter notebook notebooks/
```
Các notebook đã được tích hợp cơ chế tự động nhận diện thư mục gốc (`project root`), bạn có thể chạy tuần tự từng cell mà không lo lỗi đường dẫn dữ liệu.
