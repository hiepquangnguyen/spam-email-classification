# Model Checkpoints

Thư mục này lưu trữ các trọng số (weights), pipeline và cấu hình mô hình đã được huấn luyện.

## 🔗 Liên kết tải trọng số đầy đủ (Google Drive)
Do các mô hình Deep Learning và Transformer có kích thước lớn (vượt quá giới hạn lưu trữ khuyến nghị trên Git), các checkpoint đầy đủ được lưu trên Google Drive:
- **Google Drive**: [Truy cập thư mục Checkpoints](https://drive.google.com/drive/folders/1RxWy27Kd-Ou9iE_6dY21ueYpxCfHyTXe?usp=sharing)

## 📁 Danh sách checkpoint trong thư mục:
- `distilbert_spam_20250625_163743/`:
  - `distilbert_spam_classifier.pth`: Trọng số fine-tuned DistilBERT (~268 MB)
  - `tokenizer/`: Thư mục chứa cấu hình và từ điển tokenizer của DistilBERT
  - `training_config.json`: Cấu hình siêu tham số và thông tin huấn luyện
- `ff_spam_classifier_model_spam_ham_dataset_cleaned.pth`: Trọng số mô hình Feedforward Neural Network (PyTorch / Skorch)
- `logistic_regression_best.pkl`: Mô hình Logistic Regression hoàn chỉnh kèm bộ trích xuất đặc trưng
