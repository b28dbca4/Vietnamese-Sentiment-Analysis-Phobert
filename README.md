# IMDb Sentiment Analysis with Transformer Fine-tuning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## Tổng quan

Dự án xây dựng hệ thống phân tích cảm xúc cho bộ dữ liệu IMDb 50K Movie Reviews. Quy trình gồm tiền xử lý dữ liệu, huấn luyện mô hình baseline, fine-tune các mô hình Transformer, đánh giá định lượng, trực quan hóa kết quả và triển khai ứng dụng Streamlit để dự đoán cảm xúc theo thời gian thực.

Mục tiêu chính:

- Phân loại review phim thành hai nhãn `negative` và `positive`.
- So sánh baseline truyền thống với các mô hình Transformer fine-tuned.
- Lưu đầy đủ checkpoint, metric, biểu đồ và báo cáo đánh giá.
- Cung cấp giao diện Streamlit trực quan để phân tích một review hoặc nhiều review cùng lúc.

## Dữ liệu

- Nguồn: IMDb Dataset of 50K Movie Reviews.
- Bài toán: binary sentiment classification.
- Số mẫu sau loại trùng: 49,582 review.
- Chia dữ liệu: 80% train, 10% validation, 10% test.
- Cột sử dụng trong modeling: `clean_text`, `label`, `label_id`.
- Tiền xử lý: xóa HTML, xóa URL, chuẩn hóa khoảng trắng.
- Không lowercase, stemming, lemmatization hoặc loại stopword để giữ ngữ cảnh cho tokenizer của Transformer.

Phân chia dữ liệu hiện tại:

| Split | Số dòng gồm header | Số mẫu |
| --- | ---: | ---: |
| Train | 39,666 | 39,665 |
| Validation | 4,959 | 4,958 |
| Test | 4,960 | 4,959 |

## Mô hình

Dự án so sánh bốn hướng tiếp cận:

| Model | Vai trò | Ghi chú |
| --- | --- | --- |
| TF-IDF + Logistic Regression | Baseline | Nhanh, nhẹ, dễ giải thích |
| DistilBERT | Transformer nhẹ | Ít tham số hơn, tốc độ tốt hơn BERT |
| BERT-base | Transformer chuẩn | Mốc so sánh phổ biến cho NLP |
| RoBERTa | Transformer cải tiến | Hiệu năng tốt nhất trong thí nghiệm này |

Cấu hình fine-tuning Transformer:

| Thiết lập | Giá trị |
| --- | ---: |
| `max_length` | 256 |
| `batch_size` | 16 |
| `epochs` | 3 |
| `learning_rate` | 2e-5 |
| `weight_decay` | 0.01 |
| `warmup_ratio` | 0.1 |
| `max_grad_norm` | 1.0 |
| `early_stopping_patience` | 2 |

## Kết quả

Kết quả dưới đây được lấy từ `results/metrics/comparison_summary.json`.

| Model | Params | Size MB | Train Time | Accuracy | F1 Macro | Precision Macro | Recall Macro | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 50,000 | 1.0 | 26.4s | 90.87% | 90.86% | 90.91% | 90.86% | 0.9681 |
| DistilBERT | 66,955,010 | 255.0 | 1,746.0s | 90.82% | 90.81% | 91.02% | 90.81% | 0.9711 |
| BERT-base | 109,483,778 | 418.0 | 3,196.4s | 91.79% | 91.78% | 91.95% | 91.78% | 0.9759 |
| RoBERTa | 124,647,170 | 475.0 | 3,204.7s | **94.13%** | **94.13%** | **94.13%** | **94.13%** | **0.9853** |

RoBERTa là mô hình tốt nhất trên tập test và được lưu tại `models/best_model/` để ứng dụng Streamlit sử dụng.

Confusion matrix của RoBERTa trên tập test:

| True / Predicted | Negative | Positive |
| --- | ---: | ---: |
| Negative | 2,328 | 142 |
| Positive | 149 | 2,340 |

## Cấu trúc dự án

```text
.
├── app/
│   ├── app.py
│   ├── assets/
│   │   └── style.css
│   ├── components.py
│   ├── config.toml
│   ├── engine.py
│   ├── explain.py
│   ├── insights.py
│   ├── reliability.py
│   └── requirements-app.txt
├── data/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── raw/
│       └── IMDB Dataset.csv
├── models/
│   ├── baseline/
│   ├── bert-base-uncased/
│   ├── best_model/
│   ├── distilbert-base-uncased/
│   ├── roberta-base/
│   └── training_logs/
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Evaluation_Analysis.ipynb
│   └── 05_Experiments_Comparison.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── baseline.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
├── README.md
├── LICENSE
└── requirements.txt
```

## Cài đặt

Tạo môi trường và cài dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu chỉ chạy ứng dụng Streamlit, có thể cài thêm file dành riêng cho app:

```bash
pip install -r app/requirements-app.txt
```

Nên sử dụng GPU khi fine-tune các mô hình Transformer vì thời gian huấn luyện trên CPU sẽ rất lâu.

## Cách chạy pipeline

Đặt file dữ liệu gốc tại:

```text
data/raw/IMDB Dataset.csv
```

Chạy các notebook theo thứ tự:

```bash
jupyter notebook notebooks/01_EDA_Preprocessing.ipynb
jupyter notebook notebooks/02_Baseline_Model.ipynb
jupyter notebook notebooks/03_Model_Training.ipynb
jupyter notebook notebooks/04_Evaluation_Analysis.ipynb
jupyter notebook notebooks/05_Experiments_Comparison.ipynb
```

Ý nghĩa từng notebook:

| Notebook | Nội dung |
| --- | --- |
| `01_EDA_Preprocessing.ipynb` | Khám phá dữ liệu, làm sạch và chia train/validation/test |
| `02_Baseline_Model.ipynb` | Huấn luyện TF-IDF + Logistic Regression |
| `03_Model_Training.ipynb` | Fine-tune DistilBERT, BERT-base và RoBERTa |
| `04_Evaluation_Analysis.ipynb` | Đánh giá mô hình tốt nhất, phân tích lỗi và độ tin cậy |
| `05_Experiments_Comparison.ipynb` | So sánh toàn bộ mô hình và tạo bảng tổng hợp |

## Chạy ứng dụng Streamlit

Ứng dụng sử dụng checkpoint trong `models/best_model/`. Thư mục này hiện chứa RoBERTa fine-tuned.

```bash
streamlit run app/app.py
```

Chức năng chính của app:

- Nhập một movie review và dự đoán cảm xúc.
- Hiển thị confidence, xác suất positive/negative, số token và thời gian suy diễn.
- Chạy batch prediction từ CSV hoặc danh sách review nhập thủ công.
- Xem dashboard metric, biểu đồ đánh giá và bảng so sánh mô hình.
- Giao diện đã được phối lại theo hướng nền giấy sáng, chữ rõ, CTA tối và màu trạng thái nhất quán.

## Artifact đầu ra

Các file quan trọng sau khi chạy pipeline:

```text
models/baseline/
models/distilbert-base-uncased/
models/bert-base-uncased/
models/roberta-base/
models/best_model/
models/training_logs/
results/metrics/
results/figures/
```
Toàn bộ kết quả lưu tại: [Google Drive](https://drive.google.com/drive/folders/1LZFX9i4wC-EZ7BlLKaV8zVZRpqlc736M)

## Ghi chú triển khai

- `src/config.py` quản lý đường dẫn, hyperparameter, label mapping và danh sách model.
- `src/train.py` hỗ trợ fine-tuning với AdamW, linear warmup scheduler, gradient clipping và early stopping.
- `src/evaluate.py` tính accuracy, precision, recall, F1, ROC AUC, classification report và confusion matrix.
- `src/predict.py` đóng gói suy diễn cho Streamlit, tự đọc tokenizer/model từ `models/best_model/`.
- `app/app.py` cung cấp giao diện phân tích, dashboard kết quả và model comparison.

## License

Xem file `LICENSE`.
