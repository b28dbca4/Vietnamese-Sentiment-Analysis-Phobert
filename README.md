# IMDb Sentiment Analysis with Transformer Fine-tuning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## Tổng quan

Dự án xây dựng hệ thống phân tích cảm xúc cho bộ dữ liệu IMDb 50K Movie Reviews. Pipeline bao gồm mô hình baseline TF-IDF + Logistic Regression, ba mô hình Transformer fine-tuning (DistilBERT, BERT-base, RoBERTa), đánh giá định lượng, phân tích lỗi và ứng dụng Streamlit phục vụ suy diễn thời gian thực.

Notebook `01_EDA_Preprocessing.ipynb` đã hoàn thành bước EDA, làm sạch nhẹ văn bản và chia dữ liệu theo tỉ lệ 80/10/10. Các bước còn lại sử dụng trực tiếp dữ liệu đã xử lý trong `data/processed/`.

## Dữ liệu

- Nguồn: IMDb Dataset of 50K Movie Reviews.
- Bài toán: phân loại cảm xúc nhị phân `negative` / `positive`.
- Sau loại trùng: khoảng 49,582 mẫu.
- Cột đầu vào cho modeling: `clean_text`, `label`, `label_id`.
- Tiền xử lý: xóa HTML, xóa URL, chuẩn hóa khoảng trắng.
- Không lowercase, không loại stopword, không stemming, không lemmatization để giữ văn bản phù hợp với tokenizer Transformer.

## Mô hình

Thiết kế thí nghiệm gồm bốn mô hình:

| Model | Vai trò | Ghi chú |
| --- | --- | --- |
| TF-IDF + Logistic Regression | Baseline | Nhanh, dễ giải thích, tạo mốc hiệu năng |
| DistilBERT | Transformer nhẹ | 6 layers, cân bằng tốc độ và độ chính xác |
| BERT-base | Transformer chuẩn | 12 layers, mốc so sánh phổ biến |
| RoBERTa | Transformer cải tiến | Dynamic masking, no NSP, BPE tokenizer |

Các Transformer sử dụng cấu hình mặc định: `max_length=256`, `batch_size=16`, `epochs=3`, `learning_rate=2e-5`, `warmup_ratio=0.1`, `weight_decay=0.01`.

## Cấu trúc dự án

```text
imdb-sentiment-analysis/
├── app/
│   ├── app.py
│   └── assets/style.css
├── data/
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── raw/IMDB Dataset.csv
├── models/
│   ├── baseline/
│   ├── distilbert-base-uncased/
│   ├── bert-base-uncased/
│   ├── roberta-base/
│   ├── best_model/
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
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu chạy trên Google Colab, nên bật GPU trước khi fine-tune các mô hình Transformer.

## Cách chạy

Huấn luyện baseline:

```bash
jupyter notebook notebooks/02_Baseline_Model.ipynb
```

Fine-tune Transformer:

```bash
jupyter notebook notebooks/03_Model_Training.ipynb
```

Đánh giá và so sánh:

```bash
jupyter notebook notebooks/04_Evaluation_Analysis.ipynb
jupyter notebook notebooks/05_Experiments_Comparison.ipynb
```

Chạy ứng dụng:

```bash
streamlit run app/app.py
```

Trước khi dùng app cho suy diễn thật, cần copy hoặc symlink checkpoint thắng cuộc vào `models/best_model/`.

## Kết quả

Các file JSON trong `results/metrics/` và hình trong `results/figures/` được tạo sau khi chạy notebook tương ứng. Các file placeholder hiện có chỉ đánh dấu trạng thái chưa sinh kết quả, không phải metric thật.

Mốc kỳ vọng theo tài liệu dự án:

| Metric | Kỳ vọng |
| --- | --- |
| Accuracy | 92-94% |
| F1 Macro | 92-94% |
| AUC | 0.96-0.98 |
| Precision | 91-95% |
| Recall | 91-95% |

## Ghi chú triển khai

- `src/train.py` hiển thị tqdm theo từng epoch, cập nhật mỗi batch.
- Thanh tqdm chuyển màu đỏ -> vàng/cam -> xanh khi terminal hỗ trợ màu.
- Postfix huấn luyện hiển thị running average của loss, accuracy, F1 macro, precision macro và recall macro.
- AUC được tính ở validation/evaluation khi xác suất dự đoán phù hợp, không ép tính live theo từng batch.

## License

Xem file `LICENSE`.
