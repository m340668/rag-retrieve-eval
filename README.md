## 📌 專案目標

透過多種 Retrieve 策略於不同類型的資料集（純文字及文圖混合）中進行系統化測試，全面評估並分析 RAG 模型的 Retrieve 效能，以協助企業導入並最佳化其內部知識搜尋應用。

A systematic benchmarking framework leveraging RAGAS to evaluate RAG retrieval strategies across textual and multimodal datasets, integrated with Qdrant vector database and visualized via an interactive Gradio interface, empowering enterprises to optimize internal knowledge retrieval solutions.

## 📂 資料集

### 純文字資料集
- HotpotQA (來自 [Hugging Face - TimoImhof/HotpotQA-in-SQuAD-format](https://huggingface.co/datasets/TimoImhof/HotpotQA-in-SQuAD-format))
- 未來新增：中文繁體資料集（如 DRCD、ADL QA）

### 文圖混合資料集
- SlideVQA (來自 [Hugging Face - NTT-hil-insight/SlideVQA](https://huggingface.co/datasets/NTT-hil-insight/SlideVQA))
- 未來新增：中文繁體圖文資料集（如中文發票、合約、企業內部文件等自製資料集）

## 🛠 專案資料夾架構

```bash
rag_eval_project/
├── datasets/
│   ├── pure_text/
│   │   ├── HotpotQA/
│   │   └── zh_tw/ (待新增)
│   └── text_image/
│       ├── SlideVQA/
│       └── zh_tw/ (待新增)
│
├── retrieval_modules/
│   ├── bm25.py
│   ├── embedding_cosine.py
│
├── indexing/
│   ├── qdrant_setup.py (Qdrant本地索引設定)
│   └── index_data.py (建立與更新索引)
│
├── evaluation/
│   ├── ragas_evaluator.py (使用 RAGAS 框架評估)
│   └── run_evaluation.py
│
├── ui/
│   └── app.py (Gradio 呈現結果)
│
├── utils/
│   ├── preprocessing.py
│   ├── dataset_loader.py
│   ├── dataset_downloader.py (資料集下載與準備)
│   └── ocr_extractor.py
│
├── results/
│   ├── reports/
│   └── visualizations/
│
├── config.yaml
├── requirements.txt
└── main.py
```

## ⚙️ 專案流程

### Step 1️⃣ 資料下載與預處理
- 使用統一腳本 (`dataset_downloader.py`) 從 Hugging Face 自動下載所需資料集
- 純文字：資料清洗、標準化（CSV）
- 文圖資料：OCR萃取（使用 PaddleOCR、Tesseract），生成標準化檔案

### Step 2️⃣ 資料索引
- 使用 Docker 本地部署的 Qdrant 建立 Embedding 索引
- 支援 Embedding 模型：OpenAI Embedding, Sentence Transformers, Hugging Face

### Step 3️⃣ Retrieve 策略開發
- 純文字檢索策略（Embedding Cosine, BM25, 待新增）
- 多模態檢索策略（Multimodal Embeddings, OCR+Embedding 待新增）

### Step 4️⃣ 自動化評估
- 使用 RAGAS 框架統一評估
- 計算 Recall@K, Precision@K, MRR, NDCG 等
- 產生視覺化報告

### Step 5️⃣ 結果分析與視覺化
- 使用 Gradio 介面互動式呈現結果
- 分析各種策略在不同資料集的效能表現
- 提供最佳策略建議給企業客戶

## 📝 範例設定檔 (`config.yaml`)

```yaml
datasets:
  pure_text:
    - HotpotQA
    - zh_tw (pending)

  text_image:
    - SlideVQA
    - zh_tw (pending)

retrieval_methods:
  - bm25
  - embedding_cosine

indexing:
  vector_db: qdrant
  embedding_model: sentence-transformers/multi-qa-mpnet-base-dot-v1

evaluation:
  framework: ragas

output_path: results/
```

## 🚀 專案執行

```bash
# 安裝相依套件
pip install -r requirements.txt

# 下載資料集 (基本用法)
python utils/dataset_downloader.py

# 下載需要驗證的資料集 (使用 Hugging Face API 權杖)
# 1. 前往 https://huggingface.co/settings/tokens 取得您的權杖
# 2. 使用權杖執行下載
python utils/dataset_downloader.py --hf-token YOUR_HUGGING_FACE_TOKEN

# 執行主程式
python main.py --config config.yaml

# 啟動視覺化介面
python ui/app.py
```

## 📈 專案產出

```
results/
├── evaluation_summary.csv
├── visualizations/
│   ├── recall.png
│   ├── precision.png
│   └── ndcg_mrr.png
└── reports/
    └── strategy_analysis_report.md
```

