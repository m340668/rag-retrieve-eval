# RAG 檢索評估系統設定指南

## 環境準備

1. 安裝必要套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 確保安裝了以下主要依賴：
   - `datasets`：用於下載 Hugging Face 資料集
   - `pandas`：用於資料處理
   - `qdrant-client`：向量資料庫
   - `sentence-transformers`：文字嵌入
   - `ragas`：評估框架

## 資料集下載

### 基本下載
直接執行下載腳本：
```bash
python utils/dataset_downloader.py
```

### 需要 Hugging Face 權杖的資料集下載
某些資料集如 SlideVQA 需要 Hugging Face 權杖才能訪問：

1. 前往 https://huggingface.co/datasets/NTT-hil-insight/SlideVQA
2. 登入您的 Hugging Face 帳號
3. 點擊頁面上的 'Access repository' 並接受使用條款
4. 從您的個人設定頁面 https://huggingface.co/settings/tokens 生成一個 API 權杖
5. 使用權杖執行下載：
   ```bash
   python utils/dataset_downloader.py --hf-token YOUR_HUGGING_FACE_TOKEN
   ```

## 資料集結構

### HotpotQA
- 位置：`datasets/pure_text/HotpotQA/`
- 檔案：
  - `unmodified.csv`：原始版本
  - `modified_30_percent.csv`：30% 修改版本
  - `modified_100_percent.csv`：100% 修改版本
- 欄位：
  - `id`：問題唯一識別碼
  - `question`：問題文字
  - `context`：包含答案的上下文
  - `answers`：答案及其在上下文中的位置

### SlideVQA (需要 Hugging Face 權杖)
- 位置：`datasets/text_image/SlideVQA/`
- 需要接受使用條款才能下載

## 使用方法

1. 下載資料集：
   ```bash
   python utils/dataset_downloader.py [--hf-token YOUR_TOKEN]
   ```

2. 運行評估：
   ```bash
   python main.py --config config.yaml
   ```

3. 查看結果：
   ```bash
   python ui/app.py
   ```

## 故障排除

### SlideVQA 資料集無法下載
如果遇到「需要訪問許可」的錯誤，請確保：
1. 已接受 Hugging Face 上的使用條款
2. 使用正確的 API 權杖
3. 權杖有足夠的權限

### 其他問題
查看日誌檔案以獲取詳細錯誤信息。 