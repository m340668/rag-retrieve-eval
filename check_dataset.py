import pandas as pd
import os

# 檢查 HotpotQA 資料集
dataset_path = 'datasets/pure_text/HotpotQA'
if os.path.exists(dataset_path):
    print(f"===== HotpotQA 資料集結構 =====")
    for file in os.listdir(dataset_path):
        if file.endswith('.csv'):
            file_path = os.path.join(dataset_path, file)
            df = pd.read_csv(file_path)
            print(f"\n檔案: {file}")
            print(f"行數: {len(df)}")
            print(f"列名: {df.columns.tolist()}")
            print(f"資料預覽:\n{df.head(2).to_string()}")
else:
    print(f"找不到資料集路徑: {dataset_path}") 