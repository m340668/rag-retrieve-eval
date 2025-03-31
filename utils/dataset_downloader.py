import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datasets import load_dataset
import shutil
import argparse

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, config_path: str = "config.yaml", hf_token: Optional[str] = None):
        """初始化資料集下載器
        
        Args:
            config_path: 設定檔路徑
            hf_token: Hugging Face 存取權杖，用於存取需驗證的資料集
        """
        self.config = self._load_config(config_path)
        self.base_path = Path("datasets")
        self.hotpotqa_dir = self.base_path / "pure_text" / "HotpotQA"
        self.slidevqa_dir = self.base_path / "text_image" / "SlideVQA"
        self.hf_token = hf_token
        self._setup_directories()

    def _load_config(self, config_path: str) -> Dict:
        """載入設定檔"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"無法載入設定檔: {e}")
            raise

    def _setup_directories(self):
        """建立必要的目錄結構"""
        for dataset_type in ['pure_text', 'text_image']:
            for dataset in self.config['datasets'][dataset_type]:
                if dataset != 'zh_tw':  # 跳過待新增的資料集
                    path = self.base_path / dataset_type / dataset
                    path.mkdir(parents=True, exist_ok=True)

    def download_hotpotqa(self):
        """從 Hugging Face 下載 HotpotQA 資料集"""
        logger.info("開始下載 HotpotQA 資料集...")
        try:
            # 確保目錄存在
            os.makedirs(self.hotpotqa_dir, exist_ok=True)
            
            # 從 Hugging Face 下載資料集
            dataset = load_dataset("TimoImhof/HotpotQA-in-SQuAD-format")
            
            # 處理每個分割
            for split in dataset:
                df = pd.DataFrame(dataset[split])
                output_file = os.path.join(self.hotpotqa_dir, f"{split}.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"已儲存 {split} 分割到 {output_file}")
            
            logger.info("HotpotQA 資料集下載完成")
        except Exception as e:
            logger.error(f"下載 HotpotQA 資料集時發生錯誤: {str(e)}")

    def download_slidevqa(self):
        """從 Hugging Face 下載 SlideVQA 資料集"""
        logger.info("開始下載 SlideVQA 資料集...")
        try:
            # 確保目錄存在
            os.makedirs(self.slidevqa_dir, exist_ok=True)
            
            # 嘗試下載資料集
            try:
                # 從 Hugging Face 下載資料集
                dataset = load_dataset("NTT-hil-insight/SlideVQA", use_auth_token=self.hf_token)
                
                # 處理每個分割
                for split in dataset:
                    df = pd.DataFrame(dataset[split])
                    output_file = os.path.join(self.slidevqa_dir, f"{split}.csv")
                    df.to_csv(output_file, index=False)
                    logger.info(f"已儲存 {split} 分割到 {output_file}")
                    
                    # 如果有圖片資料，也下載下來
                    if 'image' in df.columns or 'image_path' in df.columns:
                        images_dir = self.slidevqa_dir / "images"
                        images_dir.mkdir(exist_ok=True)
                        # 下載圖片的邏輯需根據資料集的結構進行調整
                
                logger.info("SlideVQA 資料集下載完成")
            except Exception as e:
                if "gated dataset" in str(e).lower():
                    logger.error(f"SlideVQA 是一個需要訪問許可的資料集，請按照以下步驟操作：")
                    logger.error(f"1. 前往 https://huggingface.co/datasets/NTT-hil-insight/SlideVQA")
                    logger.error(f"2. 登入您的 Hugging Face 帳號")
                    logger.error(f"3. 點擊頁面上的 'Access repository' 並接受使用條款")
                    logger.error(f"4. 從您的個人設定頁面 https://huggingface.co/settings/tokens 生成一個 API 令牌")
                    logger.error(f"5. 在命令行執行: huggingface-cli login")
                    logger.error(f"   或者在程式碼中使用: load_dataset(..., use_auth_token='您的令牌')")
                else:
                    raise
        except Exception as e:
            logger.error(f"下載 SlideVQA 資料集時發生錯誤: {str(e)}")

    def download_all(self):
        """下載所有資料集"""
        logger.info("開始下載所有資料集...")
        
        # 純文字資料集
        if "HotpotQA" in self.config['datasets']['pure_text']:
            self.download_hotpotqa()
        
        # 文圖混合資料集
        if "SlideVQA" in self.config['datasets']['text_image']:
            self.download_slidevqa()
        
        logger.info("所有資料集下載完成！")

    def set_huggingface_token(self, token: str):
        """設定 Hugging Face 存取權杖
        
        Args:
            token: Hugging Face API 權杖
        """
        self.hf_token = token
        logger.info("已設定 Hugging Face API 權杖")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='下載並處理評估資料集')
    parser.add_argument('--config', type=str, default='config.yaml', help='設定檔路徑')
    parser.add_argument('--hf-token', type=str, help='Hugging Face API 權杖')
    args = parser.parse_args()
    
    try:
        downloader = DatasetDownloader(config_path=args.config, hf_token=args.hf_token)
        downloader.download_all()
    except Exception as e:
        logger.error(f"下載過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main() 