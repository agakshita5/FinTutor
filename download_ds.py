import os
import gdown
from dotenv import load_dotenv

load_dotenv()

FILE_ID = os.getenv("DATASET_FILE_ID")
DATASET_PATH = "datasets/final_combined.csv"

def download_dataset():
    os.makedirs("datasets", exist_ok=True)
    
    if os.path.exists(DATASET_PATH):
        print(f"Dataset found at {DATASET_PATH}")
        return True
    
    if not FILE_ID:
        print("DATASET_FILE_ID not set in .env")
        return False
    
    print("Downloading dataset from Google Drive...")
    try:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, DATASET_PATH, quiet=False)
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    download_dataset()