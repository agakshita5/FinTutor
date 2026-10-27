import os
import gdown
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = "datasets/final_combined.csv"
SECRET_PATH = "/run/secrets/DATASET_FILE_ID"  # Docker build secret (Hugging Face Spaces)


def _file_id():
    if os.getenv("DATASET_FILE_ID"):
        return os.getenv("DATASET_FILE_ID")
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH) as f:
            return f.read().strip()
    return None


def download_dataset():
    os.makedirs("datasets", exist_ok=True)

    if os.path.exists(DATASET_PATH):
        print(f"Dataset found at {DATASET_PATH}")
        return True

    file_id = _file_id()
    if not file_id:
        print("DATASET_FILE_ID not set in .env")
        return False

    print("Downloading dataset from Google Drive...")
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, DATASET_PATH, quiet=False)
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


if __name__ == "__main__":
    download_dataset()
