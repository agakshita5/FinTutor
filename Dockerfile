FROM python:3.12-slim
WORKDIR /app

# CPU-only torch: the default wheel bundles CUDA (~2 GB) that a CPU Space never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Build the vector index at image build time so the container starts in seconds.
# The dataset comes from datasets/ if present, otherwise from Google Drive using
# the DATASET_FILE_ID build secret.
RUN --mount=type=secret,id=DATASET_FILE_ID,mode=0444,required=false \
    python build_index.py

ENV PORT=7860
EXPOSE 7860
# One worker: each worker loads its own copy of the embedding model.
CMD gunicorn --workers 1 --threads 4 --timeout 60 --bind 0.0.0.0:$PORT app:app
