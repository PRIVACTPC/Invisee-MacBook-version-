FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas scikit-learn

COPY compute_node.py .
COPY data_handler.py .
COPY secret_sharing_node.py .

CMD ["python", "compute_node.py"]
