# Build stage for API
FROM python:3.11-slim as api
WORKDIR /app
COPY riskshield/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY riskshield/src /app/src
COPY riskshield/out /app/out
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Build stage for UI
FROM python:3.11-slim as ui
WORKDIR /app
# Install streamlit and requests
RUN pip install --no-cache-dir streamlit requests pandas plotly
COPY frontend/streamlit_app.py .
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
