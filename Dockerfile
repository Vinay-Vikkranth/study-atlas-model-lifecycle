FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib ATLAS_HOST=0.0.0.0 PORT=8740 ATLAS_DATA_DIR=/app/runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home atlas
COPY --chown=atlas:atlas . .
RUN mkdir -p /app/runtime && chown atlas:atlas /app/runtime
USER atlas
EXPOSE 8740
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8740')+'/api/ready',timeout=3)"
CMD ["python", "scripts/start.py"]
