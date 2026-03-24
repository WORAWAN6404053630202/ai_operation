# ======================================================
# STAGE: DEVELOP (Development Environment)
# ======================================================
FROM python:3.11-slim AS develop

# ติดตั้ง system dependencies สำหรับ sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements.txt ก่อนเพื่อใช้ Docker cache
COPY requirements.txt /app/

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code และ data
COPY code/ /app/code/
COPY local_chroma_v2/ /app/local_chroma_v2/

# ตั้งค่า environment variables
ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 3000

# รัน uvicorn ด้วย auto-reload สำหรับ development
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]

# ======================================================
# STAGE: STAGING (Pre-production Testing)
# ======================================================
FROM python:3.11-slim AS staging

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ /app/code/
COPY local_chroma_v2/ /app/local_chroma_v2/

ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

# ใช้ --reload เหมือน dev เพื่อทดสอบ
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]

# ======================================================
# STAGE: PRODUCTION (Optimized for Performance & Security)
# ======================================================
FROM python:3.11-slim AS prod

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libc6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy and install requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY code/ /app/code/
COPY local_chroma_v2/ /app/local_chroma_v2/
COPY pyproject.toml /app/

# Create necessary directories and set permissions
RUN mkdir -p /app/code/sessions /app/chroma_db && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app/code
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/api/operation/healthcheck || exit 1

# Expose port
EXPOSE 3000

# Production: Multi-worker setup with optimization
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4", "--loop", "uvloop", "--http", "httptools"]