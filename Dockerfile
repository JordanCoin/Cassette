FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system -e "."

# Copy application code
COPY libs/ libs/
COPY services/ services/
COPY contracts/ contracts/

# Create data directory
RUN mkdir -p /app/data/gateway

ENV CASSETTE_PROVIDER=mock
ENV CASSETTE_LLAMA_CPP_URL=http://llama:8080
ENV CASSETTE_SEARCH_URL=http://searxng:8080

EXPOSE 8000

CMD ["uvicorn", "services.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
