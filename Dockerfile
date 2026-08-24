FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev,serving]"

COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./

# Build the layers and train at image build time so the container starts able to
# answer. Training is 100 seconds; doing it at startup would make every replica
# pay it and would make the readiness probe lie.
RUN python scripts/build_data.py \
 && python -m disputes.cli build \
 && python -m disputes.cli train

EXPOSE 8000
ENTRYPOINT ["python", "-m", "disputes.cli"]
CMD ["serve", "--host", "0.0.0.0"]
