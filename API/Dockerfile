# Stage 1: Build
FROM python:3.9-slim AS builder

WORKDIR /app

COPY API .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./datasets/dev/recipient_waiting_list.csv ./datasets/dev/
COPY ./datasets/dev/donor_list.csv ./datasets/dev/
COPY ./datasets/dev/transplant_pair_list.csv ./datasets/dev/

# Stage 2: Production
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# Set IS_DOCKER flag
ENV IS_DOCKER=true

EXPOSE 5001

CMD ["python", "api.py"]
