#!/bin/bash
set -e

# Function to start the training process
start_training() {
    echo "Starting model training..."
    python src/train.py \
        --data-dir ./data \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs \
        --batch-size 128 \
        --epochs 100 \
        --learning-rate 0.001 \
        --num-workers 4
}

# Function to start the API server
start_server() {
    echo "Starting API server..."
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
}

# Main entrypoint logic
case "$1" in
    "train")
        start_training
        ;;
    "serve")
        start_server
        ;;
    *)
        if [ -f "checkpoints/best_model.pth" ]; then
            echo "Model checkpoint found, starting server..."
            start_server
        else
            echo "No model checkpoint found, starting training..."
            start_training
            start_server
        fi
        ;;
esac 