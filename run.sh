#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Run dataset preparation (optional after raw dataset is available)..."
# python -m training.prepare_dataset

echo "Train multitask IndoBERT..."
# python -m training.train

echo "Evaluate model..."
# python -m training.evaluate

echo "Start Streamlit app..."
streamlit run app/main.py
