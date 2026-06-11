#!/usr/bin/env bash
# Run the IR system UI locally (Linux / Mac)
# Data files must be in ir-system/data (or set IR_DATA_DIR)
cd "$(dirname "$0")/.."
streamlit run ui/app.py
