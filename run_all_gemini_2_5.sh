#!/bin/bash
mkdir -p ./results/gemini-2.5-flash

python -u main.py \
    --model gemini-2.5-flash \
    --output-dir ./results/gemini-2.5-flash \
    --circuits all
