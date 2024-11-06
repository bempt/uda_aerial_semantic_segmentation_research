#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/source
mkdir -p data/target
mkdir -p results/plots
mkdir -p results/metrics