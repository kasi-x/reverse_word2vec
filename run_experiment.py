#!/usr/bin/env python3
"""
Quick start script to run the Word2Vec antonym space experiment.

Usage:
    python run_experiment.py                    # Default: GloVe-100, greedy method
    python run_experiment.py --model glove-300  # Use larger GloVe model
    python run_experiment.py --method svd       # Use SVD-based axis construction
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiment import main

if __name__ == "__main__":
    main()
