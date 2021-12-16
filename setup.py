import os
import nltk
import csv
from pylab import sys

print("Creating directories...")

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Natural Language Toolkit (NLTK) downloads
nltk.download('punkt')
nltk.download('wordnet')