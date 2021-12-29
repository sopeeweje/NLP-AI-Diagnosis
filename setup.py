import os
import nltk
import csv
from pylab import sys

print("Creating directories...")

# Create directories
os.makedirs("data", exist_ok=True)
file = open("data/.gitkeep", "w")
file.close()

os.makedirs("results", exist_ok=True)
file = open("results/.gitkeep", "w")
file.close()

os.makedirs("figures", exist_ok=True)
file = open("figures/.gitkeep", "w")
file.close()

# Natural Language Toolkit (NLTK) downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')