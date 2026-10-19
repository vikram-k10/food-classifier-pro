"""Download the Kaggle 'Indian Food Classification' dataset with kagglehub.

    pip install kagglehub
    python scripts/download_dataset.py

Needs a free Kaggle account the first time (kagglehub prompts for credentials, or set
KAGGLE_USERNAME / KAGGLE_KEY). Prints the folder to pass as --data-dir.
"""
import kagglehub

path = kagglehub.dataset_download("l33tc0d3r/indian-food-classification")
print("Dataset downloaded to:", path)
print(f"Train with:  python -m foodai.train --data-dir '{path}' --arch efficientnet_b0")
