import os

# Set KAGGLE_API_TOKEN externally; never store credentials in source.

import kagglehub

print("Checking Kaggle authentication...")
print(kagglehub.whoami())

print("\nDownloading WikiArt metadata...")

path = kagglehub.dataset_download(
    "antoinegruson/-wikiart-all-images-120k-link"
)

print("\nDataset downloaded to:")
print(path)
