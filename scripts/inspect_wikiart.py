from pathlib import Path

import pandas as pd


CACHE_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "antoinegruson"
    / "-wikiart-all-images-120k-link"
    / "versions"
    / "2"
)

print("Dataset directory:")
print(CACHE_DIR)

print("\n=== FILES ===")

files = [p for p in CACHE_DIR.rglob("*") if p.is_file()]

for file in files:
    size_mb = file.stat().st_size / 1024 / 1024
    print(f"{file.relative_to(CACHE_DIR)} - {size_mb:.2f} MB")


csv_files = list(CACHE_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in WikiArt dataset.")

print("\n=== CSV FILES ===")

for file in csv_files:
    print(file.name)


csv_path = csv_files[0]

print(f"\nReading: {csv_path.name}")

df = pd.read_csv(csv_path)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")

for column in df.columns:
    print(f"- {column}")

print("\n=== FIRST 5 ROWS ===")
print(df.head().to_string())

print("\n=== NULL VALUES ===")
print(df.isna().sum().sort_values(ascending=False).to_string())


style_columns = [
    column
    for column in df.columns
    if "style" in column.lower()
]

url_columns = [
    column
    for column in df.columns
    if "url" in column.lower()
    or "link" in column.lower()
]

artist_columns = [
    column
    for column in df.columns
    if "artist" in column.lower()
]


print("\n=== POSSIBLE STYLE COLUMNS ===")
print(style_columns)

print("\n=== POSSIBLE IMAGE URL COLUMNS ===")
print(url_columns)

print("\n=== POSSIBLE ARTIST COLUMNS ===")
print(artist_columns)


if style_columns:
    style_col = style_columns[0]

    print(f"\n=== STYLE COUNTS ({style_col}) ===")

    counts = (
        df[style_col]
        .dropna()
        .astype(str)
        .value_counts()
    )

    print(counts.to_string())