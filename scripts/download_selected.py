from pathlib import Path
import time
import re

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "antoinegruson"
    / "-wikiart-all-images-120k-link"
    / "versions"
    / "2"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "selected"

SELECTED_STYLES = [
    "Baroque",
    "Impressionism",
    "Cubism",
    "Pop-Art",
    "Ukiyo-e",
]

IMAGES_PER_STYLE = 1000

RANDOM_STATE = 42

REQUEST_TIMEOUT = 20


# ============================================================
# HELPERS
# ============================================================

def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:100]


def find_column(columns, candidates):
    lower_mapping = {
        column.lower(): column
        for column in columns
    }

    # exact matches first
    for candidate in candidates:
        if candidate.lower() in lower_mapping:
            return lower_mapping[candidate.lower()]

    # partial matches
    for column in columns:
        lower = column.lower()

        for candidate in candidates:
            if candidate.lower() in lower:
                return column

    return None


def download_image(url, destination):
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        stream=True,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def validate_image(path):
    try:
        with Image.open(path) as image:
            image.verify()

        # open second time because verify() invalidates decoder state
        with Image.open(path) as image:
            image.convert("RGB")

        return True

    except Exception:
        return False


# ============================================================
# LOAD METADATA
# ============================================================

csv_files = list(DATASET_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found.")

csv_path = csv_files[0]

print(f"Using metadata: {csv_path}")

df = pd.read_csv(csv_path)

print("\nColumns:")
for column in df.columns:
    print(f"- {column}")


style_col = find_column(
    df.columns,
    [
        "Style",
        "style",
    ]
)

url_col = find_column(
    df.columns,
    [
        "Image URL",
        "ImageURL",
        "image_url",
        "URL",
        "Link",
    ]
)

artist_col = find_column(
    df.columns,
    [
        "Artist",
        "artist",
    ]
)

artwork_col = find_column(
    df.columns,
    [
        "Artwork",
        "Title",
        "Name",
    ]
)


print("\nDetected columns:")
print("Style:", style_col)
print("URL:", url_col)
print("Artist:", artist_col)
print("Artwork:", artwork_col)


if style_col is None:
    raise RuntimeError("Could not detect style column.")

if url_col is None:
    raise RuntimeError(
        "Could not detect image URL column. "
        "Look at the printed column names."
    )


# ============================================================
# DOWNLOAD
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest_rows = []


for style in SELECTED_STYLES:

    style_df = df[df[style_col] == style].copy()

    print()
    print("=" * 60)
    print(f"{style}: {len(style_df)} available")
    print("=" * 60)

    # Shuffle so we don't systematically take only early entries
    style_df = style_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    style_dir = OUTPUT_DIR / safe_filename(style)
    style_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    progress = tqdm(
        total=IMAGES_PER_STYLE,
        desc=style
    )

    for index, row in style_df.iterrows():

        if successful >= IMAGES_PER_STYLE:
            break

        url = row[url_col]

        if pd.isna(url):
            continue

        filename = f"{successful:04d}.jpg"
        destination = style_dir / filename

        try:
            download_image(
                str(url),
                destination
            )

            if not validate_image(destination):
                destination.unlink(missing_ok=True)
                failed += 1
                continue

            manifest_rows.append({
                "filename": str(
                    destination.relative_to(PROJECT_ROOT)
                ),
                "style": style,
                "artist": (
                    row[artist_col]
                    if artist_col is not None
                    else None
                ),
                "artwork": (
                    row[artwork_col]
                    if artwork_col is not None
                    else None
                ),
                "source_url": url,
            })

            successful += 1
            progress.update(1)

        except Exception:
            destination.unlink(missing_ok=True)
            failed += 1

    progress.close()

    print(
        f"{style}: "
        f"{successful} downloaded, "
        f"{failed} failed"
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = pd.DataFrame(manifest_rows)

manifest_path = OUTPUT_DIR / "manifest.csv"

manifest.to_csv(
    manifest_path,
    index=False
)

print("\nFinished.")
print(f"Total images: {len(manifest)}")
print(f"Manifest: {manifest_path}")

print("\nClass distribution:")
print(manifest["style"].value_counts())