"""
MedQuAD Dataset Downloader Script

This script downloads the MedQuAD medical question-answering dataset CSV file
from a reliable mirror repository and places it in the project data/ directory.
It ensures dataset setup is 100% reproducible without committing large CSVs to Git.
"""

import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Primary verified raw CSV mirror for MedQuAD (Optimistix / NIH mirror, ~22.9 MB)
MEDQUAD_URL = "https://raw.githubusercontent.com/Optimistix/medical_QandA_assistant/main/data/medquad.csv"
DEFAULT_OUTPUT_PATH = os.path.join("data", "medquad.csv")


def download_medquad(url: str = MEDQUAD_URL, output_path: str = DEFAULT_OUTPUT_PATH, force: bool = False) -> str:
    """
    Downloads the MedQuAD dataset CSV file if it does not already exist.

    Args:
        url: The direct download URL for the MedQuAD CSV.
        output_path: Target local file path.
        force: If True, redownload even if file exists.

    Returns:
        Absolute path to the downloaded CSV file.
    """
    output_path = os.path.abspath(output_path)
    target_dir = os.path.dirname(output_path)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(output_path) and not force:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"Dataset already exists at '{output_path}' ({file_size_mb:.2f} MB). Skipping download.")
        return output_path

    logging.info(f"Downloading MedQuAD dataset from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None
            downloaded = 0
            block_size = 8192

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size:
                    percent = (downloaded / total_size) * 100
                    sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)")
                    sys.stdout.flush()

        sys.stdout.write("\n")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"Download complete! Saved to '{output_path}' ({file_size_mb:.2f} MB).")
        return output_path
    except Exception as e:
        logging.error(f"Failed to download dataset from {url}. Error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e


if __name__ == "__main__":
    download_medquad()
