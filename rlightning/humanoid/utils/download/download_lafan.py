import requests
import os
import zipfile
import shutil
import traceback

from rlightning.utils.progress import get_progress

# --- Configuration ---
OUTPUT_DATA_DIR = ".data/lafan1"

# Direct download link for the LAFAN1 dataset archive.
DATA_ARCHIVE_URL = "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/refs/heads/master/lafan1/lafan1.zip"
ARCHIVE_FILENAME = "lafan1.zip"
TEMP_ARCHIVE_PATH = os.path.join(os.getcwd(), ARCHIVE_FILENAME)


def download_and_extract_data():
    """
    Downloads the large dataset archive from the provided URL and extracts it.
    Uses streaming download and a progress bar.
    """

    print("--- LAFAN Dataset Download & Extraction Script ---")

    url = DATA_ARCHIVE_URL
    output_dir = OUTPUT_DATA_DIR
    archive_path = TEMP_ARCHIVE_PATH

    print(f"\n1. Attempting to download data from: {url}")

    if os.path.exists(output_dir):
        print(
            f"   Output directory '{output_dir}' already exists. Delete it manually if you want to re-download."
        )
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Ensure necessary directory exists before starting the download
    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)

    try:
        # Use streaming download to handle large files
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        progress = get_progress()
        task = progress.add_task("[green]Downloading LAFAN Dataset...", total=total_size)

        with open(archive_path, "wb") as f:
            for data in response.iter_content(block_size):
                # t.update(len(data))
                progress.update(task, advance=len(data))
                f.write(data)

        progress.update(
            task, description=f"[bold green]Download complete! File saved to '{archive_path}'."
        )

        # Extraction step
        print("\n2. Extracting data...")

        # Extract the contents directly into the output directory
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            # Extract to a temporary folder first to handle potential top-level directories within the zip
            temp_extract_dir = os.path.join(output_dir, "temp_extract")
            zip_ref.extractall(temp_extract_dir)

        # Move contents from the typical top-level folder within the archive
        extracted_files = os.listdir(temp_extract_dir)

        # Heuristic: If there is a single top-level folder, move its contents up
        if len(extracted_files) == 1 and os.path.isdir(
            os.path.join(temp_extract_dir, extracted_files[0])
        ):
            print(f"   Moving contents of '{extracted_files[0]}' up one level.")
            source_dir = os.path.join(temp_extract_dir, extracted_files[0])
            for item in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir, item), output_dir)
            shutil.rmtree(temp_extract_dir)
        else:
            # Otherwise, move all files from temp_extract_dir to output_dir
            for item in os.listdir(temp_extract_dir):
                shutil.move(os.path.join(temp_extract_dir, item), output_dir)
            shutil.rmtree(temp_extract_dir)

        print(f"   Extraction complete. LAFAN data is available in the '{output_dir}' directory.")

        # Clean up the zip file
        os.remove(archive_path)
        print(f"   Cleaned up archive: {os.path.basename(archive_path)}")

    except requests.exceptions.RequestException as e:
        print(f"   ERROR during download or connection: {e}")
    except zipfile.BadZipFile:
        print("   ERROR: The downloaded file is not a valid ZIP archive.")
    except Exception as e:
        print(f"   An unexpected error occurred: {e}, traceback: {traceback.format_exc()}")

    finally:
        # Ensure the downloaded archive is cleaned up even on failure
        if os.path.exists(archive_path):
            os.remove(archive_path)

    print("\n--- Script Finished ---")
    print(f"LAFAN files are now located in: {output_dir}")


if __name__ == "__main__":
    download_and_extract_data()
