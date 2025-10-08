import requests
import zipfile
import os
from tqdm import tqdm

# --- Configuration (Updated to the latest recommended checkpoint) ---
CHECKPOINT_URL = "https://storage.googleapis.com/bleurt-checkpoints/bleurt-20-d12.zip"
TARGET_DIR = "bleurt_checkpoints"
ZIP_FILENAME = "bleurt-20-d12.zip"
CHECKPOINT_NAME = "BLEURT-20-D12"

def download_and_extract_bleurt():
    """
    Downloads and extracts the specified BLEURT checkpoint.
    """
    print("="*60)
    print(f"      {CHECKPOINT_NAME} Checkpoint Downloader      ")
    print("="*60)

    # Create the target directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Created directory: {TARGET_DIR}")

    zip_path = os.path.join(TARGET_DIR, ZIP_FILENAME)
    # The extracted folder name might be different, let's assume it matches the zip
    checkpoint_folder_name_in_zip = "bleurt-20-d12" # This is often the case
    checkpoint_path = os.path.join(TARGET_DIR, checkpoint_folder_name_in_zip)


    # Check if the checkpoint is already downloaded and extracted
    if os.path.exists(checkpoint_path):
        print(f"SUCCESS: {CHECKPOINT_NAME} checkpoint is already downloaded and extracted.")
        print(f"Location: {os.path.abspath(checkpoint_path)}")
        print("="*60)
        return

    # Download the file with a progress bar
    print(f"Downloading {CHECKPOINT_NAME} from: {CHECKPOINT_URL}")
    try:
        response = requests.get(CHECKPOINT_URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        print("Download complete.")

        # Extract the zip file
        print(f"Extracting {ZIP_FILENAME}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)
        print("Extraction complete.")
        
        # Rename the extracted folder if necessary
        extracted_folder = os.path.join(TARGET_DIR, "bleurt-20-d12")
        if os.path.exists(extracted_folder) and not os.path.exists(checkpoint_path):
             os.rename(extracted_folder, checkpoint_path)


        # Clean up the zip file
        os.remove(zip_path)
        print(f"Removed temporary file: {zip_path}")

        print(f"\nSUCCESS: {CHECKPOINT_NAME} checkpoint is now ready to use.")
        print(f"Location: {os.path.abspath(checkpoint_path)}")
        print("="*60)

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Failed to download the checkpoint. The URL may be invalid or you may have a network issue.")
        print(e)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred.")
        print(e)

if __name__ == "__main__":
    download_and_extract_bleurt()
