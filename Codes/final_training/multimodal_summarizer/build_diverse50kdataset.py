import os
import json
import logging
import pandas as pd
import requests
import easyocr
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datetime import datetime
import time
import torch
import sys
import random

# --- FIX FOR OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------

# --- Configuration ---

# 1. SET YOUR DATA SOURCE(S) HERE
DATA_PARENT_DIR = r"C:\Users\admin\Downloads\articles\articles"
DATA_SOURCE_DIRS = [
    os.path.join(DATA_PARENT_DIR, r"bbc_1"),
    os.path.join(DATA_PARENT_DIR, r"bbc_2"),
    os.path.join(DATA_PARENT_DIR, r"guardian_articles"),
    os.path.join(DATA_PARENT_DIR, r"usa_today_articles"),
    os.path.join(DATA_PARENT_DIR, r"washington_post_articles\json"),
]

# 2. SET YOUR OUTPUT DIRECTORY
OUTPUT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\diverse50kdataset_balanced"

# 3. SET TOTAL SAMPLES
TOTAL_DATASET_SIZE = 50000

# --- Derived Paths ---
IMAGE_SAVE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, f"multimodal_diverse50kdataset_{datetime.now().strftime('%Y%m%d')}.parquet")
LOG_FILENAME = os.path.join(OUTPUT_DIR, f"data_build_log_diverse50kdataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# ---------------------

def setup_logging():
    """Configures logging to file and console."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILENAME, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logging.info("--- Starting Multimodal 50k Balanced Dataset Build ---")
    logging.info(f"Source Dirs: {DATA_SOURCE_DIRS}")
    logging.info(f"Output Dir: {OUTPUT_DIR}")
    logging.info(f"Target Total Size: {TOTAL_DATASET_SIZE}")

def process_json_file(file_path):
    """Extracts key data from a single JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        article_text = data.get("body")
        base_summary = data.get("summary")
        
        relations = data.get("relations")
        if not relations or not isinstance(relations, list) or len(relations) == 0:
            return None
            
        image_content = relations[0].get("content")
        if not image_content:
            return None
            
        image_url = image_content.get("href")
        image_caption = image_content.get("caption")

        if not all([article_text, base_summary, image_url, image_caption]):
            return None
            
        return {
            "article_text": article_text,
            "base_summary": base_summary,
            "image_url": image_url,
            "image_caption": image_caption,
            "original_json_path": file_path
        }
    except Exception as e:
        logging.warning(f"Error parsing JSON {file_path}: {e}")
        return None

def download_image(url, save_path):
    """Downloads an image and saves it locally."""
    try:
        if os.path.exists(save_path):
            return True
            
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            logging.warning(f"Skipped non-image URL: {url} (Content-Type: {content_type})")
            return False

        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
        
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to download image {url}: {e}")
        return False

def get_image_description(image_path, caption, ocr_reader):
    """
    Performs 'Image Triage' (OCR vs. Caption).
    """
    try:
        ocr_result = ocr_reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result)
        
        if len(ocr_text) > 50:
            return f"The image is a document containing the text: {ocr_text}"
        else:
            if caption:
                return f"The image shows: {caption}"
            else:
                return "An image is present."
    except Exception as e:
        logging.error(f"Error during OCR/Triage for {image_path}: {e}")
        if caption:
            return f"The image shows: {caption}"
        else:
            return "An image is present."

def main():
    setup_logging()
    
    logging.info("Loading EasyOCR model...")
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        logging.info("EasyOCR model loaded.")
    except Exception as e:
        logging.error(f"Failed to load EasyOCR: {e}. Check CUDA/PyTorch setup.", exc_info=True)
        return

    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    # --- NEW PROPORTIONAL SAMPLING LOGIC ---
    logging.info("Scanning all source directories to get total file counts...")
    source_file_lists = {}
    total_files = 0
    for data_dir in DATA_SOURCE_DIRS:
        logging.info(f"Scanning directory: {data_dir}")
        source_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    source_files.append(os.path.join(root, file))
        
        if not source_files:
            logging.warning(f"No .json files found in {data_dir}. Skipping.")
            continue
            
        source_file_lists[data_dir] = source_files
        total_files += len(source_files)
        logging.info(f"Found {len(source_files)} files in {data_dir}.")
    
    if total_files == 0:
        logging.error("No .json files found in any source directory. Aborting.")
        return

    logging.info(f"Found {total_files} total files across {len(DATA_SOURCE_DIRS)} sources.")
    logging.info(f"Creating a balanced dataset of {TOTAL_DATASET_SIZE} total samples.")

    master_json_list = []
    for data_dir, file_list in source_file_lists.items():
        # Calculate proportional number of samples
        source_count = len(file_list)
        ratio = source_count / total_files
        num_samples_for_source = int(ratio * TOTAL_DATASET_SIZE)
        
        logging.info(f"Taking {num_samples_for_source} random samples from {data_dir} (Ratio: {ratio:.2%})")
        
        # Shuffle the list for this source
        random.shuffle(file_list)
        
        # Add the selected samples
        master_json_list.extend(file_list[:num_samples_for_source])

    # Shuffle the master list one more time to mix the sources
    random.shuffle(master_json_list)
    logging.info(f"Total files to process after sampling: {len(master_json_list)}")
    # --- END NEW SAMPLING LOGIC ---
        
    processed_data = []
    
    logging.info("Starting processing loop...")
    for file_path in tqdm(master_json_list, desc="Processing Files"):
        data = process_json_file(file_path)
        if not data:
            continue
            
        image_filename = os.path.basename(data['image_url'])
        image_filename = "".join(c for c in image_filename if c.isalnum() or c in ('.', '_', '-')).strip()
        if not image_filename:
             image_filename = f"{os.path.basename(data['original_json_path'])}.jpg"
             
        local_image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        
        if not download_image(data['image_url'], local_image_path):
            continue
        
        image_description = get_image_description(
            local_image_path, 
            data['image_caption'], 
            ocr_reader
        )
        
        final_summary = f"{image_description} {data['base_summary']}"
        
        processed_data.append({
            "article_text": data['article_text'],
            "local_image_path": local_image_path,
            "final_summary": final_summary
        })
        
        # time.sleep(0.01) # Optional: uncomment if you get rate-limited

    logging.info(f"Successfully processed {len(processed_data)} items.")

    if not processed_data:
        logging.warning("No data was processed. Exiting without saving.")
        return

    logging.info(f"Converting to DataFrame and saving to {OUTPUT_FILENAME}...")
    try:
        df = pd.DataFrame(processed_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, OUTPUT_FILENAME)
        
        logging.info(f"--- 60k Balanced Dataset Build Complete! ---")
        logging.info(f"Final dataset saved to {OUTPUT_FILENAME}")
        
    except Exception as e:
        logging.error(f"Failed to save Parquet file: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("--- Process Interrupted by User ---")