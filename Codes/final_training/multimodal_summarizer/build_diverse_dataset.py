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
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- FIX FOR OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------

# --- Configuration ---

# 1. SET YOUR DATA SOURCE(S) HERE
DATA_PARENT_DIR = r"D:\Articles\extracted_articles"
DATA_SOURCE_DIRS = [
    os.path.join(DATA_PARENT_DIR, r"bbc_1\bbcnews_json"),
    os.path.join(DATA_PARENT_DIR, r"bbc_2\bbcnews_stm2json")
]

# 2. SET YOUR OUTPUT DIRECTORY
OUTPUT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\diverse65kdataset_balanced"

# 3. SET TOTAL SAMPLES
TOTAL_DATASET_SIZE = 65000

# 4. PERFORMANCE TUNING
DOWNLOAD_WORKERS = 32
REQUEST_TIMEOUT = 5 # (in seconds)
# ---------------------

# --- Derived Paths ---
IMAGE_SAVE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, f"multimodal_diverse65kdataset_{datetime.now().strftime('%Y%m%d')}.parquet")
LOG_FILENAME = os.path.join(OUTPUT_DIR, f"data_build_log_diverse65kdataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# ** OPTIMIZATION: Cache path for the slow file scan **
FILE_LIST_CACHE = os.path.join(OUTPUT_DIR, "_file_scan_cache.json")
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
    logging.info(f"--- Starting Multimodal {TOTAL_DATASET_SIZE} Balanced Dataset Build (from Extracted Files) ---")
    logging.info(f"Source Dirs: {DATA_SOURCE_DIRS}")
    logging.info(f"Output Dir: {OUTPUT_DIR}")
    logging.info(f"Target Total Size: {TOTAL_DATASET_SIZE}")
    logging.info(f"Using {DOWNLOAD_WORKERS} parallel download workers.")

def create_download_session():
    """Creates a robust requests.Session with retries."""
    session = requests.Session()
    retries = Retry(
        total=3, 
        backoff_factor=0.5, 
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=DOWNLOAD_WORKERS, pool_maxsize=DOWNLOAD_WORKERS)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session

def process_json_file(file_path):
    """Extracts key data from a single JSON file *from disk*."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        
        article_text = data.get("body")
        base_summary = data.get("summary")
        
        # ** OPTIMIZATION: Filter for better data quality **
        if not article_text or len(article_text.split()) < 50:
            return None # Skip short/empty articles
        if not base_summary or len(base_summary.split()) < 5:
            return None # Skip short/empty summaries
        # ----------------------------------------------------
        
        relations = data.get("relations")
        if not relations or not isinstance(relations, list) or len(relations) == 0:
            return None
            
        image_content = relations[0].get("content")
        if not image_content:
            return None
            
        image_url = image_content.get("href")
        image_caption = image_content.get("caption")

        if not all([image_url, image_caption]): # We already checked text/summary
            return None
            
        return {
            "article_text": article_text,
            "base_summary": base_summary,
            "image_url": image_url,
            "image_caption": image_caption,
            "original_json_path": file_path
        }
    except json.JSONDecodeError:
        logging.warning(f"Skipping corrupt/empty JSON: {file_path}")
        return None
    except Exception as e:
        logging.warning(f"Error parsing JSON {file_path}: {e}")
        return None

def download_image(data_item, session):
    """Downloads an image and returns the data_item with the local_path."""
    if not data_item:
        return None

    url = data_item['image_url']
    
    image_filename = os.path.basename(url).split('?')[0] 
    image_filename = "".join(c for c in image_filename if c.isalnum() or c in ('.', '_', '-')).strip()
    if not image_filename:
             image_filename = f"{os.path.basename(data_item['original_json_path'])}.jpg"
        
    local_image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
    
    try:
        if os.path.exists(local_image_path):
            data_item['local_image_path'] = local_image_path
            return data_item
            
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            logging.warning(f"Skipped non-image URL: {url} (Content-Type: {content_type})")
            return None

        with open(local_image_path, 'wb') as f:
            f.write(response.content)
            
        data_item['local_image_path'] = local_image_path
        return data_item
        
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to download image {url}: {e}")
        return None

def get_image_description(image_path, caption, ocr_reader):
    """
    Performs 'Image Triage' (OCR vs. Caption).
    """
    try:
        ocr_result = ocr_reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result)
        
        # ** OPTIMIZATION: Smarter Triage Logic **
        if len(ocr_text) > 50:
            # Case 1: It's a document/chart. Use OCR text only.
            return f"The image is a document containing the text: {ocr_text}"
        elif len(ocr_text) > 5 and caption:
            # Case 2: It's a caption/logo. Combine them.
            return f"The image shows: {caption}. It contains the text: {ocr_text}"
        else:
            # Case 3: OCR is empty/useless. Just use the caption.
            if caption:
                return f"The image shows: {caption}"
            else:
                return "An image is present."
        # -------------------------------------------
            
    except Exception as e:
        # Fallback in case of corrupt image or OCR error
        if caption:
            return f"The image shows: {caption}"
        else:
            return "An image is present."

def get_all_json_files(source_dirs, cache_path):
    """
    Scans directories for all .json files. Uses a cache file to speed up
    subsequent runs.
    """
    if os.path.exists(cache_path):
        logging.info(f"Loading file list from cache: {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not read cache file {cache_path}: {e}. Rescanning.")

    logging.info("Scanning all source directories for .json files... (This may take a while)")
    
    source_file_lists = {}
    total_files = 0
    
    for data_dir in source_dirs:
        logging.info(f"Scanning directory: {data_dir}")
        
        if not os.path.isdir(data_dir):
            logging.warning(f"Directory not found: {data_dir}. Skipping.")
            continue
        
        json_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        
        if not json_files:
            logging.warning(f"No .json files found in {data_dir}. Skipping.")
            continue
            
        source_file_lists[data_dir] = json_files
        total_files += len(json_files)
        logging.info(f"Found {len(json_files)} .json files in {data_dir}.")
    
    if total_files == 0:
        logging.error("No .json files found in any source directory. Aborting.")
        return None

    logging.info(f"Found {total_files} total processable files across {len(source_file_lists)} sources.")
    
    # Save to cache
    try:
        logging.info(f"Saving file list cache to {cache_path}...")
        with open(cache_path, 'w') as f:
            json.dump(source_file_lists, f)
    except Exception as e:
        logging.warning(f"Could not write cache file: {e}")

    return source_file_lists

def main():
    setup_logging()
    
    logging.info("Loading EasyOCR model...")
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        if torch.cuda.is_available():
            logging.info("EasyOCR model loaded successfully on GPU.")
        else:
            logging.info("EasyOCR model loaded successfully on CPU. (Note: OCR will be slow)")
    except Exception as e:
        logging.error(f"Failed to load EasyOCR: {e}. Check CUDA/PyTorch setup.", exc_info=True)
        return

    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    try:
        # --- Proportional Sampling Logic (from extracted files) ---
        source_file_lists = get_all_json_files(DATA_SOURCE_DIRS, FILE_LIST_CACHE)
        
        if not source_file_lists:
            return # Error already logged by get_all_json_files

        total_files = sum(len(v) for v in source_file_lists.values())
        logging.info(f"Creating a balanced dataset of {TOTAL_DATASET_SIZE} total samples from {total_files} files.")

        master_json_list = []
        for dir_name, file_list in source_file_lists.items():
            source_count = len(file_list)
            ratio = source_count / total_files
            num_samples_for_source = int(ratio * TOTAL_DATASET_SIZE)
            
            logging.info(f"Taking {num_samples_for_source} random samples from {dir_name} (Ratio: {ratio:.2%})")
            
            random.shuffle(file_list)
            
            for file_path in file_list[:num_samples_for_source]:
                master_json_list.append(file_path)

        random.shuffle(master_json_list)
        
        if len(master_json_list) > TOTAL_DATASET_SIZE:
            master_json_list = master_json_list[:TOTAL_DATASET_SIZE]

        logging.info(f"Total files to process after sampling: {len(master_json_list)}")

        # --- Start Parallel Processing ---
        processed_data = []
        session = create_download_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
        
        download_jobs = []
        
        logging.info("Stage 1: Reading JSON files and submitting download jobs...")
        for file_path in tqdm(master_json_list, desc="1. Reading JSON"):
            data_item = process_json_file(file_path)
            if data_item:
                future = executor.submit(download_image, data_item, session)
                download_jobs.append(future)

        logging.info(f"Submitted {len(download_jobs)} image download jobs.")
        logging.info("Stage 2: Downloading and processing images (OCR)...")
        
        for future in tqdm(concurrent.futures.as_completed(download_jobs), total=len(download_jobs), desc="2. Downloading/OCR"):
            try:
                data_with_image = future.result()
                
                if not data_with_image:
                    continue
                
                image_description = get_image_description(
                    data_with_image['local_image_path'], 
                    data_with_image['image_caption'], 
                    ocr_reader
                )
                
                final_summary = f"{image_description} {data_with_image['base_summary']}"
                
                processed_data.append({
                    "article_text": data_with_image['article_text'],
                    "local_image_path": data_with_image['local_image_path'],
                    "final_summary": final_summary
                })
            except Exception as e:
                logging.error(f"Error in processing future: {e}", exc_info=True)

        executor.shutdown()
        session.close()
            
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("Script finished or was interrupted.")

    # --- Saving Logic ---
    logging.info(f"Successfully processed {len(processed_data)} items.")

    if not processed_data:
        logging.warning("No data was processed. Exiting without saving.")
        return

    logging.info(f"Converting to DataFrame and saving to {OUTPUT_FILENAME}...")
    try:
        df = pd.DataFrame(processed_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, OUTPUT_FILENAME)
        
        logging.info(f"--- {len(processed_data)} Balanced Dataset Build Complete! ---")
        logging.info(f"Final dataset saved to {OUTPUT_FILENAME}")
        
    except Exception as e:
        logging.error(f"Failed to save Parquet file: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("--- Process Interrupted by User ---")