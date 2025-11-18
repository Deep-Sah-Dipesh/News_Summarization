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
    # Add your other extracted sources here if you wish
    # os.path.join(DATA_PARENT_DIR, r"guardian_articles\guardian_json"),
    # os.path.join(DATA_PARENT_DIR, r"washington_post_articles\json"), 
]

# 2. SET YOUR OUTPUT PARENT DIRECTORY
OUTPUT_PARENT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\balanced_50k_dataset_sets"

# 3. SET THE SIZE FOR EACH SET
SET_SIZE = 50000

# 4. PERFORMANCE TUNING
DOWNLOAD_WORKERS = 32
REQUEST_TIMEOUT = 5 # (in seconds)
RANDOM_SEED = 42 # Ensures shuffles are deterministic
# ---------------------

# The cache will be stored in the parent directory
FILE_LIST_CACHE = os.path.join(OUTPUT_PARENT_DIR, "_file_scan_cache.json")
# ---------------------

def setup_logging(log_dir, set_number):
    """Configures logging to file and console for a specific set."""
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILENAME = os.path.join(log_dir, f"data_build_log_set{set_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Remove all previous handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILENAME, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) 
        ]
    )
    logging.info(f"--- Starting Multimodal Set {set_number} Build ---")
    logging.info(f"Source Dirs: {DATA_SOURCE_DIRS}")
    logging.info(f"Output Dir: {log_dir}")
    logging.info(f"Target Set Size: {SET_SIZE}")
    logging.info(f"Using {DOWNLOAD_WORKERS} parallel download workers.")

def create_download_session():
    """Creates a robust requests.Session with retries and backoff."""
    session = requests.Session()
    # Retry 3 times with a backoff factor (e.g., 0.5s, 1s, 2s)
    retries = Retry(
        total=3, 
        backoff_factor=0.5, 
        status_forcelist=[500, 502, 503, 504] # Retry on server errors
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
        
        # Filter for better data quality
        if not article_text or len(article_text.split()) < 50:
            return None # Skip short/empty articles
        if not base_summary or len(base_summary.split()) < 5:
            return None # Skip short/empty summaries
        
        relations = data.get("relations")
        if not relations or not isinstance(relations, list) or len(relations) == 0:
            return None
            
        image_content = relations[0].get("content")
        if not image_content:
            return None
            
        image_url = image_content.get("href")
        image_caption = image_content.get("caption")

        if not all([image_url, image_caption]):
            return None
            
        return {
            "article_text": article_text,
            "base_summary": base_summary,
            "image_url": image_url,
            "image_caption": image_caption,
            "original_json_path": file_path
        }
    except json.JSONDecodeError:
        # logging.warning(f"Skipping corrupt/empty JSON: {file_path}") # Too noisy
        return None
    except Exception as e:
        logging.warning(f"Error parsing JSON {file_path}: {e}")
        return None

def download_image(data_item, session, image_save_dir):
    """
    Downloads an image. Returns the data_item with local_path on success,
    or None on failure. Does not log every warning to avoid flooding.
    """
    if not data_item:
        return None

    url = data_item['image_url']
    
    base_json_name = os.path.splitext(os.path.basename(data_item['original_json_path']))[0]
    url_ext = os.path.splitext(os.path.basename(url).split('?')[0])[1]
    if url_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        url_ext = '.jpg'
    
    image_filename = f"{base_json_name}_img0{url_ext}"
    local_image_path = os.path.join(image_save_dir, image_filename)
    
    try:
        if os.path.exists(local_image_path):
            data_item['local_image_path'] = local_image_path
            return data_item
            
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            # logging.warning(f"Skipped non-image URL: {url}") # Too noisy
            return None

        with open(local_image_path, 'wb') as f:
            f.write(response.content)
            
        data_item['local_image_path'] = local_image_path
        return data_item
        
    except requests.exceptions.RequestException:
        # logging.warning(f"Failed to download image {url}: {e}") # Too noisy
        return None

def get_image_description(image_path, caption, ocr_reader):
    """Performs 'Image Triage' (OCR vs. Caption)."""
    try:
        ocr_result = ocr_reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result)
        
        if len(ocr_text) > 50:
            return f"The image is a document containing the text: {ocr_text}"
        elif len(ocr_text) > 5 and caption:
            return f"The image shows: {caption}. It contains the text: {ocr_text}"
        else:
            if caption:
                return f"The image shows: {caption}"
            else:
                return "An image is present."
            
    except Exception as e:
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
    
    try:
        logging.info(f"Saving file list cache to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(source_file_lists, f)
    except Exception as e:
        logging.warning(f"Could not write cache file: {e}")

    return source_file_lists

def main():
    
    # --- Load OCR Model Once ---
    print("Loading EasyOCR model...")
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        if torch.cuda.is_available():
            print("EasyOCR model loaded successfully on GPU.")
        else:
            print("EasyOCR model loaded successfully on CPU. (Note: OCR will be slow)")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load EasyOCR: {e}. Check CUDA/PyTorch setup.")
        return

    # --- Scan for all files ONCE ---
    source_file_lists = get_all_json_files(DATA_SOURCE_DIRS, FILE_LIST_CACHE)
    
    if not source_file_lists:
        return # Error already logged

    # --- Create one master list and shuffle it deterministically ---
    master_json_list = []
    for dir_name, file_list in source_file_lists.items():
        master_json_list.extend(file_list)
    
    total_files = len(master_json_list)
    print(f"Loaded {total_files} total file paths from cache/scan.")

    # Deterministic shuffle ensures sets are distinct and reproducible
    random.seed(RANDOM_SEED)
    random.shuffle(master_json_list)

    # --- Pre-filter the *entire* list to find valid items ---
    print(f"Pre-filtering all {total_files} files to find valid candidates...")
    valid_candidates = []
    for file_path in tqdm(master_json_list, desc="Filtering JSONs"):
        data_item = process_json_file(file_path)
        if data_item:
            valid_candidates.append(data_item)
    
    total_valid = len(valid_candidates)
    total_possible_sets = total_valid // SET_SIZE
    
    print(f"Found {total_valid} valid candidates after filtering.")
    if total_possible_sets == 0:
        print(f"ERROR: Not enough data ({total_valid}) to create even one set of {SET_SIZE} files.")
        return
        
    print(f"This is enough data to create {total_possible_sets} set(s) of {SET_SIZE} items.")

    # --- Main Loop to create all sets ---
    for set_number in range(1, total_possible_sets + 1):
        
        # --- Create Set-Specific Paths ---
        SET_OUTPUT_DIR = os.path.join(OUTPUT_PARENT_DIR, f"set_{set_number}")
        IMAGE_SAVE_DIR = os.path.join(SET_OUTPUT_DIR, "images")
        OUTPUT_FILENAME = os.path.join(SET_OUTPUT_DIR, f"multimodal_dataset_set{set_number}.parquet")
        
        setup_logging(SET_OUTPUT_DIR, set_number)
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

        try:
            # Calculate the slice for the requested set
            start_index = (set_number - 1) * SET_SIZE
            end_index = set_number * SET_SIZE
            
            items_to_process = valid_candidates[start_index:end_index]
            
            logging.info(f"Processing set {set_number}: items {start_index} to {end_index}.")
            logging.info(f"Total items to process for this set: {len(items_to_process)}")

            # --- Start Parallel Processing ---
            processed_data = []
            session = create_download_session()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
            
            download_jobs = []
            
            logging.info("Stage 1: Submitting all files to download queue...")
            for data_item in tqdm(items_to_process, desc="1. Submitting Jobs"):
                future = executor.submit(download_image, data_item, session, IMAGE_SAVE_DIR)
                download_jobs.append(future)

            logging.info(f"Submitted {len(download_jobs)} image download jobs.")
            logging.info("Stage 2: Downloading and processing images (OCR)...")
            
            # Statistics for final report
            download_success_count = 0
            
            for future in tqdm(concurrent.futures.as_completed(download_jobs), total=len(download_jobs), desc="2. Downloading/OCR"):
                try:
                    data_with_image = future.result()
                    
                    if not data_with_image:
                        continue # Download failed
                    
                    download_success_count += 1
                    
                    image_description = get_image_description(
                        data_with_image['local_image_path'], 
                        data_with_image['image_caption'], 
                        ocr_reader
                    )
                    
                    # Prepend image text to article, keep summary clean
                    final_summary = data_with_image['base_summary']
                    final_article_text = f"Image context: {image_description}. Article: {data_with_image['article_text']}"
                    
                    processed_data.append({
                        "article_text": final_article_text,
                        "local_image_path": data_with_image['local_image_path'],
                        "final_summary": final_summary
                    })
                except Exception as e:
                    logging.error(f"Error in processing future: {e}", exc_info=True)

            executor.shutdown()
            session.close()
            
            # --- Final Summary Report ---
            logging.info(f"--- Set {set_number} Build Summary ---")
            logging.info(f"Total JSONs processed: {len(items_to_process)}")
            logging.info(f"Image Downloads Succeeded: {download_success_count}")
            logging.info(f"Image Downloads Failed (404s, Timeouts, etc.): {len(items_to_process) - download_success_count}")
            logging.info(f"Final items saved: {len(processed_data)}")
            
        except Exception as e:
            logging.error(f"An unexpected error occurred during processing of set {set_number}: {e}", exc_info=True)
        finally:
            logging.info(f"Finished processing set {set_number}.")

        # --- Saving Logic ---
        if not processed_data:
            logging.warning("No data was processed for this set. Skipping save.")
            continue # Go to the next set in the loop

        logging.info(f"Converting to DataFrame and saving to {OUTPUT_FILENAME}...")
        try:
            df = pd.DataFrame(processed_data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, OUTPUT_FILENAME)
            
            logging.info(f"--- Set {set_number} Build Complete! ({len(processed_data)} items) ---")
            logging.info(f"Final dataset saved to {OUTPUT_FILENAME}")
            
        except Exception as e:
            logging.error(f"Failed to save Parquet file for set {set_number}: {e}", exc_info=True)

    logging.info("--- All dataset sets built. ---")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("--- Process Interrupted by User ---")