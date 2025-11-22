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
from deep_translator import GoogleTranslator # NEW IMPORT

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

# 2. SET YOUR OUTPUT PARENT DIRECTORY
OUTPUT_PARENT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\balanced_50k_dataset_sets"

# 3. SET THE SIZE FOR EACH SET
SET_SIZE = 50000

# 4. PERFORMANCE TUNING
DOWNLOAD_WORKERS = 32
REQUEST_TIMEOUT = 5 
RANDOM_SEED = 42

SKIP_DOMAINS = [
    "newsimg.bbc.co.uk",
    "c.files.bbci.co.uk",
    "news.bbcimg.co.uk"
]

FILE_LIST_CACHE = os.path.join(OUTPUT_PARENT_DIR, "_file_scan_cache.json")

# --- Translator ---
translator = GoogleTranslator(source='auto', target='hi')

def setup_logging(log_dir, set_number):
    """Configures logging."""
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILENAME = os.path.join(log_dir, f"data_build_log_set{set_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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
    logging.info(f"--- Starting Multimodal Set {set_number} Build (With Translation) ---")

def create_download_session():
    """Creates a robust requests.Session."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=DOWNLOAD_WORKERS, pool_maxsize=DOWNLOAD_WORKERS)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session

def process_json_file(file_path):
    """Extracts key data from JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        
        article_text = data.get("body")
        base_summary = data.get("summary")
        
        if not article_text or len(article_text.split()) < 50: return None
        if not base_summary or len(base_summary.split()) < 5: return None
        
        relations = data.get("relations")
        if not relations or not isinstance(relations, list) or len(relations) == 0: return None
            
        image_content = relations[0].get("content")
        if not image_content: return None
            
        image_url = image_content.get("href")
        image_caption = image_content.get("caption")

        if not all([image_url, image_caption]): return None
            
        return {
            "article_text": article_text,
            "base_summary": base_summary,
            "image_url": image_url,
            "image_caption": image_caption,
            "original_json_path": file_path
        }
    except (json.JSONDecodeError, Exception):
        return None

def translate_to_hindi(text):
    """Translates text to Hindi using deep_translator."""
    try:
        # Basic cleanup to avoid translation errors
        text = text.strip()
        if not text: return ""
        # API limits often exist; consider truncating very long summaries if needed
        return translator.translate(text)
    except Exception as e:
        logging.warning(f"Translation failed: {e}")
        return text # Fallback to English if translation fails

def download_image(data_item, session, image_save_dir):
    """Downloads image."""
    if not data_item: return None

    url = data_item['image_url']
    if any(d in url for d in SKIP_DOMAINS): return None
    
    base_json_name = os.path.splitext(os.path.basename(data_item['original_json_path']))[0]
    url_ext = os.path.splitext(os.path.basename(url).split('?')[0])[1]
    if url_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']: url_ext = '.jpg'
    
    image_filename = f"{base_json_name}_img0{url_ext}"
    local_image_path = os.path.join(image_save_dir, image_filename)
    
    try:
        if os.path.exists(local_image_path):
            data_item['local_image_path'] = local_image_path
            return data_item
            
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        if 'image' not in response.headers.get('Content-Type', ''): return None

        with open(local_image_path, 'wb') as f:
            f.write(response.content)
            
        data_item['local_image_path'] = local_image_path
        return data_item
    except requests.exceptions.RequestException:
        return None

def get_image_description(image_path, caption, ocr_reader):
    """Performs 'Image Triage'."""
    try:
        ocr_result = ocr_reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result)
        
        if len(ocr_text) > 50:
            return f"The image is a document containing the text: {ocr_text}"
        elif len(ocr_text) > 5 and caption:
            return f"The image shows: {caption}. It contains the text: {ocr_text}"
        else:
            return f"The image shows: {caption}" if caption else "An image is present."
    except Exception:
        return f"The image shows: {caption}" if caption else "An image is present."

def get_all_json_files(source_dirs, cache_path):
    """Scans directories for JSON files."""
    if os.path.exists(cache_path):
        logging.info(f"Loading file list from cache: {cache_path}")
        try:
            with open(cache_path, 'r') as f: return json.load(f)
        except Exception: pass

    logging.info("Scanning source directories...")
    source_file_lists = {}
    for data_dir in source_dirs:
        json_files = []
        if os.path.isdir(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".json"):
                        json_files.append(os.path.join(root, file))
            source_file_lists[data_dir] = json_files
            
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f: json.dump(source_file_lists, f)
    except Exception: pass
    return source_file_lists

def main():
    print("Loading EasyOCR model...")
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load EasyOCR: {e}")
        return

    source_file_lists = get_all_json_files(DATA_SOURCE_DIRS, FILE_LIST_CACHE)
    if not source_file_lists: return

    master_json_list = []
    for file_list in source_file_lists.values():
        master_json_list.extend(file_list)
    
    random.seed(RANDOM_SEED)
    random.shuffle(master_json_list)
    
    # Filter valid candidates
    print("Filtering valid files...")
    valid_candidates = []
    for fp in tqdm(master_json_list):
        item = process_json_file(fp)
        if item: valid_candidates.append(item)
    
    total_possible_sets = len(valid_candidates) // SET_SIZE
    print(f"Found {len(valid_candidates)} valid items. Creating {total_possible_sets} sets.")

    for set_number in range(1, total_possible_sets + 1):
        SET_OUTPUT_DIR = os.path.join(OUTPUT_PARENT_DIR, f"set_{set_number}")
        IMAGE_SAVE_DIR = os.path.join(SET_OUTPUT_DIR, "images")
        OUTPUT_FILENAME = os.path.join(SET_OUTPUT_DIR, f"multimodal_dataset_set{set_number}.parquet")
        
        setup_logging(SET_OUTPUT_DIR, set_number)
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        
        start_idx = (set_number - 1) * SET_SIZE
        end_idx = set_number * SET_SIZE
        items = valid_candidates[start_idx:end_idx]
        
        processed_data = []
        session = create_download_session()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
        download_jobs = []
        
        logging.info("Submitting download jobs...")
        for item in items:
            download_jobs.append(executor.submit(download_image, item, session, IMAGE_SAVE_DIR))
            
        logging.info("Processing results (OCR + Translation)...")
        for future in tqdm(concurrent.futures.as_completed(download_jobs), total=len(download_jobs)):
            try:
                res = future.result()
                if not res: continue
                
                desc = get_image_description(res['local_image_path'], res['image_caption'], ocr_reader)
                
                # --- TRANSLATION STEP ---
                hindi_summary = translate_to_hindi(res['base_summary'])
                
                processed_data.append({
                    "article_text": f"Image context: {desc}. Article: {res['article_text']}",
                    "local_image_path": res['local_image_path'],
                    "final_summary": hindi_summary # Now in Hindi!
                })
            except Exception as e:
                logging.error(f"Error processing item: {e}")
                
        executor.shutdown()
        session.close()
        
        if processed_data:
            df = pd.DataFrame(processed_data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, OUTPUT_FILENAME)
            logging.info(f"Saved {len(processed_data)} items to {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()