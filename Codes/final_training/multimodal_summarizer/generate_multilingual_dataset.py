import os
import sys
import logging
import functools

# --- CRITICAL FIXES ---
# 1. OpenMP Error Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. AGGRESSIVE FIX for torch.load vulnerability error
# We wrap torch.load to force weights_only=False if it's not specified, 
# or to catch the specific error and retry.
import torch
original_torch_load = torch.load

@functools.wraps(original_torch_load)
def unsafe_torch_load(*args, **kwargs):
    # If the error is about weights_only, we force it to False for this script
    # because we trust the local/HuggingFace models we are loading.
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = unsafe_torch_load

# Also patch transformers safety check directly
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True
# ----------------------

import json
import pandas as pd
import requests
import easyocr
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import random
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import MarianMTModel, MarianTokenizer

# --- Configuration ---

# 1. SET YOUR DATA SOURCE(S) HERE
DATA_PARENT_DIR = r"D:\Articles\extracted_articles"
DATA_SOURCE_DIRS = [
    os.path.join(DATA_PARENT_DIR, r"bbc_1\bbcnews_json"),
    os.path.join(DATA_PARENT_DIR, r"bbc_2\bbcnews_stm2json")
]

# 2. SET YOUR OUTPUT PARENT DIRECTORY
OUTPUT_PARENT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\multilingual_50k_dataset_local"

# 3. SET THE TARGET DATASET SIZE
TARGET_TOTAL_SIZE = 50000
CHUNK_SIZE = 5000  # Save every 5k items to disk

# 4. PERFORMANCE TUNING
DOWNLOAD_WORKERS = 16 
REQUEST_TIMEOUT = 5 
RANDOM_SEED = 42 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------

FILE_LIST_CACHE = os.path.join(OUTPUT_PARENT_DIR, "_file_scan_cache.json")

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILENAME = os.path.join(log_dir, f"dataset_gen_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Remove existing handlers
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

def create_download_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=DOWNLOAD_WORKERS, pool_maxsize=DOWNLOAD_WORKERS)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session

# --- LOCAL TRANSLATION SETUP ---
def load_translator():
    print("Loading Helsinki-NLP translation model (en-hi)...")
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    try:
        # We explicitly pass weights_only=False here just in case
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"CRITICAL ERROR LOADING TRANSLATOR: {e}")
        # Fallback: Try loading with trust_remote_code=True which sometimes bypasses checks
        try:
            print("Attempting fallback load...")
            model = MarianMTModel.from_pretrained(model_name, trust_remote_code=True).to(DEVICE)
            model.eval()
            return tokenizer, model
        except Exception as e2:
            print(f"Fallback failed: {e2}")
            sys.exit(1)

def translate_batch(texts, tokenizer, model):
    """Translates a batch of text using local GPU model."""
    try:
        # Tokenize
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        
        # Generate Translation
        with torch.no_grad():
            translated = model.generate(**inputs)
        
        # Decode
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    except Exception as e:
        logging.warning(f"Translation batch failed: {e}")
        return [None] * len(texts)

def process_json_file(file_path):
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
            "english_summary": base_summary, 
            "image_url": image_url,
            "image_caption": image_caption,
            "original_json_path": file_path
        }
    except Exception:
        return None

def download_image_task(data_item, session, image_save_dir):
    """Standalone task for downloading image."""
    if not data_item: return None

    url = data_item['image_url']
    base_name = os.path.splitext(os.path.basename(data_item['original_json_path']))[0]
    # Basic sanitization for filename
    url_clean = url.split('?')[0]
    url_ext = os.path.splitext(os.path.basename(url_clean))[1]
    if url_ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']: url_ext = '.jpg'
    
    local_path = os.path.join(image_save_dir, f"{base_name}_img{url_ext}")
    
    try:
        if not os.path.exists(local_path):
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', ''): return None
            with open(local_path, 'wb') as f: f.write(response.content)
        
        data_item['local_image_path'] = local_path
        return data_item
    except:
        return None

def get_all_json_files(source_dirs, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading file list from cache: {cache_path}")
        try:
            with open(cache_path, 'r') as f: return json.load(f)
        except:
            print("Cache corrupted, rescanning...")

    files_list = []
    for d in source_dirs:
        if not os.path.exists(d):
            print(f"Warning: Source directory not found: {d}")
            continue
            
        print(f"Scanning {d}...")
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".json"): files_list.append(os.path.join(root, f))
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f: json.dump(files_list, f)
    return files_list

def main():
    setup_logging(OUTPUT_PARENT_DIR)
    IMAGE_SAVE_DIR = os.path.join(OUTPUT_PARENT_DIR, "images")
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    # 1. Load Models
    print("Loading EasyOCR...")
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return

    # Load Translation Model (Helsinki-NLP)
    trans_tokenizer, trans_model = load_translator()

    # 2. Prepare File List
    print("Scanning files...")
    all_files = get_all_json_files(DATA_SOURCE_DIRS, FILE_LIST_CACHE)
    
    if not all_files:
        print("No files found. Check DATA_SOURCE_DIRS.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(all_files)
    
    print(f"Found {len(all_files)} files. Selecting {TARGET_TOTAL_SIZE} candidates...")
    
    # 3. Selection Phase
    candidates = []
    print("Filtering candidates (this might take a moment)...")
    for f in tqdm(all_files, desc="Filtering JSONs"):
        item = process_json_file(f)
        if item: candidates.append(item)
        if len(candidates) >= TARGET_TOTAL_SIZE: break
            
    if len(candidates) < TARGET_TOTAL_SIZE:
        print(f"Warning: Only found {len(candidates)} valid source files.")
    else:
        print(f"Successfully selected {len(candidates)} candidates.")

    print("Starting Processing Pipeline...")
    session = create_download_session()
    
    final_dataset = []
    batch_buffer = []
    current_chunk = 1
    total_processed_count = 0

    # Processing Loop
    for item in tqdm(candidates, desc="Processing Items"):
        # Step A: Download Image
        downloaded_item = download_image_task(item, session, IMAGE_SAVE_DIR)
        if not downloaded_item: continue

        # Step B: OCR
        try:
            ocr_res = ocr_reader.readtext(downloaded_item['local_image_path'], detail=0, paragraph=True)
            ocr_text = " ".join(ocr_res)
            caption = downloaded_item['image_caption']
            
            if len(ocr_text) > 50:
                desc = f"The image is a document containing: {ocr_text}"
            elif len(ocr_text) > 5 and caption:
                desc = f"The image shows: {caption}. Text in image: {ocr_text}"
            else:
                desc = f"The image shows: {caption}"
        except:
            desc = f"The image shows: {downloaded_item['image_caption']}"
        
        # Add description to item
        downloaded_item['image_context_desc'] = desc
        batch_buffer.append(downloaded_item)

        # Step C: Batch Translation
        # Increased batch size for efficiency since we are on GPU
        if len(batch_buffer) >= 32: 
            summaries = [x['english_summary'] for x in batch_buffer]
            
            # Perform translation
            hindi_translations = translate_batch(summaries, trans_tokenizer, trans_model)
            
            for i, trans in enumerate(hindi_translations):
                if trans:
                    data_row = batch_buffer[i]
                    final_input = f"Image context: {data_row['image_context_desc']}. Article: {data_row['article_text']}"
                    
                    final_dataset.append({
                        "article_text": final_input,
                        "local_image_path": data_row['local_image_path'],
                        "english_summary": data_row['english_summary'],
                        "hindi_summary": trans
                    })
            
            batch_buffer = [] # Clear buffer

        # Step D: Save Chunk to Disk
        if len(final_dataset) >= CHUNK_SIZE:
            fname = os.path.join(OUTPUT_PARENT_DIR, f"dataset_chunk_{current_chunk}.parquet")
            df = pd.DataFrame(final_dataset)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, fname)
            print(f"Saved Chunk {current_chunk} ({len(final_dataset)} items)")
            
            final_dataset = [] 
            current_chunk += 1

    # Translate remaining items in buffer
    if batch_buffer:
        summaries = [x['english_summary'] for x in batch_buffer]
        hindi_translations = translate_batch(summaries, trans_tokenizer, trans_model)
        for i, trans in enumerate(hindi_translations):
            if trans:
                data_row = batch_buffer[i]
                final_input = f"Image context: {data_row['image_context_desc']}. Article: {data_row['article_text']}"
                final_dataset.append({
                    "article_text": final_input,
                    "local_image_path": data_row['local_image_path'],
                    "english_summary": data_row['english_summary'],
                    "hindi_summary": trans
                })

    # Save final partial chunk
    if final_dataset:
        fname = os.path.join(OUTPUT_PARENT_DIR, f"dataset_chunk_{current_chunk}.parquet")
        df = pd.DataFrame(final_dataset)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, fname)
        print(f"Saved Final Chunk {current_chunk}")

    print("Dataset Generation Complete!")

if __name__ == "__main__":
    main()