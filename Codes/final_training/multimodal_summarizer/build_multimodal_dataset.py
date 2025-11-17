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

# --- FIX FOR OMP: Error #15 ---
# This must be at the top before importing torch/easyocr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------

# --- Configuration ---

# 1. SET YOUR DATA SOURCE(S) HERE
#    Run the pilot on just one directory first.
DATA_SOURCE_DIRS = [
    r"C:\Users\admin\Downloads\articles\articles\bbc_2\bbcnews_stm2json"
    # Add other dirs later, e.g.:
    # r"H:\News_Summarization\Dataset\News_Articles_with_Images\bbc_1",
    # r"H:\News_Summarization\Dataset\News_Articles_with_Images\guardian_articles",
]

# 2. SET YOUR OUTPUT DIRECTORY
#    All outputs (logs, images, parquet) will be saved here.
OUTPUT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\bbc2news_processed"

# 3. SET MAX FILES FOR PILOT RUN
MAX_FILES_TO_PROCESS = 25000

# --- End Configuration ---

# --- Derived Paths ---
IMAGE_SAVE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, f"multimodal_dataset_generation_{datetime.now().strftime('%Y%m%d')}.parquet")
LOG_FILENAME = os.path.join(OUTPUT_DIR, f"data_build_log_pilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    logging.info("--- Starting Multimodal Dataset Build ---")
    logging.info(f"Source Dirs: {DATA_SOURCE_DIRS}")
    logging.info(f"Output Dir: {OUTPUT_DIR}")
    logging.info(f"Max Files: {MAX_FILES_TO_PROCESS or 'All'}")

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

        # Skip if any essential field is missing
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
        # Avoid re-downloading
        if os.path.exists(save_path):
            return True
            
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Verify it's an image before saving
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            logging.warning(f"Skipped non-image URL: {url} (Content-Type: {content_type})")
            return False

        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
        
    except requests.exceptions.RequestException as e:
        logging.warning(f"Failed to download image {url}: {e}")
        return False

def get_image_description(image_path, caption, ocr_reader):
    """
    Performs 'Image Triage' (OCR vs. Caption).
    Uses OCR if text is found, otherwise defaults to the caption.
    """
    try:
        # 1. Run OCR
        # detail=0 and paragraph=True are optimizations for speed
        ocr_result = ocr_reader.readtext(image_path, detail=0, paragraph=True)
        ocr_text = " ".join(ocr_result)
        
        # 2. Triage Logic
        # If OCR finds more than 50 characters, treat it as a document.
        if len(ocr_text) > 50:
            return f"The image is a document containing the text: {ocr_text}"
        else:
            # Otherwise, use the caption.
            if caption:
                return f"The image shows: {caption}"
            else:
                # Fallback if caption is also empty
                return "An image is present."
    except Exception as e:
        # Fallback in case of corrupt image or OCR error
        logging.error(f"Error during OCR/Triage for {image_path}: {e}")
        if caption:
            return f"The image shows: {caption}"
        else:
            return "An image is present."

def main():
    setup_logging()
    
    logging.info("Loading EasyOCR model... (This may take a moment on first run)")
    try:
        # Load OCR model, use GPU if available
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        logging.info("EasyOCR model loaded.")
    except Exception as e:
        logging.error(f"Failed to load EasyOCR: {e}. Check CUDA/PyTorch setup.", exc_info=True)
        return

    # Ensure the main image save directory exists
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    logging.info("Scanning for .json files...")
    json_files = []
    for data_dir in DATA_SOURCE_DIRS:
        logging.info(f"Scanning directory: {data_dir}")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
                
    logging.info(f"Found {len(json_files)} total .json files.")
    
    if MAX_FILES_TO_PROCESS:
        logging.warning(f"Processing a subset of {MAX_FILES_TO_PROCESS} files.")
        json_files = json_files[:MAX_FILES_TO_PROCESS]
        
    processed_data = []
    
    logging.info("Starting processing loop...")
    for file_path in tqdm(json_files, desc="Processing Files"):
        data = process_json_file(file_path)
        if not data:
            continue
            
        image_filename = os.path.basename(data['image_url'])
        image_filename = "".join(c for c in image_filename if c.isalnum() or c in ('.', '_', '-')).strip()
        if not image_filename:
             # Fallback for weird URLs
             image_filename = f"{os.path.basename(data['original_json_path'])}.jpg"
             
        local_image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
        
        # Try to download the image
        if not download_image(data['image_url'], local_image_path):
            continue
        
        # Run our triage logic
        image_description = get_image_description(
            local_image_path, 
            data['image_caption'], 
            ocr_reader
        )
        
        # Create the final target summary
        final_summary = f"{image_description} {data['base_summary']}"
        
        processed_data.append({
            "article_text": data['article_text'],
            "local_image_path": local_image_path,
            "final_summary": final_summary
        })
        
        # Sleep to avoid rate-limiting on image servers (optional)
        # time.sleep(0.01) 

    logging.info(f"Successfully processed {len(processed_data)} items.")

    if not processed_data:
        logging.warning("No data was processed. Exiting without saving.")
        return

    logging.info(f"Converting to DataFrame and saving to {OUTPUT_FILENAME}...")
    try:
        df = pd.DataFrame(processed_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, OUTPUT_FILENAME)
        
        logging.info(f"--- Dataset Build Complete! ---")
        logging.info(f"Final dataset saved to {OUTPUT_FILENAME}")
        logging.info("You can now inspect this file to verify the data.")
        
    except Exception as e:
        logging.error(f"Failed to save Parquet file: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("--- Process Interrupted by User ---")