import pandas as pd
import textwrap
import logging
import os
import sys

# --- FIX FOR OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -----------------------------

# --- Configuration ---
# This is the file you just created
FILE_TO_INSPECT = r"H:\News_Summarization\Dataset\News_Articles_with_Images\balanced_50k_dataset_sets\set_1\multimodal_dataset_set1.parquet"
# --- End Configuration ---

def setup_logging():
    """Configures basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def inspect_parquet_file(file_path):
    """Loads and prints a summary of the Parquet file."""
    try:
        logging.info(f"--- Inspecting {file_path} ---")
        
        df = pd.read_parquet(file_path)
        
        logging.info("--- DataFrame Info ---")
        print(df.info())
        
        logging.info("\n--- DataFrame Head (First 5 Rows) ---")
        print(df.head().to_string())
        
        logging.info("\n--- Detailed View of First Item ---")
        if not df.empty:
            first_item = df.iloc[0]
            
            print("\n[Article Text (first 500 chars)]:")
            print(textwrap.fill(first_item['article_text'][:500] + "...", 80))
            
            print("\n[Local Image Path]:")
            print(first_item['local_image_path'])
            
            print("\n[FINAL_SUMMARY (our new target)]:")
            print(textwrap.fill(first_item['final_summary'], 80))
            
        else:
            logging.warning("DataFrame is empty. No items to display.")
            
        logging.info("\n--- Inspection Complete ---")

    except FileNotFoundError:
        logging.error(f"FATAL ERROR: The file '{file_path}' was not found.")
    except Exception as e:
        logging.error(f"An error occurred during inspection: {e}", exc_info=True)

if __name__ == "__main__":
    setup_logging()
    inspect_parquet_file(FILE_TO_INSPECT)