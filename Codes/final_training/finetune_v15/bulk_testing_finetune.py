import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import textwrap
import os
import logging
from tqdm import tqdm
from datetime import datetime
import sys 

import config_finetune as config
import data_utils
import logging_utils

# --- Configuration ---
DATA_PATH = [
    "../../Dataset/final_cleaned_dataset_CNN.csv",
    "../../Dataset/filtered_articles_CNN.csv"
]
NUM_SAMPLES = 50 # Set to None to test all samples
# ----------------------------------------------

model = None
tokenizer = None

def load_model():
    """Loads the final v15_fine_tuned model and tokenizer."""
    global model, tokenizer
    
    if not os.path.exists(config.FINAL_SAVE_PATH):
        logging.error(f"Model path not found: {config.FINAL_SAVE_PATH}")
        print(f"Error: Model path not found: {config.FINAL_SAVE_PATH}")
        print("Please run 'train_finetune.py' and 'evaluate_finetune.py' first.")
        return False

    logging.info(f"Loading final model from: {config.FINAL_SAVE_PATH}...")
    try:
        model = MBartForConditionalGeneration.from_pretrained(config.FINAL_SAVE_PATH).to(config.DEVICE)
        tokenizer = MBart50TokenizerFast.from_pretrained(config.FINAL_SAVE_PATH)
        model.eval()
        logging.info(f"Model v15_FINE_TUNED loaded successfully on {config.DEVICE}")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        return False

def load_test_data(num_samples=None):
    """Loads, cleans, and subsets the test data from the CSV."""
    logging.info(f"Loading test data from: {DATA_PATH}")
    
    data_paths = DATA_PATH
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    all_dfs = []
    logging.info(f"Loading test data from {len(data_paths)} file(s)...")
    for path in data_paths:
        try:
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
            all_dfs.append(df)
            logging.info(f"Loaded {len(df)} rows from {path}")
        except FileNotFoundError:
            logging.error(f"Test data file not found: {path}. Skipping.")
    
    if not all_dfs:
        logging.error("No test data loaded. Aborting.")
        print("Error: No valid test data files were loaded. Aborting.")
        return None
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Total test rows after combining: {len(combined_df)}")
        
    combined_df.dropna(subset=['raw_news_article', 'hindi_summary'], inplace=True)
    
    combined_df['raw_news_article'] = combined_df['raw_news_article'].apply(data_utils.sanitize_text).apply(data_utils.normalize_text)
    combined_df['hindi_summary'] = combined_df['hindi_summary'].apply(data_utils.sanitize_text).apply(data_utils.normalize_text)
    
    if num_samples and num_samples > 0:
        if num_samples > len(df):
             logging.warning(f"Requested {num_samples} samples, but only {len(df)} are available. Using all.")
             num_samples = len(df)
        
        logging.warning(f"Selecting a random subset of {num_samples} articles for testing.")
        df = combined_df.sample(n=num_samples, random_state=42) 
    else:
        logging.info(f"Processing all {len(combined_df)} articles found in the file.")
        df = combined_df
        
    return df

def run_inference_and_log(df):
    """Generates summaries for the dataframe, printing results in real-time."""
    generated_summaries = []
    
    gen_kwargs = {
        "num_beams": 12,
        "length_penalty": 2.0,
        "repetition_penalty": 2.5,
        "no_repeat_ngram_size": 3,
        "do_sample": False,
        "early_stopping": True,
        "min_length": 30,
        "max_length": 250,
    }

    logging.info(f"Starting inference for {len(df)} articles...")
    df = df.reset_index(drop=True) 
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing Articles"):
        article = row['raw_news_article']
        
        tokenizer.src_lang = "en_XX"
        inputs = tokenizer(
            article, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True, 
            padding=True
        ).to(config.DEVICE)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs.input_ids,
                forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"],
                **gen_kwargs
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(summary)
        
        log_output_console = (
            f"\n\n--- Article {index+1}/{len(df)} ---\n"
            f"\nSOURCE:\n{textwrap.fill(article, width=80)}\n"
            f"\nGENERATED HINDI (v15_fine_tuned):\n{textwrap.fill(summary, width=80)}\n"
            f"{'='*80}"
        )
        print("\n\n" + log_output_console)
        
        log_output_file = (
            f"\n\n--- Article {index+1}/{len(df)} (Original Index: {row.name}) ---\n"
            f"\nSOURCE:\n{textwrap.fill(article, width=80)}\n"
            f"\nGENERATED HINDI (v15_fine_tuned):\n{textwrap.fill(summary, width=80)}\n"
            f"\nREFERENCE HINDI:\n{textwrap.fill(row['hindi_summary'], width=80)}\n"
            f"{'='*80}"
        )
        logging.info(log_output_file)
        
    return generated_summaries

def save_results(df, generated_summaries):
    """Saves the final results to a CSV file."""
    df['generated_hindi_summary_finetuned'] = generated_summaries
    
    output_df = df[['raw_news_article', 'hindi_summary', 'generated_hindi_summary_finetuned']]
    
    output_filename = os.path.join(
        config.MODEL_OUTPUT_DIR, 
        f"bulk_test_results_finetuned_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logging.info(f"Successfully saved full results to: {output_filename}")
    print(f"\nSuccessfully saved full results to: {output_filename}")

def main():
    log_path = logging_utils.setup_logging(config.MODEL_OUTPUT_DIR, "bulk_test_finetune_log_v15")
    print(f"--- Starting FINE-TUNED Bulk Test Script ---")
    print(f"Logging all output to: {log_path}\n")
    logging.info("--- Starting FINE-TUNED Bulk Test Script ---")
    
    if load_model():
        test_data = load_test_data(NUM_SAMPLES)
        if test_data is not None:
            generated_summaries = run_inference_and_log(test_data)
            save_results(test_data, generated_summaries)
            logging.info("--- FINE-TUNED Bulk test finished successfully ---")
            print("--- FINE-TUNED Bulk test finished successfully ---")
        else:
            logging.error("Failed to load test data. Aborting.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("--- Bulk Test Interrupted by User ---")
        print("\n\n--- Bulk Test Interrupted by User ---")
        sys.exit(0)
    except Exception as e:
        logging.error("--- Bulk Test FAILED ---", exc_info=True)
        print(f"\n--- Bulk Test FAILED --- \nError: {e}\nSee log file for details.")
        sys.exit(1)