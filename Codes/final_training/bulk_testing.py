import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import textwrap
import os
import logging
from tqdm import tqdm
from datetime import datetime
import sys 

import config
import data_utils
import logging_utils

# --- Configuration ---
DATA_PATH = r"H:\News_Summarization\Dataset\filtered_articles_CNN.csv" 
NUM_SAMPLES = 10
# ----------------------------------------------

model = None
tokenizer = None

def load_model():
    """Loads the final v15 model and tokenizer."""
    global model, tokenizer
    
    if not os.path.exists(config.FINAL_SAVE_PATH):
        logging.error(f"Model path not found: {config.FINAL_SAVE_PATH}")
        print(f"Error: Model path not found: {config.FINAL_SAVE_PATH}")
        return False

    logging.info(f"Loading final model from: {config.FINAL_SAVE_PATH}...")
    try:
        model = MBartForConditionalGeneration.from_pretrained(config.FINAL_SAVE_PATH).to(config.DEVICE)
        tokenizer = MBart50TokenizerFast.from_pretrained(config.FINAL_SAVE_PATH)
        model.eval()
        logging.info(f"Model v15 loaded successfully on {config.DEVICE}")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        return False

def load_test_data(num_samples=None):
    """Loads, cleans, and subsets the test data from the CSV."""
    logging.info(f"Loading test data from: {DATA_PATH}")
    
    try:
        df = pd.read_csv(DATA_PATH, engine='python', on_bad_lines='skip')
    except FileNotFoundError:
        logging.error(f"Data file not found at: {DATA_PATH}")
        print(f"Error: Data file not found at: {DATA_PATH}")
        return None
        
    df.dropna(subset=['raw_news_article', 'hindi_summary'], inplace=True)
    
    # Clean the text
    df['raw_news_article'] = df['raw_news_article'].apply(data_utils.sanitize_text).apply(data_utils.normalize_text)
    df['hindi_summary'] = df['hindi_summary'].apply(data_utils.sanitize_text).apply(data_utils.normalize_text)
    
    if num_samples and num_samples > 0:
        if num_samples > len(df):
             logging.warning(f"Requested {num_samples} samples, but only {len(df)} are available. Using all articles.")
             num_samples = len(df)
        
        logging.warning(f"Selecting a random subset of {num_samples} articles for testing.")
        df = df.sample(n=num_samples, random_state=42) 
        
    else:
        logging.info(f"Processing all {len(df)} articles found in the file.")
        
    return df

def run_inference_and_log(df):
    """
    Generates summaries for the dataframe, printing results in real-time.
    Returns a list of generated summaries.
    """
    generated_summaries = []
    
    batch_size = 1 
    
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
            f"\nGENERATED HINDI:\n{textwrap.fill(summary, width=80)}\n"
            f"{'='*80}"
        )
        
        print("\n\n" + log_output_console)
        
        log_output_file = (
            f"\n\n--- Article {index+1}/{len(df)} (Original Index: {row.name}) ---\n"
            f"\nSOURCE:\n{textwrap.fill(article, width=80)}\n"
            f"\nGENERATED HINDI:\n{textwrap.fill(summary, width=80)}\n"
            f"\nREFERENCE HINDI:\n{textwrap.fill(row['hindi_summary'], width=80)}\n"
            f"{'='*80}"
        )
        logging.info(log_output_file)
        
    return generated_summaries

def save_results(df, generated_summaries):
    """Saves the final results to a CSV file."""
    df['generated_hindi_summary'] = generated_summaries
    
    output_df = df[['raw_news_article', 'hindi_summary', 'generated_hindi_summary']]
    
    output_filename = os.path.join(
        config.MODEL_OUTPUT_DIR, 
        f"bulk_test_results_interactive_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    
    output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    logging.info(f"Successfully saved full results to: {output_filename}")
    print(f"\nSuccessfully saved full results to: {output_filename}")

def main():
    log_path = logging_utils.setup_logging(config.MODEL_OUTPUT_DIR, "bulk_test_interactive_log_v15")
    print(f"--- Starting Bulk Test Script ---")
    print(f"Logging all output to: {log_path}\n")
    logging.info("--- Starting Bulk Test Script ---")
    
    if load_model():
        test_data = load_test_data(NUM_SAMPLES)
        if test_data is not None:
            generated_summaries = run_inference_and_log(test_data)
            save_results(test_data, generated_summaries)
            logging.info("--- Bulk test finished successfully ---")
            print("--- Bulk test finished successfully ---")
        else:
            logging.error("Failed to load test data. Aborting.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("--- Bulk Test Interrupted by User ---")
        print("\n\n--- Bulk Test Interrupted by User ---")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        logging.error("--- Bulk Test FAILED ---", exc_info=True)
        print(f"\n--- Bulk Test FAILED --- \nError: {e}\nSee log file for details.")
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)