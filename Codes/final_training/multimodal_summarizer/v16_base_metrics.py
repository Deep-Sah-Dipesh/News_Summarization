import torch
import pandas as pd
import logging
import os
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast, ViTImageProcessor
from PIL import Image
from tqdm import tqdm
import evaluate # For ROUGE and BLEURT
import numpy as np
import warnings
from datetime import datetime # <-- FIX: Added this import

# Import our custom model architecture
from v16_base_model import MultimodalSummarizerV16_Base

# --- FIX FOR OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- FIX FOR TENSORFLOW INFO LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# --- FIX FOR TRANSFORMERS USERWARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
# ----------------------------------------

# --- Configuration ---
PARQUET_PATH = r"H:\News_Summarization\Dataset\News_Articles_with_Images\bbc2news_processed\multimodal_dataset_generation_20251114.parquet"
MODEL_CHECKPOINT = r"v16_model_output_BASE\v16_base_checkpoint_epoch_3.pth"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
MBART_MODEL_NAME = "facebook/mbart-large-50"

BATCH_SIZE = 16 # Use a larger batch size for faster evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "v16_model_output_BASE"
LOG_FILENAME = os.path.join(OUTPUT_DIR, f"v16_METRICS_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# ---------------------

def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILENAME, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger('tensorflow').setLevel(logging.ERROR) 
    logging.info(f"--- Starting v16 Metrics Calculation ---")
    logging.info(f"Using device: {DEVICE}")

# We can reuse the same Dataset class from training
class MultimodalDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, image_processor):
        logging.info(f"Loading dataset from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        logging.info(f"Loaded {len(self.df)} items.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        try:
            image = Image.open(item['local_image_path']).convert("RGB")
            image_inputs = self.image_processor(image, return_tensors="pt")
            image_pixel_values = image_inputs['pixel_values'].squeeze(0)
        except Exception:
            image_pixel_values = torch.zeros((3, 224, 224))

        text_inputs = self.tokenizer(
            item['article_text'],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For evaluation, we need the raw target text, not tokenized labels
        return {
            "article_input_ids": text_inputs['input_ids'].squeeze(0),
            "article_attention_mask": text_inputs['attention_mask'].squeeze(0),
            "image_pixel_values": image_pixel_values,
            "target_summary_text": item['final_summary'] # Get the raw text
        }

def calculate_metrics():
    setup_logging()

    # 1. Load Tokenizer and Image Processor
    logging.info("Loading tokenizer and image processor...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL_NAME, src_lang="en_XX", tgt_lang="en_XX")
    image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    
    # 2. Create Dataset and DataLoader
    try:
        dataset = MultimodalDataset(PARQUET_PATH, tokenizer, image_processor)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size]) # We only need the validation set

        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
        logging.info(f"Loaded {len(val_dataset)} validation samples.")
        
    except FileNotFoundError:
        logging.error(f"FATAL: Dataset file not found: {PARQUET_PATH}")
        return
    except Exception as e:
        logging.error(f"FATAL: Error loading dataset: {e}", exc_info=True)
        return

    # 3. Initialize Model and load checkpoint
    logging.info(f"Initializing v16 model and loading weights from {MODEL_CHECKPOINT}...")
    model = MultimodalSummarizerV16_Base(vit_model_name=VIT_MODEL_NAME, mbart_model_name=MBART_MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 4. Load Metrics
    rouge = evaluate.load('rouge')
    bleurt = evaluate.load('bleurt', 'bleurt-20') # Using bleurt-20 for better quality
    
    all_predictions = []
    all_references = []

    # 5. Generation Loop
    logging.info("Generating summaries for validation set...")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Generating Summaries"):
            input_ids = batch['article_input_ids'].to(DEVICE)
            attention_mask = batch['article_attention_mask'].to(DEVICE)
            pixel_values = batch['image_pixel_values'].to(DEVICE)
            
            # Generate summary IDs
            generated_ids = model.generate(
                article_input_ids=input_ids,
                article_attention_mask=attention_mask,
                image_pixel_values=pixel_values,
                num_beams=4,       # Use a smaller num_beams for faster eval
                max_length=256,
                min_length=30,
                early_stopping=True
            )
            
            # Decode and store
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_predictions.extend(preds)
            all_references.extend(batch['target_summary_text'])

    # 6. Compute Final Scores
    logging.info("Generation complete. Calculating scores...")
    
    rouge_results = rouge.compute(predictions=all_predictions, references=all_references)
    bleurt_results = bleurt.compute(predictions=all_predictions, references=all_references)
    
    logging.info("\n" + "="*80)
    logging.info(f"--- Final Metrics for {MODEL_CHECKPOINT} ---")
    logging.info(f"--- (Calculated on {len(all_references)} validation samples) ---")
    logging.info("="*80)
    
    logging.info(f"  eval_rouge1: {rouge_results['rouge1']:.4f}")
    logging.info(f"  eval_rouge2: {rouge_results['rouge2']:.4f}")
    logging.info(f"  eval_rougeL: {rouge_results['rougeL']:.4f}")
    logging.info(f"  eval_bleurt_f1: {np.mean(bleurt_results['scores']):.4f}")
    
    logging.info("="*80)
    logging.info("Metric calculation complete.")


if __name__ == "__main__":
    try:
        calculate_metrics()
    except KeyboardInterrupt:
        logging.warning("--- Metric Calculation Interrupted by User ---")