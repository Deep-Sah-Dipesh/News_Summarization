import torch
import pandas as pd
import logging
import os
import sys
from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast, ViTImageProcessor, get_scheduler
# --- FIX: Import AdamW from torch.optim ---
from torch.optim import AdamW 
# ----------------------------------------
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import warnings

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
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
MBART_MODEL_NAME = "facebook/mbart-large-50" # <-- Base mBART

# Hyperparameters
NUM_EPOCHS = 3       
BATCH_SIZE = 8       
LEARNING_RATE = 5e-5 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "v16_model_output_BASE" # <-- Separate output dir
LOG_FILENAME = os.path.join(OUTPUT_DIR, f"v16_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    logging.info(f"--- Starting v16 Multimodal Training (BASE Model) ---")
    logging.info(f"Using device: {DEVICE}")

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
        except Exception as e:
            logging.warning(f"Error loading image {item['local_image_path']}: {e}. Using black placeholder.")
            image_pixel_values = torch.zeros((3, 224, 224))

        text_inputs = self.tokenizer(
            item['article_text'],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            text_target=item['final_summary'], 
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels_ids = labels['input_ids'].squeeze(0)
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        return {
            "article_input_ids": text_inputs['input_ids'].squeeze(0),
            "article_attention_mask": text_inputs['attention_mask'].squeeze(0),
            "image_pixel_values": image_pixel_values,
            "labels": labels_ids
        }

def train_model():
    setup_logging()

    logging.info("Loading tokenizer and image processor...")
    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL_NAME, src_lang="en_XX", tgt_lang="en_XX")
    image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    
    try:
        dataset = MultimodalDataset(PARQUET_PATH, tokenizer, image_processor)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
        logging.info(f"Data split: {train_size} train ({len(train_dataloader)} steps), {val_size} validation ({len(val_dataloader)} steps).")
        
    except FileNotFoundError:
        logging.error(f"FATAL: Dataset file not found: {PARQUET_PATH}")
        return
    except Exception as e:
        logging.error(f"FATAL: Error loading dataset: {e}", exc_info=True)
        return

    logging.info("Initializing v16 model...")
    model = MultimodalSummarizerV16_Base(vit_model_name=VIT_MODEL_NAME, mbart_model_name=MBART_MODEL_NAME)
    model.to(DEVICE)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    logging.info(f"--- Starting Training for {NUM_EPOCHS} Epochs ---")
    logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(NUM_EPOCHS):
        logging.info(f"--- Starting Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        
        for batch in train_progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['article_input_ids'].to(DEVICE)
            attention_mask = batch['article_attention_mask'].to(DEVICE)
            pixel_values = batch['image_pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(
                article_input_ids=input_ids,
                article_attention_mask=attention_mask,
                image_pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            
            if torch.isnan(loss):
                logging.warning(f"NaN loss detected at step. Skipping batch.")
                continue
                
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        logging.info(f"Running validation for Epoch {epoch + 1}...")
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                input_ids = batch['article_input_ids'].to(DEVICE)
                attention_mask = batch['article_attention_mask'].to(DEVICE)
                pixel_values = batch['image_pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(
                    article_input_ids=input_ids,
                    article_attention_mask=attention_mask,
                    image_pixel_values=pixel_values,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        logging.info(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
        
        epoch_save_path = os.path.join(OUTPUT_DIR, f"v16_base_checkpoint_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        logging.info(f"Checkpoint saved to {epoch_save_path}")

    logging.info("--- Training Complete ---")
    logging.info(f"Final model checkpoints saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        logging.warning("--- Training Interrupted by User ---")