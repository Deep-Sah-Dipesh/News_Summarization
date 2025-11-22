import torch
from transformers import MBart50TokenizerFast, ViTImageProcessor
from PIL import Image
import textwrap
import os
import config_v16 as config
import logging
import sys
import pandas as pd
import random
from multimodal_model import MultimodalSummarizer

model = None
tokenizer = None
image_processor = None

def load_model():
    """Loads the final v16 model and its components."""
    global model, tokenizer, image_processor
    
    if not os.path.exists(config.FINAL_SAVE_PATH):
        print(f"Error: Model path not found: {config.FINAL_SAVE_PATH}")
        print("Please run 'train_v16.py' and 'evaluate_v16.py' first.")
        return False

    print(f"Loading final v16 model from: {config.FINAL_SAVE_PATH}...")
    try:
        model = MultimodalSummarizer.from_pretrained(config.FINAL_SAVE_PATH).to(config.DEVICE)
        tokenizer = MBart50TokenizerFast.from_pretrained(config.FINAL_SAVE_PATH)
        
        try:
            image_processor = ViTImageProcessor.from_pretrained(config.FINAL_SAVE_PATH)
        except Exception as e:
            print(f"Warning: Could not load local image processor ({e}). Using default.")
            image_processor = ViTImageProcessor.from_pretrained(config.VIT_MODEL_NAME)
            
        model.eval()
        print(f"--- Model v16 loaded successfully on {config.DEVICE} ---")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        logging.error("Failed to load v16 model in testing_v16.py", exc_info=True)
        return False

def summarize_multimodal(article_text: str, image_path: str):
    """Generates a Hindi summary from text and an image."""
    if model is None:
        print("Model is not loaded.")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # 1. Process Text
    tokenizer.src_lang = "en_XX"
    text_inputs = tokenizer(
        article_text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    ).to(config.DEVICE)

    # 2. Process Image
    image_inputs = image_processor(
        images=image, 
        return_tensors="pt"
    ).to(config.DEVICE)

    # 3. Force Hindi Output
    hi_lang_id = tokenizer.lang_code_to_id["hi_IN"]
    
    # CRITICAL FIX: Set it on the model config directly
    model.config.forced_bos_token_id = hi_lang_id

    gen_kwargs = {
        "num_beams": 12,
        "length_penalty": 2.0,
        "repetition_penalty": 2.5,
        "no_repeat_ngram_size": 3,
        "max_length": 250,
        "forced_bos_token_id": hi_lang_id 
    }
    
    print("\n" + "="*80)
    print(f"SOURCE ARTICLE (Snippet): {article_text[:200]}...")
    print(f"SOURCE IMAGE: {image_path}")

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=image_inputs.pixel_values,
            **gen_kwargs
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("-" * 80)
    print("GENERATED HINDI SUMMARY (v16):")
    print(textwrap.fill(summary, width=80)) 
    print("="*80)

def main():
    print(f"--- Starting v16 Multimodal Test Script (Batch Mode) ---")
    
    if load_model():
        # Load Set 2 for testing
        test_parquet_path = config.EVAL_DATA_PATH # Using Set 2 as requested
        
        if not os.path.exists(test_parquet_path):
            print(f"Error: Test dataset not found at {test_parquet_path}")
            return

        print(f"Loading 10 random samples from: {test_parquet_path}")
        try:
            df = pd.read_parquet(test_parquet_path)
            # Select 10 random samples
            samples = df.sample(n=10, random_state=42)
            
            for i, row in samples.iterrows():
                print(f"\nProcessing Sample {i}...")
                
                # The article text in the dataset already has "Image context: ..." prepended.
                # We can pass it directly.
                full_text = row['article_text']
                img_path = row['local_image_path']
                
                if os.path.exists(img_path):
                    summarize_multimodal(full_text, img_path)
                else:
                    print(f"Skipping sample {i}: Image not found at {img_path}")

        except Exception as e:
            print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Test Interrupted ---")
    except Exception as e:
        print(f"\n--- Test FAILED: {e}")
        sys.exit(1)