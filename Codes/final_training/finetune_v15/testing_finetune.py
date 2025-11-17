import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import textwrap
import os
import config_finetune as config
import logging
import sys

model = None
tokenizer = None

def load_model():
    """Loads the final v15_fine_tuned model and tokenizer."""
    global model, tokenizer
    
    if not os.path.exists(config.FINAL_SAVE_PATH):
        print(f"Error: Model path not found: {config.FINAL_SAVE_PATH}")
        print("Please run 'train_finetune.py' and 'evaluate_finetune.py' first.")
        return False

    print(f"Loading final model from: {config.FINAL_SAVE_PATH}...")
    try:
        model = MBartForConditionalGeneration.from_pretrained(config.FINAL_SAVE_PATH).to(config.DEVICE)
        tokenizer = MBart50TokenizerFast.from_pretrained(config.FINAL_SAVE_PATH)
        model.eval()
        print(f"--- Model v15_FINE_TUNED loaded successfully on {config.DEVICE} ---")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        logging.error("Failed to load model in testing_finetune.py", exc_info=True)
        return False

def summarize_text(article_text: str):
    """Generates a high-quality Hindi summary."""
    if model is None or tokenizer is None:
        print("Model is not loaded. Please load the model first.")
        return

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
    
    print("\n" + "="*80)
    print("SOURCE ARTICLE:")
    print(textwrap.fill(article_text, width=80))

    tokenizer.src_lang = "en_XX"
    inputs = tokenizer(article_text, return_tensors="pt", max_length=1024, truncation=True).to(config.DEVICE)

    with torch.no_grad():
        hin_summary_ids = model.generate(
            inputs.input_ids,
            forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"],
            **gen_kwargs
        )
    
    hindi_summary = tokenizer.decode(hin_summary_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("GENERATED HINDI SUMMARY (v15_fine_tuned):")
    print(textwrap.fill(hindi_summary, width=80)) 
    print("\n" + "="*80)

def main():
    print(f"--- Starting FINE-TUNED Test Script ---")
    
    if load_model():
        article_to_test = """
        A major tech firm today unveiled its latest flagship smartphone, 
        featuring a revolutionary new camera system with 'periscope zoom' 
        technology. The device, which also boasts a foldable OLED display 
        and 5G connectivity, aims to redefine the premium mobile market. 
        Analysts are optimistic, noting that the innovative camera could be 
        a key differentiator in a crowded field. However, concerns remain 
        about the device's high price point, which exceeds $1,500, 
        potentially limiting its mass-market appeal despite the advanced features.
        """
        summarize_text(article_to_test)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- Test Interrupted by user ---")
    except Exception as e:
        print(f"\n--- Test FAILED with an unexpected error: {e}")
        logging.error("Test script failed", exc_info=True)
        sys.exit(1)