import torch
from transformers import MBart50TokenizerFast, ViTImageProcessor, GenerationConfig
from PIL import Image
import textwrap
import os
import config_v16 as config
import logging
import sys
import pandas as pd
from multimodal_model import MultimodalSummarizer

# Global components
model = None
tokenizer = None
image_processor = None

def load_model_robustly():
    """
    Loads the model by manually mapping state_dict keys to fix the
    mismatch between 'model.encoder' (checkpoint) and 'mbart.model.encoder' (class).
    Also handles specific renaming for 'vision_projector' -> 'visual_projection'.
    """
    global model, tokenizer, image_processor
    
    if not os.path.exists(config.FINAL_SAVE_PATH):
        print(f"Error: Model path not found: {config.FINAL_SAVE_PATH}")
        return False

    print(f"Loading final v16 model from: {config.FINAL_SAVE_PATH}...")
    
    try:
        # 1. Initialize the Model Structure (Random Weights)
        model = MultimodalSummarizer.from_pretrained(
            config.FINAL_SAVE_PATH, 
            ignore_mismatched_sizes=True
        ).to(config.DEVICE)
        
        # 2. Manually Load and Remap Weights
        bin_path = os.path.join(config.FINAL_SAVE_PATH, "pytorch_model.bin")
        safe_path = os.path.join(config.FINAL_SAVE_PATH, "model.safetensors")
        state_dict = None

        # Check for file type
        if os.path.exists(bin_path):
            print("  - Found pytorch_model.bin, loading...")
            state_dict = torch.load(bin_path, map_location="cpu")
        elif os.path.exists(safe_path):
            print("  - Found model.safetensors, loading...")
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safe_path, device="cpu")
            except ImportError:
                print("    Error: Found .safetensors but 'safetensors' library is missing.")
                print("    Run: pip install safetensors")
                return False
        else:
            print("  - Warning: No weight file (bin/safetensors) found in directory.")
            print("  - Relying on standard loading (High risk of random weights).")

        # 3. Apply Key Remapping (The Fix)
        if state_dict:
            new_state_dict = {}
            print("  - Remapping weights to match MultimodalSummarizer structure...")
            
            for key, value in state_dict.items():
                # FIX 1: Handle Projector Name Mismatch
                if "vision_projector" in key:
                    new_key = key.replace("vision_projector", "visual_projection")
                    new_state_dict[new_key] = value
                    continue

                # FIX 2: Remap standard MBart keys to wrapped keys
                if key.startswith("model."):
                    new_key = f"mbart.{key}"
                elif key.startswith("lm_head."):
                    new_key = f"mbart.{key}"
                elif key.startswith("final_logits_bias"):
                    new_key = f"mbart.{key}"
                else:
                    new_key = key
                
                new_state_dict[new_key] = value

            # FIX 3: Explicitly link Shared Embeddings
            if "mbart.model.shared.weight" in new_state_dict:
                print("  - Linking shared embeddings...")
                shared_weight = new_state_dict["mbart.model.shared.weight"]
                
                if "mbart.model.encoder.embed_tokens.weight" not in new_state_dict:
                    new_state_dict["mbart.model.encoder.embed_tokens.weight"] = shared_weight
                if "mbart.model.decoder.embed_tokens.weight" not in new_state_dict:
                    new_state_dict["mbart.model.decoder.embed_tokens.weight"] = shared_weight
                if "mbart.lm_head.weight" not in new_state_dict:
                    new_state_dict["mbart.lm_head.weight"] = shared_weight
            
            # Load the remapped weights
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            
            # Filter out trivial missing keys
            real_missing = [k for k in missing if "position_ids" not in k]
            print(f"  - Weights Loaded. Missing critical keys: {len(real_missing)}")
            if len(real_missing) > 0:
                print(f"    First 5 missing: {real_missing[:5]}")
        
        # 4. Load Tokenizer & Image Processor
        tokenizer = MBart50TokenizerFast.from_pretrained(config.FINAL_SAVE_PATH)
        print(f"Loading image processor from: {config.VIT_MODEL_NAME}")
        image_processor = ViTImageProcessor.from_pretrained(config.VIT_MODEL_NAME)

        model.eval()
        print(f"--- Model v16 loaded and patched successfully on {config.DEVICE} ---")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        logging.error("Failed to load v16 model", exc_info=True)
        return False

def summarize_multimodal(article_text: str, image_path: str):
    """Generates a Hindi summary from text and an image."""
    if model is None:
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

    # 3. Configure Generation
    hi_lang_id = tokenizer.lang_code_to_id["hi_IN"]
    
    # REFINED GENERATION PARAMETERS TO STOP LOOPS
    gen_kwargs = {
        "num_beams": 4,             # Standard for translation/summarization
        "length_penalty": 1.0,      # Neutral length preference
        "repetition_penalty": 1.2,  # LOWERED: 2.0 was causing the "g g g" loops
        "no_repeat_ngram_size": 3,  # ADDED: Hard block on repeating phrases
        "max_length": 200,
        "min_length": 40,
        "early_stopping": True,
        "forced_bos_token_id": hi_lang_id,
        "decoder_start_token_id": hi_lang_id
    }
    
    print("\n" + "="*80)
    print(f"SOURCE ARTICLE (Snippet): {article_text[:200]}...")
    print(f"SOURCE IMAGE: {image_path}")

    with torch.no_grad():
        try:
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
            
        except Exception as e:
            print(f"Generation Error: {e}")

def main():
    print(f"--- Starting v16 Multimodal Test Script (Batch Mode) ---")
    
    if load_model_robustly():
        test_parquet_path = config.EVAL_DATA_PATH
        
        if not os.path.exists(test_parquet_path):
            print(f"Error: Test dataset not found at {test_parquet_path}")
            return

        print(f"Loading 10 random samples from: {test_parquet_path}")
        try:
            df = pd.read_parquet(test_parquet_path)
            samples = df.sample(n=10, random_state=42)
            
            for i, row in samples.iterrows():
                print(f"\nProcessing Sample {i}...")
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