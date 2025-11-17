import torch
from transformers import MBart50TokenizerFast, ViTImageProcessor, MBartForConditionalGeneration
from PIL import Image
import textwrap
import os
import sys
import pandas as pd
import warnings
import re # Added for parsing

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
# --- IMPORTANT: Once Stage 2 is trained, update this to your v16_stage2_checkpoint ---
V16_MODEL_CHECKPOINT = r"v16_model_output_BASE\v16_base_checkpoint_epoch_3.pth"
V15_MODEL_PATH = r"H:\News_Summarization\codes\final_training\mbart-large-50-cnn-summarizer-v15\final_model" # Your v15 model

VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
MBART_MODEL_NAME = "facebook/mbart-large-50" # Base mBART for v16 tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------

# --- Global variables ---
v16_model = None
v15_model = None
tokenizer_v16 = None
tokenizer_v15 = None
image_processor = None
gen_kwargs_v16 = {}
gen_kwargs_v15 = {}
# ---------------------

def load_models():
    """Loads BOTH the v16 multimodal and v15 text-only models."""
    global v16_model, v15_model, tokenizer_v16, tokenizer_v15, image_processor, gen_kwargs_v16, gen_kwargs_v15
    
    if v16_model is not None and v15_model is not None:
        print("Models are already loaded.")
        return True

    print(f"--- Loading Models ---")
    print(f"Using device: {DEVICE}")

    try:
        # 1. Load v16 (Multimodal) Components
        print(f"Loading v16 tokenizer and image processor...")
        tokenizer_v16 = MBart50TokenizerFast.from_pretrained(MBART_MODEL_NAME, src_lang="en_XX", tgt_lang="en_XX")
        image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
        
        print("Initializing v16 model architecture...")
        v16_model = MultimodalSummarizerV16_Base(vit_model_name=VIT_MODEL_NAME, mbart_model_name=MBART_MODEL_NAME)
        
        print(f"Loading v16 trained weights from: {V16_MODEL_CHECKPOINT}")
        v16_model.load_state_dict(torch.load(V16_MODEL_CHECKPOINT, map_location=DEVICE))
        v16_model.to(DEVICE)
        v16_model.eval()
        print("--- v16 Multimodal Model Loaded Successfully ---")

        # 2. Load v15 (Text-Only, Multilingual) Components
        print(f"Loading v15 text-only model from: {V15_MODEL_PATH}...")
        v15_model = MBartForConditionalGeneration.from_pretrained(V15_MODEL_PATH, use_safetensors=True)
        tokenizer_v15 = MBart50TokenizerFast.from_pretrained(V15_MODEL_PATH, src_lang="en_XX", tgt_lang="en_XX")
        v15_model.to(DEVICE)
        v15_model.eval()
        print("--- v15 Text-Only Model Loaded Successfully ---")

        # 3. Set universal generation parameters
        base_gen_kwargs = {
            "num_beams": 8,
            "min_length": 30,
            "length_penalty": 2.0,
            "repetition_penalty": 2.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }
        # --- FIX: Use different max_lengths ---
        gen_kwargs_v16 = {**base_gen_kwargs, "max_length": 256}
        gen_kwargs_v15 = {**base_gen_kwargs, "max_length": 512} # <-- Increased for full Hindi
        
        return True
        
    except FileNotFoundError as e:
        print(f"FATAL: A model file was not found. Check paths.")
        print(e)
        return False
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def generate_summaries(article_text, image_path):
    """
    Generates both English (multimodal) and Hindi (text-only) summaries.
    """
    if not all([v16_model, v15_model, tokenizer_v16, tokenizer_v15, image_processor]):
        print("Models not loaded. Please run the setup cell first.")
        return

    # --- 1. Process Image (for v16) ---
    try:
        image = Image.open(image_path).convert("RGB")
        image_inputs = image_processor(image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(DEVICE)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # --- 2. Process Text (for v16) ---
    text_inputs_v16 = tokenizer_v16(
        article_text, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids_v16 = text_inputs_v16['input_ids'].to(DEVICE)
    attention_mask_v16 = text_inputs_v16['attention_mask'].to(DEVICE)
    
    # --- 3. Process Text (for v15) ---
    text_inputs_v15 = tokenizer_v15(
        article_text, max_length=1024, # v15 was trained on 1024
        padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids_v15 = text_inputs_v15['input_ids'].to(DEVICE)
    attention_mask_v15 = text_inputs_v15['attention_mask'].to(DEVICE)

    # --- 4. Generate Summaries ---
    print("\n" + "="*80)
    print("SOURCE ARTICLE (truncated):")
    print(textwrap.fill(article_text[:500] + "...", 80))
    print(f"\nSOURCE IMAGE:\n{image_path}")
    print("\n" + "="*80)
    print("GENERATING SUMMARIES...")

    with torch.no_grad():
        # --- ENGLISH (Multimodal v16) ---
        eng_output_ids = v16_model.generate(
            article_input_ids=input_ids_v16,
            article_attention_mask=attention_mask_v16,
            image_pixel_values=pixel_values,
            forced_bos_token_id=tokenizer_v16.lang_code_to_id["en_XX"],
            **gen_kwargs_v16
        )
        
        # --- HINDI (Text-Only v15) ---
        hin_output_ids = v15_model.generate(
            input_ids=input_ids_v15,
            attention_mask=attention_mask_v15,
            forced_bos_token_id=tokenizer_v15.lang_code_to_id["hi_IN"],
            **gen_kwargs_v15
        )
    
    eng_summary = tokenizer_v16.decode(eng_output_ids[0], skip_special_tokens=True)
    hin_summary = tokenizer_v15.decode(hin_output_ids[0], skip_special_tokens=True)
    
    # --- 5. Format the Output (as requested) ---
    
    # Extract image info from the v16 output
    image_info = "Image information could not be extracted." # Default
    match = re.search(r"The image (?:shows|is a document containing the text): (.*?) (?:[A-Z][a-z])", eng_summary)
    if match:
        image_info = match.group(1).strip()

    # Split the Hindi summary for the two-paragraph format
    # This is a simple split (2 sentences). More complex logic could be used.
    hin_sentences = re.split(r'(?<=[।|?|!])\s', hin_summary)
    para1 = " ".join(hin_sentences[:2])
    para2_body = " ".join(hin_sentences[2:])
    
    final_hindi_output = (
        f"{para1}\n\n"
        f"चित्र को देख के यह पता लगता है कि: {image_info}\n"
        f"{para2_body}"
    )

    print("\n" + "="*80)
    print("GENERATED ENGLISH SUMMARY (v16 Multimodal):")
    print(textwrap.fill(eng_summary, 80))
    print("\n" + "="*80)
    print("GENERATED HINDI SUMMARY (v15 + v16 Image Info):")
    print(textwrap.fill(final_hindi_output, 80))
    print("="*80)

def get_sample_data(parquet_path, index=0):
    """Utility to grab a test sample from our dataset."""
    try:
        df = pd.read_parquet(parquet_path)
        sample = df.iloc[index]
        return sample['article_text'], sample['local_image_path']
    except Exception as e:
        print(f"Could not load sample data from {parquet_path}: {e}")
        return None, None

# --- Main execution ---
if __name__ == "__main__":
    if load_models():
        print("\nLoading a sample article and image from the dataset...")
        article, image_path = get_sample_data(PARQUET_PATH, index=10) 
        
        if article and image_path:
            generate_summaries(article, image_path)
        else:
            print("Could not load sample data. Using a placeholder test.")