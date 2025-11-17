import torch
from transformers import MBart50TokenizerFast, ViTImageProcessor
from PIL import Image
import textwrap
import os
import sys

# Import our custom model architecture
from v16_base_model import MultimodalSummarizerV16_Base 

# --- FIX FOR OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- FIX FOR TENSORFLOW INFO LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# ----------------------------------------

# --- Configuration ---
# 1. This is the PARQUET file you created in Phase 1
# We need it to find the images
PARQUET_PATH = r"H:\News_Summarization\Dataset\News_Articles_with_Images\bbc2news_processed\multimodal_dataset_generation_20251114.parquet"

# 2. This is the model you just trained
MODEL_CHECKPOINT = r"v16_model_output_BASE\v16_base_checkpoint_epoch_3.pth"

# 3. Model names (must match training)
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
MBART_MODEL_NAME = "facebook/mbart-large-50"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------

# --- Global variables ---
model = None
tokenizer = None
image_processor = None
# ---------------------

def load_model():
    """Loads the v16 model, tokenizer, and image processor."""
    global model, tokenizer, image_processor
    
    if model is not None:
        print("Model is already loaded.")
        return True

    print(f"--- Loading v16 Model ---")
    print(f"Using device: {DEVICE}")

    try:
        # 1. Load Tokenizer and Image Processor
        tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL_NAME, src_lang="en_XX", tgt_lang="en_XX")
        image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
        
        # 2. Initialize the model architecture
        model = MultimodalSummarizerV16_Base(vit_model_name=VIT_MODEL_NAME, mbart_model_name=MBART_MODEL_NAME)
        
        # 3. Load the saved weights from your checkpoint
        print(f"Loading trained weights from: {MODEL_CHECKPOINT}")
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        
        print("--- Model v16 (Base) Loaded Successfully ---")
        return True
        
    except FileNotFoundError:
        print(f"FATAL: Checkpoint file not found at {MODEL_CHECKPOINT}")
        return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_summary(article_text, image_path):
    """
    Generates a multimodal summary from an article and an image path.
    """
    if not all([model, tokenizer, image_processor]):
        print("Model is not loaded. Please run the cell with load_model() first.")
        return

    # --- 1. Process Image ---
    try:
        image = Image.open(image_path).convert("RGB")
        image_inputs = image_processor(image, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(DEVICE)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # --- 2. Process Text ---
    text_inputs = tokenizer(
        article_text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_inputs['input_ids'].to(DEVICE)
    attention_mask = text_inputs['attention_mask'].to(DEVICE)

    # --- 3. Generate Summary ---
    print("\n" + "="*80)
    print("SOURCE ARTICLE (truncated):")
    print(textwrap.fill(article_text[:500] + "...", 80))
    print(f"\nSOURCE IMAGE:\n{image_path}")
    print("\n" + "="*80)
    print("GENERATING SUMMARY...")

    with torch.no_grad():
        output_ids = model.generate(
            article_input_ids=input_ids,
            article_attention_mask=attention_mask,
            image_pixel_values=pixel_values,
            # Generation parameters
            num_beams=8,
            max_length=256,
            min_length=30,
            length_penalty=2.0,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("GENERATED MULTIMODAL SUMMARY (v16):")
    print(textwrap.fill(summary, 80))
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
    if load_model():
        # --- Get a real example from our dataset ---
        print("\nLoading a sample article and image from the dataset...")
        article, image_path = get_sample_data(PARQUET_PATH, index=10) # Get 11th item
        
        if article and image_path:
            # --- Run inference ---
            generate_summary(article, image_path)
        else:
            print("Could not load sample data. Using a placeholder test.")
            # Fallback test if dataset isn't found
            generate_summary(
                "This is an article about a new car.", 
                "v16_model_output_BASE/images/_42369098_ciaflight_ap203b.jpg" # Example image
            )