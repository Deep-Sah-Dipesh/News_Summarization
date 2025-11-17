import os
import torch

# --- Main Paths ---
# Point to your trained v15 model
BASE_MODEL = "../mbart-large-50-cnn-summarizer-v15/final_model"

# TODO: Update this to your new fine-tuning dataset
DATA_PATH = [
    "../../../Dataset/final_cleaned_dataset_CNN.csv",
    "../../../Dataset/filtered_articles_CNN.csv"
]

# Outputs for this fine-tuning job
MODEL_OUTPUT_DIR = "mbart-large-50-cnn-summarizer-v15_fine_tuned"
FINAL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "final_model")

# --- Fine-Tuning Hyperparameters ---
LEARNING_RATE = 2e-6  # Lowered for fine-tuning
NUM_EPOCHS = 3        # Reduced for fine-tuning
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.1

# --- Evaluation & Generation ---
METRIC_FOR_BEST_MODEL = "eval_bleurt_f1"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 256
EVAL_BEAMS = 5
EVAL_SUBSET_SIZE = 250 # Use subset to speed up evaluation

# --- Compute ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")