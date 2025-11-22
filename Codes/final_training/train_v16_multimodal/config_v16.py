import os
import torch

# --- Model Paths ---
V15_TEXT_MODEL_PATH = "../finetune_v15/mbart-large-50-cnn-summarizer-v15_fine_tuned/final_model"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
V16_MODEL_OUTPUT_DIR = "mbart-vit-summarizer-v16"
FINAL_SAVE_PATH = os.path.join(V16_MODEL_OUTPUT_DIR, "final_model")

# --- Dataset Paths ---
DATASET_PARENT_DIR = r"H:\News_Summarization\Dataset\News_Articles_with_Images\balanced_50k_dataset_sets"
TRAIN_DATA_PATH = os.path.join(DATASET_PARENT_DIR, "set_1", "multimodal_dataset_set1.parquet")
EVAL_DATA_PATH = os.path.join(DATASET_PARENT_DIR, "set_2", "multimodal_dataset_set2.parquet")
TEST_DATA_PATH = os.path.join(DATASET_PARENT_DIR, "set_3", "multimodal_dataset_set3.parquet")
IMAGE_DIR = os.path.join(DATASET_PARENT_DIR, "set_1", "images") 

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8 
FREEZE_ENCODERS = True 

# --- Evaluation & Generation ---
METRIC_FOR_BEST_MODEL = "eval_bleurt_f1"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 256
EVAL_BEAMS = 5
EVAL_SUBSET_SIZE = 800 

# --- Compute ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")