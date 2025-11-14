import os
import torch

# --- Main Paths ---
BASE_MODEL = "facebook/mbart-large-50"
DATA_PATH = "../../Dataset/new_large_CNN_dataset.csv" 
MODEL_OUTPUT_DIR = "mbart-large-50-cnn-summarizer-v15"
FINAL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "final_model")

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.1 # Regularization

# --- Evaluation & Generation ---
METRIC_FOR_BEST_MODEL = "eval_bleurt_f1"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 256
EVAL_BEAMS = 5
# Speed up evaluation by using a subset. Set to None to use all samples.
EVAL_SUBSET_SIZE = 500 

# --- Compute ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import torch

# --- Main Paths ---
BASE_MODEL = "facebook/mbart-large-50"
DATA_PATH = "../../Dataset/new_large_CNN_dataset.csv" 
MODEL_OUTPUT_DIR = "mbart-large-50-cnn-summarizer-v15"
FINAL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "final_model")

# --- Training Hyperparameters ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.1 # Regularization

# --- Evaluation & Generation ---
METRIC_FOR_BEST_MODEL = "eval_bleurt_f1"
MAX_INPUT_LENGTH = 1024
MAX_SUMMARY_LENGTH = 256
EVAL_BEAMS = 5
EVAL_SUBSET_SIZE = 250 

# --- Compute ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")