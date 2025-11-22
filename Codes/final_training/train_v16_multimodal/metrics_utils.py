import evaluate
import numpy as np
import os
import logging 

# Suppress excessive tensorflow warnings from BLEURT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Global placeholders - loaded lazily
rouge_metric = None
bleurt_metric = None

def load_metrics():
    """Loads the heavy metric models. Call this once before training."""
    global rouge_metric, bleurt_metric
    if rouge_metric is None:
        print("Loading ROUGE metric...")
        rouge_metric = evaluate.load("rouge")
    
    if bleurt_metric is None:
        print("Loading BLEURT metric (this may take a moment)...")
        bleurt_metric = evaluate.load("bleurt", "bleurt-20")
    
    print("Metrics loaded successfully.")

def compute_metrics(eval_pred, tokenizer):
    """Decodes predictions and computes ROUGE and BLEURT scores."""
    # Ensure metrics are loaded (just in case)
    if rouge_metric is None or bleurt_metric is None:
        load_metrics()

    predictions, labels = eval_pred
    
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleurt_result = bleurt_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    result = {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleurt_f1": np.mean(bleurt_result["scores"])
    }
    
    return {k: round(v, 4) for k, v in result.items()}