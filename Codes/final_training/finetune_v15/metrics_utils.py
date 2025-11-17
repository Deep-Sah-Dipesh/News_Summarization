import evaluate
import numpy as np
import os
import logging 

# Suppress excessive tensorflow warnings from BLEURT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize metrics globally
rouge_metric = evaluate.load("rouge")
bleurt_metric = evaluate.load("bleurt", "bleurt-20")

def compute_metrics(eval_pred, tokenizer):
    """Decodes predictions and computes ROUGE and BLEURT scores."""
    predictions, labels = eval_pred
    
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Compute BLEURT
    bleurt_result = bleurt_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    result = {
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bleurt_f1": np.mean(bleurt_result["scores"])
    }
    
    return {k: round(v, 4) for k, v in result.items()}