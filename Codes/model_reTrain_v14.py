import logging
import pandas as pd
import numpy as np
import torch
import evaluate
import os
import unicodedata
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)

# --- Configuration ---
BASE_MODEL_PATH = "facebook/mbart-large-50"
NEW_MODEL_OUTPUT_DIR = "mbart-large-50-cnn-summarizer-v14"
NEW_DATA_PATH = "../Dataset/new_large_CNN_dataset.csv"

# --- Hyperparameters ---
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4 
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.3 
NUM_BEAMS_EVAL = 6
MAX_SUMMARY_LENGTH_EVAL = 256
METRIC_FOR_BEST_MODEL = "bleurt_f1"

# --- Setup Logging ---
log_filename = f"mbart_large_training_log_v14_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

class ZeroLossCallback(TrainerCallback):
    """A callback that stops training if the training loss is zero to prevent wasted resources."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs and logs['loss'] == 0.0:
            logging.error("CRITICAL: Training loss is zero. This indicates a data issue. Stopping training.")
            control.should_training_stop = True

def sanitize_text(text):
    if not isinstance(text, str): return ""
    return text.replace('""', '"').strip()

def normalize_text(text):
    if not isinstance(text, str): return ""
    return ' '.join(unicodedata.normalize('NFKC', text).split())

def main():
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL_PATH)
        model = MBartForConditionalGeneration.from_pretrained(BASE_MODEL_PATH)

        df_new = pd.read_csv(NEW_DATA_PATH, engine='python', on_bad_lines='skip')
        df_new.dropna(subset=['raw_news_article', 'english_summary', 'hindi_summary'], inplace=True)
        
        logging.info("--- Starting Text Sanitization & Normalization ---")
        for col in ['raw_news_article', 'english_summary', 'hindi_summary']:
            df_new[col] = df_new[col].apply(sanitize_text).apply(normalize_text)
        logging.info("--- Text Sanitization & Normalization Finished ---")
        
        raw_dataset = Dataset.from_pandas(df_new)

        # --- KEY CHANGE 1: Preserve language info for each example ---
        def format_dataset_mbart(batch):
            inputs, targets, langs = [], [], []
            for article, eng_summary, hin_summary in zip(
                batch['raw_news_article'], batch['english_summary'], batch['hindi_summary']
            ):
                if isinstance(article, str) and article:
                    # English example
                    inputs.append(article)
                    targets.append(eng_summary)
                    langs.append("en_XX")
                    # Hindi example
                    inputs.append(article)
                    targets.append(hin_summary)
                    langs.append("hi_IN")
            return {'article': inputs, 'summary': targets, 'target_lang': langs}

        processed_dataset = raw_dataset.map(
            format_dataset_mbart, batched=True, remove_columns=raw_dataset.column_names
        )
        
        train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
        final_datasets = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })
        
        # --- KEY CHANGE 2: Robust tokenization function ---
        def tokenize_function(examples):
            # Set the source language for all articles in the batch
            tokenizer.src_lang = "en_XX"
            model_inputs = tokenizer(examples['article'], max_length=1024, truncation=True)
            
            # Process labels individually to set the correct target language for each one
            labels_batch = []
            for i in range(len(examples['summary'])):
                tokenizer.tgt_lang = examples['target_lang'][i]
                labels = tokenizer(
                    text_target=examples['summary'][i], 
                    max_length=MAX_SUMMARY_LENGTH_EVAL, 
                    truncation=True
                )
                labels_batch.append(labels['input_ids'])
            
            model_inputs["labels"] = labels_batch
            return model_inputs

        tokenized_datasets = final_datasets.map(tokenize_function, batched=True, remove_columns=['article', 'summary', 'target_lang'])
        
        rouge_metric = evaluate.load("rouge")
        bleurt_metric = evaluate.load("bleurt", "bleurt-20")

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            # Important: The decoded predictions won't have lang codes, which is fine for evaluation.
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
            bleurt_result = bleurt_metric.compute(predictions=decoded_preds, references=decoded_labels)
            
            result = {
                "rouge1": rouge_result["rouge1"], "rouge2": rouge_result["rouge2"],
                "rougeL": rouge_result["rougeL"], "bleurt_f1": np.mean(bleurt_result["scores"])
            }
            return {k: round(v * 100, 4) for k, v in result.items()}

        training_args = Seq2SeqTrainingArguments(
            output_dir=NEW_MODEL_OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            weight_decay=WEIGHT_DECAY,
            logging_dir=f"{NEW_MODEL_OUTPUT_DIR}/logs",
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=NUM_EPOCHS,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=False,
            report_to="tensorboard",
            generation_max_length=MAX_SUMMARY_LENGTH_EVAL,
            generation_num_beams=NUM_BEAMS_EVAL,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[ZeroLossCallback()]
        )

        logging.info("Starting final training (v14) from scratch with mBART-LARGE...")
        trainer.train()
        logging.info("Training finished. All checkpoints and logs are saved.")
        
    except Exception as e:
        logging.error(f"An unexpected error occurred during the main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()

