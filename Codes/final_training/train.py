import logging
import os
from functools import partial
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

import config
import data_utils
import metrics_utils
import logging_utils 

def main():
    # --- Setup robust logging ---
    log_path = logging_utils.setup_logging(config.MODEL_OUTPUT_DIR, "training_log_v15")
    print(f"--- Starting mBART v15 Training ---")
    print(f"Logging all output to: {log_path}\n")
    logging.info(f"--- Starting mBART v15 Training ---")
    
    # 1. Load and Prepare Data
    print(f"Loading and processing data from: {config.DATA_PATH}")
    logging.info(f"Loading and processing data from: {config.DATA_PATH}")
    final_datasets = data_utils.load_and_prep_dataset(config.DATA_PATH)
    logging.info(f"Dataset split: {len(final_datasets['train'])} train, {len(final_datasets['test'])} test")
    print(f"Dataset split: {len(final_datasets['train'])} train, {len(final_datasets['test'])} test")

    # 2. Load Base Model and Tokenizer
    logging.info(f"Loading base model and tokenizer from: {config.BASE_MODEL}")
    tokenizer = MBart50TokenizerFast.from_pretrained(config.BASE_MODEL)
    model = MBartForConditionalGeneration.from_pretrained(config.BASE_MODEL, use_safetensors=True)
    
    # 3. Tokenize Datasets
    logging.info("Tokenizing datasets...")
    print("Tokenizing datasets...")
    _tokenize_fn = partial(
        data_utils.tokenize_function,
        tokenizer=tokenizer,
        max_input_len=config.MAX_INPUT_LENGTH,
        max_summary_len=config.MAX_SUMMARY_LENGTH
    )
    tokenized_datasets = final_datasets.map(
        _tokenize_fn,
        batched=True,
        remove_columns=['article', 'summary', 'target_lang']
    )
    
    # Select a subset for evaluation to save time
    eval_dataset = tokenized_datasets["test"]
    if config.EVAL_SUBSET_SIZE is not None:
        if config.EVAL_SUBSET_SIZE > 0 and config.EVAL_SUBSET_SIZE < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(config.EVAL_SUBSET_SIZE))
            logging.info(f"Using a subset of {config.EVAL_SUBSET_SIZE} examples for evaluation.")
            print(f"Using a subset of {config.EVAL_SUBSET_SIZE} examples for evaluation.")

    # 4. Define Training Arguments
    logging.info("Configuring training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.MODEL_OUTPUT_DIR,
        
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        
        # Logging and Saving Strategy
        logging_dir=os.path.join(config.MODEL_OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=50,       # Log training loss every 50 steps
        log_level="info",       # Print trainer logs to console/notebook
        eval_strategy="epoch",  # Run full evaluation every epoch
        save_strategy="epoch",
        
        load_best_model_at_end=False,
        save_total_limit=config.NUM_EPOCHS,
        
        predict_with_generate=True,
        fp16=config.DEVICE.type == 'cuda',
        report_to="tensorboard",
        generation_max_length=config.MAX_SUMMARY_LENGTH,
        generation_num_beams=config.EVAL_BEAMS,
    )
    
    # 5. Initialize Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    _compute_metrics_fn = partial(metrics_utils.compute_metrics, tokenizer=tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_fn,
    )

    # 6. Start Training
    logging.info(f"--- Training started for {config.NUM_EPOCHS} epochs ---")
    print(f"\n--- Training started for {config.NUM_EPOCHS} epochs ---")
    trainer.train()
    
    logging.info("--- Training finished successfully ---")
    print("\n--- Training finished successfully ---")
    logging.info(f"Checkpoints and logs are saved in: {config.MODEL_OUTPUT_DIR}")
    print(f"Checkpoints and logs are saved in: {config.MODEL_OUTPUT_DIR}")
    # logging.info("Run 'evaluate.py' to find the best model and save it.")
    # print("Run 'evaluate.py' to find the best model and save it.")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt):
        # This will be logged by logging_utils.py
        print("\n--- Run Interrupted by user ---")
    except Exception as e:
        # This will be logged by logging_utils.py
        print(f"\n--- Run FAILED --- \nError: {e}\nSee log file for details.")