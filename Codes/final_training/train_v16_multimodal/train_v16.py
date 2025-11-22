import logging
import os
import torch
from functools import partial
from transformers import (
    MBart50TokenizerFast,
    ViTFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    MBartForConditionalGeneration,
    GenerationConfig 
)

import config_v16 as config
from multimodal_model import MultimodalSummarizer
from multimodal_data import MultimodalDataset, custom_data_collator
import metrics_utils
import logging_utils 

def perform_sanity_check(model, train_dataset, device):
    logging.info("--- Performing Pre-Training Sanity Check ---")
    print("\n--- Performing Pre-Training Sanity Check ---")
    try:
        model.eval()
        sample = train_dataset[0]
        batch = custom_data_collator([sample])
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        
        if outputs.loss is not None:
            logging.info("Sanity Check Passed: Model forward pass successful.")
            print("Sanity Check Passed: Model forward pass successful.")
            return True
        else:
            logging.warning("Sanity Check Warning: Forward pass ran but returned no loss.")
            return False
            
    except Exception as e:
        logging.error(f"Sanity Check FAILED: {e}")
        print(f"\nCRITICAL ERROR during sanity check: {e}")
        print("Fix this error before starting full training.")
        raise e
    finally:
        model.train()

def main():
    log_path = logging_utils.setup_logging(config.V16_MODEL_OUTPUT_DIR, "training_log_v16")
    print(f"--- Starting mBART-ViT v16 Multimodal Training ---")
    print(f"Logging all output to: {log_path}\n")
    logging.info(f"--- Starting mBART-ViT v16 Training ---")

    # --- NEW: Explicitly load metrics so user sees progress ---
    logging.info("Initializing metrics (ROUGE/BLEURT)...")
    metrics_utils.load_metrics()
    # ----------------------------------------------------------
    
    logging.info(f"Loading tokenizer from: {config.V15_TEXT_MODEL_PATH}")
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(config.V15_TEXT_MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        print(f"Error: Could not load tokenizer from {config.V15_TEXT_MODEL_PATH}. Check path.")
        return
    
    logging.info(f"Loading feature extractor from: {config.VIT_MODEL_NAME}")
    feature_extractor = ViTFeatureExtractor.from_pretrained(config.VIT_MODEL_NAME)

    logging.info(f"Loading training data from: {config.TRAIN_DATA_PATH}")
    train_dataset = MultimodalDataset(
        parquet_path=config.TRAIN_DATA_PATH,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_text_len=config.MAX_INPUT_LENGTH
    )
    logging.info(f"Training data loaded: {len(train_dataset)} samples")
    
    logging.info(f"Loading evaluation data from: {config.EVAL_DATA_PATH}")
    eval_dataset = MultimodalDataset(
        parquet_path=config.EVAL_DATA_PATH,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_text_len=config.MAX_INPUT_LENGTH,
        subset_size=config.EVAL_SUBSET_SIZE
    )
    logging.info(f"Evaluation data loaded: {len(eval_dataset)} samples")

    logging.info("Loading custom v16 multimodal model architecture...")
    base_config = MBartForConditionalGeneration.from_pretrained(config.V15_TEXT_MODEL_PATH).config
    
    model = MultimodalSummarizer(
        config=base_config,
        v15_text_model_path=config.V15_TEXT_MODEL_PATH,
        vit_model_name=config.VIT_MODEL_NAME,
        freeze_encoders=config.FREEZE_ENCODERS
    )
    
    try:
        gen_config = GenerationConfig.from_pretrained(config.V15_TEXT_MODEL_PATH)
    except:
        gen_config = GenerationConfig()
        
    gen_config.early_stopping = True 
    gen_config.max_length = config.MAX_SUMMARY_LENGTH
    gen_config.num_beams = config.EVAL_BEAMS
    gen_config.decoder_start_token_id = tokenizer.lang_code_to_id["hi_IN"]
    gen_config.forced_bos_token_id = tokenizer.lang_code_to_id["hi_IN"]
    
    model.config.early_stopping = gen_config.early_stopping 
    model.generation_config = gen_config
    model.to(config.DEVICE)

    logging.info("Model loaded successfully.")
    
    perform_sanity_check(model, train_dataset, config.DEVICE)

    logging.info("Configuring training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.V16_MODEL_OUTPUT_DIR,
        
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        
        logging_dir=os.path.join(config.V16_MODEL_OUTPUT_DIR, "logs"),
        logging_strategy="steps",
        logging_steps=50,
        log_level="info",
        eval_strategy="epoch",
        save_strategy="epoch",
        
        load_best_model_at_end=False, 
        save_total_limit=config.NUM_EPOCHS,
        
        predict_with_generate=True,
        fp16=config.DEVICE.type == 'cuda',
        report_to="tensorboard",
        generation_max_length=config.MAX_SUMMARY_LENGTH,
        generation_num_beams=config.EVAL_BEAMS,
        
        dataloader_num_workers=4 if os.name != 'nt' else 0
    )
    
    _compute_metrics_fn = partial(metrics_utils.compute_metrics, tokenizer=tokenizer)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, 
        data_collator=custom_data_collator,
        compute_metrics=_compute_metrics_fn,
    )

    logging.info(f"--- v16 Training started for {config.NUM_EPOCHS} epochs ---")
    print(f"\n--- v16 Training started for {config.NUM_EPOCHS} epochs ---")
    trainer.train()
    
    logging.info("--- v16 Training finished successfully ---")
    print("\n--- v16 Training finished successfully ---")
    logging.info(f"Checkpoints and logs are saved in: {config.V16_MODEL_OUTPUT_DIR}")
    print(f"Checkpoints and logs are saved in: {config.V16_MODEL_OUTPUT_DIR}")
    print("Run 'evaluate_v16.py' to save the best model.")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt):
        print("\n--- Run Interrupted by user ---")
    except Exception as e:
        print(f"\n--- Run FAILED --- \nError: {e}\nSee log file for details.")