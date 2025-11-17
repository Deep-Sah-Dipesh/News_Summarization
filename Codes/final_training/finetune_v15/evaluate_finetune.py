import os
import json
import shutil
import logging
import config_finetune as config
import logging_utils

def find_best_checkpoint():
    """
    Parses trainer_state.json files to find the best checkpoint
    and copies it to the final_model directory.
    """
    log_path = logging_utils.setup_logging(config.MODEL_OUTPUT_DIR, "evaluation_log_v15_fine_tuned")
    print(f"--- Starting FINE-TUNED Evaluation & Save Script ---")
    print(f"Logging all output to: {log_path}\n")
    logging.info("--- Starting Post-Fine-Tuning Evaluation & Save Script ---")
    
    logging.info(f"Using '{config.METRIC_FOR_BEST_MODEL}' as the key metric.")
    print(f"Using '{config.METRIC_FOR_BEST_MODEL}' as the key metric.")
    
    best_metric_value = -1.0 
    best_checkpoint_path = None
    all_results = []

    checkpoint_dirs = [
        d for d in os.listdir(config.MODEL_OUTPUT_DIR) 
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(config.MODEL_OUTPUT_DIR, d))
    ]
    
    if not checkpoint_dirs:
        logging.error(f"No checkpoint directories found in {config.MODEL_OUTPUT_DIR}")
        print(f"Error: No checkpoint directories found in {config.MODEL_OUTPUT_DIR}")
        return

    logging.info(f"Found {len(checkpoint_dirs)} checkpoints to evaluate.")
    print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate.")

    for chkpt_dir in checkpoint_dirs:
        state_path = os.path.join(config.MODEL_OUTPUT_DIR, chkpt_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            logging.warning(f"No trainer_state.json found in {chkpt_dir}, skipping.")
            continue

        with open(state_path, "r") as f:
            state = json.load(f)
        
        eval_log = None
        for log in reversed(state["log_history"]):
            if config.METRIC_FOR_BEST_MODEL in log:
                eval_log = log
                break
        
        if eval_log:
            all_results.append(eval_log)
            metric_value = eval_log[config.METRIC_FOR_BEST_MODEL]
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_checkpoint_path = os.path.join(config.MODEL_OUTPUT_DIR, chkpt_dir)
                logging.info(f"*** New best checkpoint: {chkpt_dir} ({config.METRIC_FOR_BEST_MODEL}: {metric_value}) ***")
                print(f"*** New best checkpoint: {chkpt_dir} ({config.METRIC_FOR_BEST_MODEL}: {metric_value}) ***")
        else:
            logging.warning(f"No evaluation metrics found in {state_path}.")

    summary_header = "\n" + "="*80 + "\n" + "--- FINAL EVALUATION SUMMARY (FINE-TUNED) ---".center(80) + "\n"
    table_header = f"{'Checkpoint':<20} | {'Step':<10} | {'Loss':<10} | {config.METRIC_FOR_BEST_MODEL:<12} | {'RougeL':<10}"
    summary_divider = "-" * len(table_header)
    
    logging.info(summary_header); print(summary_header)
    logging.info(table_header); print(table_header)
    logging.info(summary_divider); print(summary_divider)
    
    for log in sorted(all_results, key=lambda x: x['step']):
        name = f"checkpoint-{log['step']}"
        loss = log.get('eval_loss', 0.0)
        bleurt = log.get(config.METRIC_FOR_BEST_MODEL, 0.0)
        rougeL = log.get('eval_rougeL', 0.0)
        row = f"{name:<20} | {log['step']:<10} | {loss:<10.4f} | {bleurt:<12.4f} | {rougeL:<10.4f}"
        logging.info(row); print(row)
    
    logging.info("="*80 + "\n"); print("="*80 + "\n")

    if best_checkpoint_path:
        logging.info(f"Best model identified: {best_checkpoint_path}")
        print(f"Best model identified: {best_checkpoint_path}")
        
        if os.path.exists(config.FINAL_SAVE_PATH):
            logging.warning(f"Removing existing directory: {config.FINAL_SAVE_PATH}")
            print(f"Removing existing directory: {config.FINAL_SAVE_PATH}")
            shutil.rmtree(config.FINAL_SAVE_PATH)
            
        logging.info(f"Copying {best_checkpoint_path} to {config.FINAL_SAVE_PATH}...")
        print(f"Copying {best_checkpoint_path} to {config.FINAL_SAVE_PATH}...")
        shutil.copytree(best_checkpoint_path, config.FINAL_SAVE_PATH)
        
        logging.info("--- Best fine-tuned model saved successfully! ---")
        print("\n--- Best fine-tuned model saved successfully! ---")
    else:
        logging.error("Could not determine the best model.")
        print("Error: Could not determine the best model.")

if __name__ == "__main__":
    try:
        find_best_checkpoint()
    except KeyboardInterrupt:
        logging.warning("--- Evaluation Interrupted by user ---")
        print("\n--- Evaluation Interrupted ---")
    except Exception as e:
        logging.error("--- Evaluation FAILED ---", exc_info=True)
        print(f"\n--- Evaluation FAILED --- \nError: {e}\nSee log file for details.")