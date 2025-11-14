import logging
import os
import sys
from datetime import datetime

def setup_logging(log_dir: str, log_file_prefix: str) -> str:
    """
    Configures logging to a file, capturing root and transformers logs.
    Also captures and logs uncaught exceptions.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = f"{log_file_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = os.path.join(log_dir, log_file_name)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [file_handler] 
    
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)
    transformers_logger.addHandler(file_handler)
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            logging.warning("--- Run Interrupted by User (KeyboardInterrupt) ---")
            return
        
        logging.error("--- Uncaught Exception ---", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    
    return log_path