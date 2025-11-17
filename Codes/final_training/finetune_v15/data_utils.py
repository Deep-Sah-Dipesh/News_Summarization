import unicodedata
import pandas as pd
from datasets import Dataset, DatasetDict
import logging

def sanitize_text(text: str) -> str:
    """Basic text cleanup."""
    if not isinstance(text, str): 
        return ""
    return text.replace('""', '"').strip()

def normalize_text(text: str) -> str:
    """Normalize unicode characters."""
    if not isinstance(text, str): 
        return ""
    return ' '.join(unicodedata.normalize('NFKC', text).split())

def load_and_prep_dataset(data_paths) -> DatasetDict:
    """Loads, concatenates, cleans, formats, and splits dataset(s)."""
    
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    all_dfs = []
    logging.info(f"Loading data from {len(data_paths)} file(s)...")
    for path in data_paths:
        try:
            df = pd.read_csv(path, engine='python', on_bad_lines='skip')
            all_dfs.append(df)
            logging.info(f"Loaded {len(df)} rows from {path}")
        except FileNotFoundError:
            logging.error(f"File not found: {path}. Skipping.")
    
    if not all_dfs:
        logging.error("No data loaded. Please check DATA_PATH in config.")
        raise FileNotFoundError("No valid data files were loaded.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Total rows after combining: {len(combined_df)}")
    
    combined_df.dropna(subset=['raw_news_article', 'english_summary', 'hindi_summary'], inplace=True)
    
    for col in ['raw_news_article', 'english_summary', 'hindi_summary']:
        combined_df[col] = combined_df[col].apply(sanitize_text).apply(normalize_text)
    
    raw_dataset = Dataset.from_pandas(combined_df)

    processed_dataset = raw_dataset.map(
        _format_dataset_mbart, 
        batched=True, 
        remove_columns=raw_dataset.column_names
    )
    
    train_test_split = processed_dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

def _format_dataset_mbart(batch):
    """Duplicates each article for its English and Hindi summary."""
    inputs, targets, langs = [], [], []
    for article, eng_summary, hin_summary in zip(
        batch['raw_news_article'], batch['english_summary'], batch['hindi_summary']
    ):
        if isinstance(article, str) and article:
            inputs.append(article)
            targets.append(eng_summary)
            langs.append("en_XX")
            
            inputs.append(article)
            targets.append(hin_summary)
            langs.append("hi_IN")
            
    return {'article': inputs, 'summary': targets, 'target_lang': langs}

def tokenize_function(examples, tokenizer, max_input_len, max_summary_len):
    """Tokenizes articles (inputs) and summaries (labels)."""
    
    tokenizer.src_lang = "en_XX"
    model_inputs = tokenizer(
        examples['article'], 
        max_length=max_input_len, 
        truncation=True
    )
    
    labels_batch = []
    for i in range(len(examples['summary'])):
        tokenizer.tgt_lang = examples['target_lang'][i]
        labels = tokenizer(
            text_target=examples['summary'][i], 
            max_length=max_summary_len, 
            truncation=True
        )
        labels_batch.append(labels['input_ids'])
    
    model_inputs["labels"] = labels_batch
    return model_inputs