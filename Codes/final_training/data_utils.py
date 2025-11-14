import unicodedata
import pandas as pd
from datasets import Dataset, DatasetDict

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

def load_and_prep_dataset(data_path: str) -> DatasetDict:
    """Loads, cleans, formats, and splits the dataset."""
    
    df = pd.read_csv(data_path, engine='python', on_bad_lines='skip')
    df.dropna(subset=['raw_news_article', 'english_summary', 'hindi_summary'], inplace=True)
    
    for col in ['raw_news_article', 'english_summary', 'hindi_summary']:
        df[col] = df[col].apply(sanitize_text).apply(normalize_text)
    
    raw_dataset = Dataset.from_pandas(df)

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