import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import logging

class MultimodalDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, feature_extractor, max_text_len, subset_size=None):
        super().__init__()
        try:
            self.df = pd.read_parquet(parquet_path)
        except Exception as e:
            logging.error(f"Could not read parquet file at {parquet_path}: {e}")
            raise
            
        if subset_size is not None and subset_size > 0 and subset_size < len(self.df):
            logging.info(f"Using a random subset of {subset_size} samples.")
            self.df = self.df.sample(n=subset_size, random_state=42)
            self.df = self.df.reset_index(drop=True)
            
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = row['article_text']
        summary = row['final_summary']
        
        try:
            image_path = row['local_image_path']
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.warning(f"Could not load image {image_path}, using blank image. Error: {e}")
            image = Image.new("RGB", (self.feature_extractor.size['height'], self.feature_extractor.size['width']), (255, 255, 255))

        text_inputs = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summary,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids

        image_inputs = self.feature_extractor(
            images=image,
            return_tensors="pt"
        )

        return {
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
            "pixel_values": image_inputs.pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
        }

def custom_data_collator(features):
    batch = {}
    batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
    batch['attention_mask'] = torch.stack([f['attention_mask'] for f in features])
    batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
    batch['labels'] = torch.stack([f['labels'] for f in features])
    
    return batch