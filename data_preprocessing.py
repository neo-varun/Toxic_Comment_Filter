import re
import pandas as pd
from transformers import AutoTokenizer
import torch
import numpy as np

class ToxicCommentProcessor:

    def __init__(self, model_name='distilbert-base-uncased', max_length=None, nrows=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use specified max_length or default to 512 (max supported by the model)
        self.max_length = max_length if max_length is not None else 512
        self.nrows = nrows
        self.target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s!?.,\'\"@#$%^&*()]+', '', text)
        text = text.strip()
        return text
    
    def load_data(self, file_path):
        df = pd.read_csv(file_path, nrows=self.nrows)
        df.drop_duplicates(inplace=True)
        return df
    
    def calculate_max_length(self, texts):
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=True)) for text in texts]
        length_stats = {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            '95th_percentile': np.percentile(lengths, 95),
            'max': np.max(lengths)
        }
        # Calculate max_length based on 95th percentile but cap at model's max of 512
        calculated_length = int(np.ceil(length_stats['95th_percentile'] / 32) * 32)
        max_length = min(calculated_length, 512)
        print(f"Calculated max_length: {calculated_length}, using: {max_length} (model limit: 512)")
        return max_length
    
    def prepare_data(self, file_path='data/train.csv'):
        df = self.load_data(file_path)
        X = df['comment_text'].apply(self.preprocess_text)
        if self.max_length is None:
            self.max_length = self.calculate_max_length(X)
        encoded_data = self.tokenizer(
            list(X),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        y = df[self.target_columns]
        labels = torch.tensor(y.values, dtype=torch.float32)
        return input_ids, attention_masks, labels
    
    def get_data_for_training(self, file_path='data/train.csv'):
        input_ids, attention_masks, labels = self.prepare_data(file_path)
        return {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'labels': labels,
            'max_length': self.max_length,
            'target_columns': self.target_columns
        }