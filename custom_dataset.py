from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class MemeDataset(Dataset):
    def __init__(self, GT_path, image_path, transform=None):
        self.GT_path = GT_path
        self.GT_data = pd.read_csv(GT_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256

    def __len__(self):
        return len(self.GT_data)

    def __getitem__(self, index):
        text = self.GT_data.iloc[index, 3]
        label = self.GT_data.iloc[index, 2]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), torch.tensor([label], dtype=torch.float)
        