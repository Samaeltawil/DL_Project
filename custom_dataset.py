from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

from torch.utils.data import Dataset, DataLoader
import torch
import json

class CustomDataset(Dataset):
    def __init__(self, GT_path, image_path, img_txt_path, transform=None):
        self.GT_path = GT_path
        self.image_path = image_path
        self.img_txt_path = img_txt_path
        self.transform = transform

        self.GT_data = pd.read_csv(GT_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256 # max length of the text (arbitrary)

    def __len__(self):
        return len(self.GT_data)

    def __getitem__(self, index):
        id = self.GT_data.iloc[index, 0]
        text = self.GT_data.iloc[index, 3]

        # get the image data
        # img_path = os.path.join(self.image_path, str(id) + '.jpg')
        # img = Image.open(img_path).convert('RGB')
        # if self.transform:
        #     img = self.transform(img)

        # get the img_txt data and append it to the text data
        text_img_txt_path = os.path.join(self.img_txt_path, str(id) + '.json')
        if os.path.exists(text_img_txt_path):
            with open(text_img_txt_path, 'r') as file:
                img_txt = json.load(file)
            text += img_txt['img_text']

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
        