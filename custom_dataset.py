from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

# class MMHS_150KDataset(Dataset):
#     def __init__(self, GT_path, image_path, transform=None):
#         self.GT_path = GT_path
#         self.GT_data = pd.read_csv(GT_path)
#         self.image_path = image_path
        
#         self.transform = transform
#         self.len_samples = len(self.GT_data.iloc[:, 0])
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def __len__(self):
#         return self.len_samples

#     def __getitem__(self, idx):
#         ID = self.GT_data.iloc[idx, 0]
#         img_path = os.path.join(self.image_path, str(ID) + '.jpg').replace("\\","/")
#         image = Image.open(img_path).convert('RGB')
#         label = self.GT_data.iloc[idx, 2]
#         text = self.GT_data.iloc[idx, 3] # text is not used in this example

#         if self.transform:
#             image = self.transform(image)

#         # Tokenize text and get input IDs
#         tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
#         input_ids = tokens['input_ids']

#         # Construct attention mask
#         attention_mask = (input_ids != 0).float()

#         # return image, input_ids, attention_mask,label
#         return {
#                     'input_ids': input_ids.flatten(),
#                     'attention_mask': encoding['attention_mask'].flatten(),
#                     'labels': torch.tensor(label, dtype=torch.long)
#         }

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

class MemeDataset(Dataset):
    def __init__(self, GT_path, image_path, transform=None):
        self.GT_path = GT_path
        self.GT_data = pd.read_csv(GT_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 512

    def __len__(self):
        return len(self.GT_data)

    def __getitem__(self, index):
        text = self.GT_data.iloc[index, 3] # text is not used in this example
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
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }