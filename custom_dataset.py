from torch.utils.data import Dataset
import os
from transformers import BertTokenizer

from torch.utils.data import Dataset
import torch
import json

class CustomDataset(Dataset):
    """
    A custom dataset class for processing image and text data.

    Args:
        GT_data (pandas.DataFrame): The ground truth data containing image and text information.
        image_path (str): The path to the directory containing the image files.
        img_txt_path (str): The path to the directory containing the image text files.
        transform (callable, optional): A function/transform to apply to the image data. Default is None.

    Attributes:
        image_path (str): The path to the directory containing the image files.
        img_txt_path (str): The path to the directory containing the image text files.
        transform (callable): A function/transform to apply to the image data.
        GT_data (pandas.DataFrame): The ground truth data containing image and text information.
        labels (numpy.ndarray): The labels extracted from the ground truth data.
        ids (numpy.ndarray): The IDs extracted from the ground truth data.
        tokenizer (BertTokenizer): The tokenizer used for encoding the text data.
        max_len (int): The maximum length of the text data.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the encoded text data, attention mask, image data, and label for the given index.
    """

    def __init__(self, GT_data, image_path, img_txt_path):
        self.image_path = image_path
        self.img_txt_path = img_txt_path

        self.GT_data = GT_data
        self.labels = self.GT_data.iloc[:, 2].values
        self.ids = self.GT_data.iloc[:, 0].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256 # max length of the text (arbitrary)

    def __len__(self):
        return len(self.GT_data)

    def __getitem__(self, index):
        id = self.GT_data.iloc[index, 0]
        text = self.GT_data.iloc[index, 3]

        # get the image data
        img_path = os.path.join(self.image_path, str(id) + '.jpg')
        try:
            img = torch.load(img_path)
        except:
            print(f'\n\nError loading image {img_path}')
            print('The preprocessing images script should be run before running this script.\n\n')

        # get the img_txt data and append it to the text data
        text_img_txt_path = os.path.join(self.img_txt_path, str(id) + '.json')
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
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), img, torch.tensor([label], dtype=torch.float)
        