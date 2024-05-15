from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from transformers import BertTokenizer
import torch
import json
import torchvision.transforms as transforms
from torchvision.models import resnet50

class CustomDataset(Dataset):
    def __init__(self, GT_path, image_path, img_txt_path, transform=None):
        self.GT_path = GT_path
        self.image_path = image_path
        self.img_txt_path = img_txt_path
        self.transform = transform

        self.GT_data = pd.read_csv(GT_path)
        self.GT_data_subset = self.GT_data.head(6)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 256  # max length of the text (arbitrary)
        self.default_attention_mask = torch.ones((224, 224))  # Default attention mask for images

        # Initialize the ResNet model for visual embeddings extraction
        self.visual_model = resnet50(pretrained=True)
        self.visual_model.eval()
        self.image_preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.GT_data_subset)

    def get_visual_embeddings(self, image_tensor):
        """
        Extract visual embeddings from an image tensor using a pretrained ResNet model.
        """
        with torch.no_grad():
            image_preprocessed = self.image_preprocess(image_tensor)
            image_batch = image_preprocessed.unsqueeze(0)
            visual_output = self.visual_model(image_batch)
            visual_embeddings = visual_output.squeeze()  # Remove batch dimension
        return visual_embeddings

    def __getitem__(self, index):
        id = self.GT_data_subset.iloc[index, 0]
        text = self.GT_data_subset.iloc[index, 3]

        # Load the image
        img_path = os.path.join(self.image_path, f"{id}.jpg")
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if specified
        # if self.transform:
        #     img = self.transform(img)

        # Extract visual embeddings
        print("now")
        visual_embeddings = self.get_visual_embeddings(img)

        # Generate visual_token_type_ids based on visual_embeddings
        visual_token_type_ids = torch.arange(visual_embeddings.size(0))

        label = self.GT_data_subset.iloc[index, 2]
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

        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0), visual_embeddings, self.default_attention_mask, visual_token_type_ids, torch.tensor([label], dtype=torch.float)
