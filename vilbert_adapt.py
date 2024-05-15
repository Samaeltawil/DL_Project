import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, VisualBertModel

class CustomBert(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', visual_bert_model='visualbert-vqa-coco-pre', num_labels=1):
        super(CustomBert, self).__init__()

        # Load pre-trained BERT for text processing
        self.text_bert = BertModel.from_pretrained(bert_model)
        self.text_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.text_embedding_dim = self.text_bert.config.hidden_size

        # Load pre-trained VisualBERT for image processing
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        self.image_embedding_dim = self.visual_bert.config.hidden_size

        # Freeze BERT and VisualBERT weights
        for param in self.text_bert.parameters():
            param.requires_grad = False
        for param in self.visual_bert.parameters():
            param.requires_grad = False

        # Classifier layer for combined text and image features
        self.classifier = nn.Linear(self.text_embedding_dim + self.image_embedding_dim, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, visual_embeddings, image_attention_mask, visual_token):
        # Tokenize text inputs
        text_outputs = self.text_bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text_output = text_outputs.pooler_output  # Use the pooled output from BERT
        
        # Process image inputs using VisualBERT
        visual_outputs = self.visual_bert(visual_embeddings,
                                        image_attention_mask,
                                        visual_token)

        
        pooled_image_output = visual_outputs.pooler_output  # Use the pooled output from VisualBERT

        # Concatenate text and image features
        combined_features = torch.cat((pooled_text_output, pooled_image_output), dim=1)

        # Pass through the classifier
        logits = self.classifier(combined_features)
        logits = self.sigmoid(logits)

        return logits