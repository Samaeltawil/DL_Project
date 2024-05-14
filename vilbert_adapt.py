import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, VisualBertModel
from transformers.models.bert.modeling_bert import BertPooler

class MemeClassifier(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', num_labels=1):
        super(MemeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)  # Load pre-trained BERT

        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # Classifier layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Get the pooled output from BERT

        # Pass through the classifier layer
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)
        return logits