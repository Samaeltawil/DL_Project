import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CustomBert(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', num_labels=1):
        super(CustomBert, self).__init__()
        self.name = "CustomBert"
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

class ResNet_Bert(nn.Module):
    def __init__(self, resnet_model = 'resnet18', bert_model='bert-base-uncased', num_labels = 1):
        super(ResNet_Bert, self).__init__()
        self.name = "ResNet_Bert"
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, pretrained=True)
        self.resnet.eval()
        self.res_num_features = 1000 # ResNet18
        self.sigmoid = nn.Sigmoid()

        self.bert = BertModel.from_pretrained(bert_model)  # Load pre-trained BERT
        self.bert_num_features = 768

        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # Freeze ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet_fc = nn.Linear(self.res_num_features, 512)
        self.bert_fc = nn.Linear(self.bert_num_features, 512)

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # Classifier layer
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(1024, 1)
        
    def forward(self, input_ids, attention_mask, input_tensor_image):
        res_outputs = self.resnet(input_tensor_image)
        res_outputs = self.resnet_fc(res_outputs) # goes to 512 layers
        res_outputs = F.relu(res_outputs)

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output  # Get the pooled output from BERT
        bert_logits = self.bert_fc(pooled_output)
        bert_logits = F.relu(bert_logits)

        # concatenates the two outputs
        logits = torch.cat((res_outputs, bert_logits), 1)
        logits = self.fc(logits)
        logits = self.sigmoid(logits)

        return logits
    
