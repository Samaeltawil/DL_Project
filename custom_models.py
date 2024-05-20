import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CustomBert(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', num_labels=1):
        super(CustomBert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.name ="CustomBert"
        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

class ResNet_Bert(nn.Module):
    def __init__(self, resnet_model='resnet18', bert_model='bert-base-uncased', num_labels=1):
        super(ResNet_Bert, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', resnet_model, pretrained=True)
        self.name ="ResNet_Bert"
        self.resnet.eval()
        self.res_num_features = 1000

        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_num_features = 768

        # Freeze BERT and ResNet layers
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet_fc = nn.Linear(self.res_num_features, 512)
        self.bert_fc = nn.Linear(self.bert_num_features, 512)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(512, num_labels)
        self.bn2 = nn.BatchNorm1d(512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, input_tensor_image):
        with torch.no_grad():
            res_outputs = self.resnet(input_tensor_image)
        res_outputs = self.resnet_fc(res_outputs)
        res_outputs = F.relu(res_outputs)
        res_outputs = self.dropout(res_outputs)

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        bert_logits = self.bert_fc(pooled_output)
        bert_logits = F.relu(bert_logits)
        bert_logits = self.dropout(bert_logits)

        logits = torch.cat((res_outputs, bert_logits), 1)
        logits = self.fc1(logits)
        logits = self.bn1(logits)
        logits = F.relu(logits)
        logits = self.fc2(logits)
        logits = self.bn2(logits)
        logits = self.sigmoid(logits)

        return logits
