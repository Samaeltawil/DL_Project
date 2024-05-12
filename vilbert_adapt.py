import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertPooler

class VilBertForHatefulContentDetection(nn.Module):
    def __init__(self, num_classes):
        super(VilBertForHatefulContentDetection, self).__init__()
        # Load VilBert vision model (ResNet-101)
        self.image_feature_extractor = models.resnet101(pretrained=True)
        self.image_feature_extractor.fc = nn.Identity()  # Remove classification layer

        # Load VilBert language model (BERT)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.text_feature_extractor = BertModel(bert_config)

        # Pooler layer for text features
        self.text_pooler = BertPooler(bert_config)

        # Classification layer
        self.classifier = nn.Linear(2048 + bert_config.hidden_size, num_classes)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_feature_extractor(images)
        text_features = self.text_feature_extractor(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled_text_features = self.text_pooler(text_features)
        
        # Concatenate image and text features
        combined_features = torch.cat((image_features, pooled_text_features), dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output
