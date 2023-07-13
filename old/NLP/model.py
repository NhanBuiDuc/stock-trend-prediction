from torch import nn
from transformers import BertModel, BertTokenizer


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
        super(SentimentClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = PRE_TRAINED_MODEL_NAME

        self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        encoded_input = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        encoded_input.data["input_ids"] = encoded_input.data["input_ids"].to("cuda")
        encoded_input.data["token_type_ids"] = encoded_input.data["token_type_ids"].to("cuda")
        encoded_input.data["attention_mask"] = encoded_input.data["attention_mask"].to("cuda")
        output = self.bert(**encoded_input).pooler_output
        output = self.drop(output)
        output = self.out(output)
        output = self.softmax(output)
        return output
