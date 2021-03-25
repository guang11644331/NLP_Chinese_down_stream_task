import torch.nn as nn
from pytorch_BERT import BertTokenizer, BertModel


class Config(object):
    def __init__(self):
        self.model_name = 'BERT + bilstm'
        self.train_path = 'data/THUCNews/train.txt'
        self.dev_path = 'data/THUCNews/dev.txt'
        self.test_path = 'data/THUCNews/test.txt'
        self.class_list = [line.strip() for line in open('data/THUCNews/class.txt')]
        self.save_path = 'saved_models/'

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.early_stop = 1000  # 若超过1000bath效果还没有提升，提前结束训练

        self.n_classes = len(self.class_list)
        self.epochs = 3
        self.batch_size = 128
        self.max_len = 32
        self.learning_rate = 1e-5

        self.pretrain_path = 'bert_pretrain/'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_path)
        self.bert_hidden_size = 768

        self.rnn_layers = 2
        self.dropout = 0.5
        self.rnn_hidden_size = 256


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.bilstm = nn.LSTM(config.bert_hidden_size, config.rnn_hidden_size, config.rnn_layers, batch_first=True,
                              dropout=config.dropout, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden_size * 2, config.n_classes)  # 因为双向，所以乘2

    def forward(self, x):
        # x: (token_ids, attention_mask)
        # pooled_output: [CLS], 详见BertModel类里的注释, dim: 768
        encoded_layers, pooled_output = self.bert(input_ids=x[0], attention_mask=x[1], token_type_ids=None,
                                                  output_all_encoded_layers=False)
        output, (h_n, c_n) = self.bilstm(encoded_layers)  # output.shape [128, 32, 512]
        out = self.dropout(output)
        # 要取rnn的最后一个输出 [128, 1, 512]
        out = out[:, -1, :]  # shape [128, 512], 它会自动从三维降到二维, 而不是 [128, 1, 512]
        out = self.fc(out)  # out shape [128, 10]
        return out


# bert + fc
# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.bert = BertModel.from_pretrained(config.pretrain_path)
#         for param in self.bert.parameters():
#             param.requires_grad = True
#
#         self.fc = nn.Linear(config.bert_hidden_size, config.n_classes)
#
#     def forward(self, x):
#         # x: (token_ids, attention_mask)
#         # pooled_output: [CLS], 详见BertModel类里的注释, dim: 768
#         encoded_layers, pooled_output = self.bert(input_ids=x[0], attention_mask=x[1], token_type_ids=None,
#                                                   output_all_encoded_layers=False)
#         out = self.fc(pooled_output)
#         return out
