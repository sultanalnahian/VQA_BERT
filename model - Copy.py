import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from multiprocessing.dummy import Pool as ThreadPool
from attention import Attention, NewAttention
from multiprocessing import Pool
from collections import Counter
import pickle
from transformers import *

class ImageEncoder(nn.Module):
    def __init__(self, target_size):
        super(ImageEncoder, self).__init__()

        resnet = models.resnet152(pretrained=True)
        # moduels_1 = list(resnet.children())[:-1]
        modules = list(resnet.children())[:-2]
        # resnext_modules = list(resnext.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.target_size = target_size
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        
        features = torch.transpose(features, 1,2)
        features = torch.transpose(features, 2,3)
        print("features_size = ", features.size())
        feature_size = features.size()
        features = features.contiguous().view(-1, features.size(3))
        print("features_size = ", features.size())
        features = self.linear(features)
        features = self.bn(features)
        features = features.view(feature_size[0], -1, self.target_size)
        return features

class BertEncoder(nn.Module):
    def __init__(self, target_size):
        super(BertEncoder, self).__init__()

        model_class = BertModel
        pretrained_weights = 'bert-base-uncased'
        #self.tokenizer_class = BertTokenizer
        self.model = model_class.from_pretrained(pretrained_weights)
        self.linear = nn.Linear(768, target_size)

    def get_params(self):
        return list(self.linear.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, question_ids):
        output = self.model(question_ids)[1]
        output = self.linear(output)
        return output

class VQA_Model(nn.Module):
    def __init__(self, image_emb_size, qst_emb_size, no_ans):
        super(VQA_Model, self).__init__()
        emb_size = image_emb_size + qst_emb_size
        self.img_att = NewAttention(image_emb_size, qst_emb_size, qst_emb_size)
        #fc1_size = 1024
        #emb_size = fc1_size + qst_desc_emb_size
        #self.linear1 =  nn.Linear(img_ques_emb_size, fc1_size)
        self.linear =  nn.Linear(emb_size, no_ans)

    def get_params(self):
        return list(self.linear.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        # self.linear2.weight.data.normal_(0.0, 0.02)
        # self.linear2.bias.data.fill_(0)
        
    def forward(self, image_emb, question_emb, ques_desc_emb):
        embedding = torch.cat((image_emb, question_emb, ques_desc_emb),1)
        #fc1 = self.linear1(img_ques_embedding)
        #comb_desc_emb = torch.cat((fc1, ques_desc_emb),1)
        output = self.linear(embedding)
        return output
# class TransformerModel(nn.Module):
#     def __init__(self, vocab_size, emb_size, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(emb_size, dropout)
#         encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.encoder = nn.Embedding(vocab_size, emb_size)
#         self.emb_size = emb_size
#         self.decoder = nn.Linear(emb_size, vocab_size)

#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src, mask = None):
#         src = self.encoder(src) * math.sqrt(self.emb_size)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         return output

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)