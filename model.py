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
from classifier import SimpleClassifier
from fc import FCNet
from transformers import *

class ImageEncoder(nn.Module):
    def __init__(self, target_size):
        super(ImageEncoder, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.target_size = target_size
        self.init_weights()

    # def get_params(self):
    #     return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        
        features = torch.transpose(features, 1,2)
        features = torch.transpose(features, 2,3)
        feature_size = features.size()
        features = features.contiguous().view(-1, features.size(3))
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
        self.init_weights()

    # def get_params(self):
    #     return list(self.linear.parameters())

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
        num_hid = 1024
        #emb_size = image_emb_size + qst_emb_size
        self.img_att = NewAttention(image_emb_size, qst_emb_size, qst_emb_size)
        # self.linear =  nn.Linear(emb_size, no_ans)
        self.q_net = FCNet([image_emb_size, num_hid])
        self.v_net = FCNet([qst_emb_size, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, no_ans, 0.5)
        
    def forward(self, image, question):
        img_att = self.img_att(image, question)
        img_emb = (img_att * image).sum(1) # [batch, v_dim]
        
        q_repr = self.q_net(question)
        v_repr = self.v_net(img_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        return logits