import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image

from vqa_data import VQA



class VQADataset(data.Dataset):
    def __init__(self, image_dir, vqainfo_file, ans_to_ix_file, description_file, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.vqa_data_file = vqainfo_file
        self.vqa = VQA(vqainfo_file, ans_to_ix_file, description_file)
        self.no_answer = len(self.vqa.ans_to_ix.items())
        self.max_q_word = 22
        self.max_q_desc_word = 45
        pretrained_weights = 'bert-base-uncased'
        #tokenizer_class = BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    # def __getitem__(self, index):
        
    #     #target_answer = [0]*self.no_answer
    #     image = Image.new('RGB', (256, 256))
    #     image_name = self.vqa.images[index]
    #     image = Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB')
    #     if self.transform is not None:
    #             image = self.transform(image)
    #     image = torch.Tensor(image)
    #     question_token = self.vqa.questions[index]
    #     question_arr = []
    #     question_arr.append(self.vocab('<start>'))
    #     question_arr.extend([self.vocab(token) for token in question_token])
    #     question_arr.append(self.vocab('<end>'))
        
    #     padded_question = [self.vocab('<pad>')] * self.max_q_word
    #     if len(question_arr) > self.max_q_word:
    #         padded_question[0:self.max_q_word] = question_arr[0:self.max_q_word]
    #     else:
    #         padded_question[0:len(question_arr)] = question_arr


    #     padded_question = torch.tensor(padded_question)
    #     answer = self.vqa.answers[index]
    #     #target_answer[int(self.vqa.ans_to_ix[answer])] = 1
    #     target_answer = int(self.vqa.ans_to_ix[answer])
    #     #target_answer = torch.tensor(target_answer)

    #     return padded_question, image, target_answer, answer

    def __getitem__(self, index):
        
        #target_answer = [0]*self.no_answer
        image = Image.new('RGB', (256, 256))
        image_name = self.vqa.images[index]
        image = Image.open(os.path.join(self.image_dir, str(image_name))).convert('RGB')
        if self.transform is not None:
                image = self.transform(image)
        image = torch.Tensor(image)
        question = self.vqa.questions[index]
        description = self.vqa.img_to_semantic_info[image_name]
        ques_desc = question + " "+ description
        question_bert_ids = torch.tensor(self.tokenizer.encode(question, add_special_tokens=True, max_length = self.max_q_word, pad_to_max_length = True))
        ques_desc_bert_ids = torch.tensor(self.tokenizer.encode(ques_desc, add_special_tokens=True, max_length = self.max_q_desc_word, pad_to_max_length = True))
        answer = self.vqa.answers[index]
        #target_answer[int(self.vqa.ans_to_ix[answer])] = 1
        target_answer = int(self.vqa.ans_to_ix[answer])
        #target_answer = torch.tensor(target_answer)

        return question_bert_ids, ques_desc_bert_ids, image, target_answer, answer
        
    def __len__(self):
        return len(self.vqa.images)
    

def collate_fn(data):

    question, description, image, answer, answer_str = zip(*data)

    return question, description, image, answer, answer_str


def get_loader(image_dir, vqa_data_path, ans_to_ix_file, description_file, vocab, transform, batch_size, shuffle, num_workers):
    vqa = VQADataset(image_dir=image_dir, vqainfo_file=vqa_data_path, ans_to_ix_file=ans_to_ix_file, description_file=description_file, vocab=vocab, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=vqa, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader
