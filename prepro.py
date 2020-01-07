"""
Preoricess a raw json dataset into hdf5/json files.

Caption: Use NLTK or split function to get tokens. 
"""
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json
import re
from vocabulary import Vocabulary
import pickle

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, params):
  
    # preprocess all the question
    print ('example processed tokens:')
    for i,img in enumerate(imgs):
        s = img['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print (txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs


def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top words and their counts:')
    print ('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print ('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    words = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print ('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(words), ))
    print ('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else '<unk>' for w in txt]
        img['final_question'] = question

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top answer and their counts:'    )
    print ('\n'.join(map(str,cw[:20])))
    
    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'],len(atoi)+1) != len(atoi)+1:
            new_imgs.append(img)

    print ('question number reduce from %d to %d '%(len(imgs), len(new_imgs)))
    return new_imgs

def main(params):

    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w:i for i,w in enumerate(top_ans)}
    itoa = {i:w for i,w in enumerate(top_ans)}
    ix_to_ans_file_name = "preprocessed_data/ix_to_ans_"+str(params['num_ans'])+ ".json"
    #json.dump(itoa, open('ix_to_ans_1000.json', 'w'))
    json.dump(itoa, open(ix_to_ans_file_name, 'w'))

    # filter question, which isn't in the top answers.
    imgs_train = filter_question(imgs_train, atoi)

    seed(123) # make reproducible
    shuffle(imgs_train) # shuffle the order

    # tokenization and preprocessing training question
    imgs_train = prepro_question(imgs_train, params)
    # tokenization and preprocessing testing question
    imgs_test = prepro_question(imgs_test, params)

    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    vocab_path = params['vocab_path']
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)

    json.dump(imgs_train, open(params['output_train_json'], 'w'))
    json.dump(imgs_test, open(params['output_test_json'], 'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='vqa_redefined_train.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='vqa_redefined_val.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    parser.add_argument('--output_train_json', default='preprocessed_data/data_prepro_train_1000_ans.json', help='output train json file')
    parser.add_argument('--output_test_json', default='preprocessed_data/data_prepro_test_1000_ans.json', help='output test json file')
    parser.add_argument('--vocab_path', default='preprocessed_data/vocab_1000_ans.pkl', help='output test json file')
  
    # options
    parser.add_argument('--max_length', default=26, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=1, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')

    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print ('parsed input parameters:')
    print (json.dumps(params, indent = 2))
    main(params)
