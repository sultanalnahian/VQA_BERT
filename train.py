import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from data_loader import get_loader
from vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from model import ImageEncoder, BertEncoder, VQA_Model

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # val_transform = transforms.Compose([
    #     transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Build data loader
    train_data_loader = get_loader(args.train_image_dir, args.train_vqa_path, args.ix_to_ans_file, args.train_description_file, vocab, train_transform, args.batch_size, shuffle=True, num_workers=args.num_workers)
    #val_data_loader = get_loader(args.val_image_dir, args.val_vqa_path, args.ix_to_ans_file, vocab, val_transform, args.batch_size, shuffle=False, num_workers=args.num_workers)

    image_encoder = ImageEncoder(args.img_feature_size)
    question_emb_size = 1024
    # description_emb_size = 512
    no_ans = 1000
    question_encoder = BertEncoder(question_emb_size)
    # ques_description_encoder = BertEncoder(description_emb_size)
    # vqa_decoder = VQA_Model(args.img_feature_size, question_emb_size, description_emb_size, no_ans)
    vqa_decoder = VQA_Model(args.img_feature_size, question_emb_size, no_ans)
    
    

    pretrained_epoch = 0
    if args.pretrained_epoch > 0:
        pretrained_epoch = args.pretrained_epoch
        image_encoder.load_state_dict(torch.load('./models/image_encoder-' + str(pretrained_epoch) + '.pkl'))
        question_encoder.load_state_dict(torch.load('./models/question_encoder-' + str(pretrained_epoch) + '.pkl'))
        # ques_description_encoder.load_state_dict(torch.load('./models/ques_description_encoder-' + str(pretrained_epoch) + '.pkl'))
        vqa_decoder.load_state_dict(torch.load('./models/vqa_decoder-' + str(pretrained_epoch) + '.pkl'))

    if torch.cuda.is_available():
        image_encoder.cuda()
        question_encoder.cuda()
        # ques_description_encoder.cuda()
        vqa_decoder.cuda()
        print("Cuda is enabled...")

    criterion = nn.CrossEntropyLoss()
    # params = image_encoder.get_params() + question_encoder.get_params() + ques_description_encoder.get_params() + vqa_decoder.get_params()
    params = list(image_encoder.parameters()) + list(question_encoder.parameters())  + list(vqa_decoder.parameters())
    #print("params: ", params)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_train_step = len(train_data_loader)

    min_avg_loss = float("inf")
    overfit_warn = 0

    for epoch in range(args.num_epochs):
        if epoch < pretrained_epoch:
            continue

        image_encoder.train()
        question_encoder.train()
        #ques_description_encoder.train()
        vqa_decoder.train()
        avg_loss = 0.0
        avg_acc = 0.0
        for bi, (question_arr, image_vqa, target_answer, answer_str) in enumerate(train_data_loader):
            loss = 0
            image_encoder.zero_grad()
            question_encoder.zero_grad()
            #ques_description_encoder.zero_grad()
            vqa_decoder.zero_grad()
            
            images = to_var(torch.stack(image_vqa))    
            question_arr = to_var(torch.stack(question_arr))
            #ques_desc_arr = to_var(torch.stack(ques_desc_arr))
            target_answer = to_var(torch.tensor(target_answer))

            image_emb = image_encoder(images)
            question_emb = question_encoder(question_arr)
            #ques_desc_emb = ques_description_encoder(ques_desc_arr)
            #output = vqa_decoder(image_emb, question_emb, ques_desc_emb)
            output = vqa_decoder(image_emb, question_emb)
            
            loss = criterion(output, target_answer)

            _, prediction = torch.max(output,1)
            no_correct_prediction = prediction.eq(target_answer).sum().item()
            accuracy = no_correct_prediction * 100/ args.batch_size

            ####
            target_answer_no = target_answer.tolist()
            prediction_no = prediction.tolist()
            ####
            loss_num = loss.item()
            avg_loss += loss.item()
            avg_acc += no_correct_prediction
            #loss /= (args.batch_size)
            loss.backward()
            optimizer.step()

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Acc: %.4f'
                      %(epoch + 1, args.num_epochs, bi, total_train_step, loss.item(), accuracy))
            
        avg_loss /= (args.batch_size * total_train_step)
        avg_acc /= (args.batch_size * total_train_step)
        print('Epoch [%d/%d], Average Train Loss: %.4f, Average Train acc: %.4f' %(epoch + 1, args.num_epochs, avg_loss, avg_acc))

        # Save the models
        
        torch.save(image_encoder.state_dict(), os.path.join(args.model_path, 'image_encoder-%d.pkl' %(epoch+1)))
        torch.save(question_encoder.state_dict(), os.path.join(args.model_path, 'question_encoder-%d.pkl' %(epoch+1)))
        #torch.save(ques_description_encoder.state_dict(), os.path.join(args.model_path, 'ques_description_encoder-%d.pkl' %(epoch+1)))
        torch.save(vqa_decoder.state_dict(), os.path.join(args.model_path, 'vqa_decoder-%d.pkl' %(epoch+1)))

        overfit_warn = overfit_warn + 1 if (min_avg_loss < avg_loss) else 0
        min_avg_loss = min(min_avg_loss, avg_loss)
        lossFileName = "result/result_"+str(epoch)+".txt"
        test_fd = open(lossFileName, 'w')
        test_fd.write('Epoch: '+ str(epoch) + ' avg_loss: ' + str(avg_loss)+ " avg_acc: "+ str(avg_acc)+"\n")
        test_fd.close()

        if overfit_warn >= 5:
            print("terminated as overfitted")
            break
        

                
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--image_size', type=int, default=224 ,
                        help='size for input images')
    parser.add_argument('--vocab_path', type=str, default='./preprocessed_data/vocab_1000_ans.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='./data/resized_train2014' ,
                        help='directory for resized train images')
    parser.add_argument('--val_image_dir', type=str, default='./data/resized_val2014' ,
                        help='directory for resized val images')
    parser.add_argument('--train_vqa_path', type=str,
                        default='./preprocessed_data/data_prepro_train_1000_ans.json',
                        help='path for train vqa json file')
    parser.add_argument('--val_vqa_path', type=str,
                        default='./preprocessed_data/data_prepro_train_1000_ans.json',
                        help='path for val vqa json file')
    parser.add_argument('--ix_to_ans_file', type=str,
                        default='./preprocessed_data/ix_to_ans_1000.json',
                        help='path for id to answer json file')
    parser.add_argument('--train_description_file', type=str,
                        default='./preprocessed_data/semantic_v2_train.pkl',
                        help='path for id to answer json file')
    parser.add_argument('--log_step', type=int , default=40,
                        help='step size for prining log info')
    parser.add_argument('--img_feature_size', type=int , default=1024 ,
                        help='dimension of image feature')
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    # parser.add_argument('--hidden_size', type=int , default=1024,
    #                     help='dimension of lstm hidden states')
    # parser.add_argument('--num_layers', type=int , default=2 ,
    #                     help='number of layers in lstm')

    parser.add_argument('--pretrained_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()
    print(args)
    main(args)
