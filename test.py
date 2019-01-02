import re
from data_util import Data
from network import Net
from torch.autograd import Variable
import numpy as np
import torch
import argparse
import pkuseg

def test_vis(args):
    #visulize the results
    data = Data(use_embedding=args.use_embedding)
    net = Net(len(data.wv_dict),args)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    test_file = args.test_file
    raw_test_data = open(test_file).readlines()
    for line in raw_test_data:
        sentences = re.split('，|。|？', line.strip())
        for sentence in sentences:
            if (sentence != ''):
                test,tag = data.convert(sentence,fix_len=False)
                test = torch.Tensor(test).long()
                test = test.view(1,test.size(0))
                out = net(Variable(test))
                pred = torch.argmax(out,1)
                pred_word = ""
                # sentence = sentence.strip().split("  ")
                for idx, word in enumerate(sentence):
                    if(pred[idx] == 2):
                        pred_word+=sentence[idx]
                        pred_word+='/'
                    else:
                        pred_word+=sentence[idx]
                print(pred_word)

def test_PRF(args):
    #test the Precision, Recall and F-Score of model
    data = Data(use_embedding=args.use_embedding)
    net = Net(len(data.wv_dict), args)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    test_file = args.test_file
    test_answers = args.test_answers
    raw_test_data = open(test_file).readlines()
    raw_ans = open(test_answers).readlines()

    total_correct = 0
    total_pred = 0
    total_gt = 0

    for index,line in enumerate(raw_test_data):
        test_sentences = re.split('，|。|？', line.strip())
        ans_sentences = re.split('，|。|？',raw_ans[index].strip())
        for s_ind,sentence in enumerate(test_sentences):
            if (sentence != ''):
                test, tag = data.convert(sentence, fix_len=False)
                test = torch.Tensor(test).long()
                test = test.view(1, test.size(0))
                out = net(Variable(test))
                pred = torch.argmax(out, 1).numpy()
                gt_s = ans_sentences[s_ind]
                gt = []
                for w in gt_s.split("  "):
                    if(w!=''):
                        if(len(w) == 1):
                            gt.append(2)
                        else:
                            for i in range(len(w)-1):
                                gt.append(1)
                            gt.append(2)

                gt = np.asarray(gt)
                pred = np.asarray(np.where(pred==2))[0]
                gt = np.asarray(np.where(gt == 2))[0]
                correct = 0
                for x in pred:
                    if x in gt:
                        correct += 1
                pred_num = len(pred)
                gt_num = len(gt)
                total_pred +=pred_num
                total_gt += gt_num
                total_correct+=correct
        if index%100 == 0:
            print(index,'/',len(raw_test_data))
    print("total ground truth ",total_gt)
    print("total predicted ",total_pred)
    print("total correct ",total_correct)


def test_pkuseg(args):
    #test PKUseg
    test_file = args.test_file
    test_answers = args.test_answers
    raw_test_data = open(test_file).readlines()
    raw_ans = open(test_answers).readlines()
    total_correct = 0
    total_pred = 0
    total_gt = 0
    seg = pkuseg.pkuseg()
    for index, line in enumerate(raw_test_data):
        test_sentences = re.split('，|。|？', line.strip())
        ans_sentences = re.split('，|。|？', raw_ans[index].strip())
        for s_ind, sentence in enumerate(test_sentences):
            if (sentence != ''):
                pred_words = seg.cut(sentence)
                gt_s = ans_sentences[s_ind]
                gt = []
                pred = []
                for w in gt_s.split("  "):
                    if (w != ''):
                        if (len(w) == 1):
                            gt.append(2)
                        else:
                            for i in range(len(w) - 1):
                                gt.append(1)
                            gt.append(2)
                for w in pred_words:
                    if (w != ''):
                        if (len(w) == 1):
                            pred.append(2)
                        else:
                            for i in range(len(w) - 1):
                                pred.append(1)
                            pred.append(2)
                gt = np.asarray(gt)
                pred = np.asarray(pred)
                pred = np.asarray(np.where(pred == 2))[0]
                gt = np.asarray(np.where(gt == 2))[0]
                correct = 0
                for x in pred:
                    if x in gt:
                        correct += 1
                pred_num = len(pred)
                gt_num = len(gt)
                total_pred += pred_num
                total_gt += gt_num
                total_correct += correct
        if index % 100 == 0:
            print(index, '/', len(raw_test_data))
    print("total ground truth ", total_gt)
    print("total predicted ", total_pred)
    print("total correct ", total_correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=bool, default=True)
    parser.add_argument("--EmbedSize", type=int, default=64)
    parser.add_argument("--hiddenSize", type=int, default=256)
    parser.add_argument("--hiddenNum", type=int, default=2)
    parser.add_argument("--outSize", type=int, default=3)
    parser.add_argument("--use_embedding", type=bool, default=True)
    parser.add_argument("--embedding_dir", type=str, default='data/embedding/zh_char.64')
    parser.add_argument("--model", type=str, default='model/with_embedding.pkl')
    parser.add_argument("--test_file", type=str, default='data/test.txt')
    parser.add_argument("--test_answers",type=str,default='data/test.answer.txt')
    args = parser.parse_args()
    test_PRF(args)
    # test_pkuseg(args)
    # test_vis(args)