import os
import re
import pickle
import random
import torch
from torch.utils import data
class Data(data.Dataset):
    def __init__(self):
        self.train_file = 'data/train.txt'
        self.test_file = 'data/test.txt'
        self.wv_dict = {}
        self.sentence_len = 32
        self.raw_train_data = open(self.train_file).readlines()
        self.raw_test_data = open(self.test_file).readlines()
        self.get_dict()
        self.train_data = []
        self.test_data = []

        print("loading data")
        self.gen_train_data()
        print("data loaded!")
        self.total_train = len(self.train_data)

    def convert(self,sentence):
        train = []
        tag = []
        sentence = sentence.strip().split("  ")
        for word in sentence:
            if(len(word) == 0):
                continue
            if(len(word)==1):
                #single word
                train.append(self.wv_dict[word])
                tag.append(0)#s
            else:
                wlen = len(word)
                train.append(self.wv_dict[word[0]])
                tag.append(1)#begin
                for i in range(1,wlen-1):
                    train.append(self.wv_dict[word[i]])
                    tag.append(2)#middle
                train.append(self.wv_dict[word[wlen-1]])
                tag.append(3)#end

        if(len(train)>self.sentence_len ):
            train = train[:self.sentence_len ]
            tag = tag[:self.sentence_len ]
        elif(len(train)<self.sentence_len ):
            for i in range(len(train),self.sentence_len ):
                train.append(0)
                tag.append(0)
        return train,tag

    def get_dict(self):
        ind = 1
        dict = {'null':0}
        for line in self.raw_train_data:
            line = line.strip()
            for word in line:
                if not word in dict.keys():
                    dict[word] = ind
                    ind+=1
        self.wv_dict = dict
    def gen_train_data(self):
        if(os.path.exists('data/train_data.pkl')):
            with open('data/train_data.pkl','rb') as f:
                self.train_data = pickle.load(f)
        else:
            for line in self.raw_train_data:
                sentences = re.split('，|。|？',line.strip())
                for sentence in sentences:
                    if(sentence!=''):
                        train,tag = self.convert(sentence)
                        self.train_data.append([train,tag])
            with open('data/train_data.pkl','wb') as f:
                pickle.dump(self.train_data,f)
    def get_batch(self,batch_size):
        data = random.sample(self.train_data,batch_size)
        data = torch.Tensor(data).long()
        x = data[:,0,:]
        y = data[:,1,:]
        return x,y
    def __getitem__(self, index):
        data = self.train_data[index]
        data = torch.Tensor(data).long()
        x = data[0]
        y = data[1]
        return x,y
    def __len__(self):
        return len(self.train_data)
if __name__ == '__main__':
    d = Data()
    data,label = d[1]
    print(data,label)