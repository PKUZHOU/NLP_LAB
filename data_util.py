import os
import re
import pickle
import random
import torch
from torch.utils import data
import bcolz

def load_embeddings(folder_path):
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words' % folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings' % folder_path, mode='r')
    return words, embeddings


class Data(data.Dataset):
    def __init__(self,use_embedding = False):
        self.train_file = 'data/train.txt'
        self.wv_dict = {}
        self.sentence_len = 32
        self.use_embedding = use_embedding
        self.raw_train_data = open(self.train_file).readlines()
        self.embedding_dir = 'data/embedding/zh_char.64'
        self.get_dict(use_embedding)
        self.train_data = []
        self.test_data = []

        print("loading data")
        self.gen_train_data()
        print("data loaded!")
        self.total_train = len(self.train_data)

    def get_vect(self,ch):
        if (not ch in self.wv_dict.keys()):
            if(self.use_embedding):
                return self.wv_dict['<UNK>']
            else:
                return self.wv_dict['<PAD>']
        else:
            return self.wv_dict[ch]

    def convert(self,sentence,fix_len = True):
        train = []
        tag = []
        sentence = sentence.strip().split("  ")
        for word in sentence:
            if(len(word) == 0):
                continue
            if(len(word)==1):
                #single word
                train.append(self.get_vect(word))
                tag.append(2)#s
            else:
                wlen = len(word)
                train.append(self.get_vect(word[0]))
                tag.append(1)#b

                for i in range(1,wlen-1):
                    train.append(self.get_vect(word[i]))
                    tag.append(1)#middle

                train.append(self.get_vect(word[wlen-1]))
                tag.append(2)#end

        if(fix_len):
            if(len(train)>self.sentence_len ):
                train = train[:self.sentence_len ]
                tag = tag[:self.sentence_len ]
            elif(len(train)<self.sentence_len ):
                for i in range(len(train),self.sentence_len ):
                    train.append(self.wv_dict['<PAD>'])
                    tag.append(0)
        return train,tag

    def get_dict(self,use_embedding):
        if(not use_embedding):
            ind = 1
            dic = {'<PAD>':0}
            for line in self.raw_train_data:
                line = line.strip()
                for word in line:
                    if not word in dic.keys():
                        dic[word] = ind
                        ind+=1
            self.wv_dict = dic
        else:
            words, embedding = load_embeddings(self.embedding_dir)
            words = list(words)
            dic = dict(zip(words,range(len(words))))
            self.wv_dict = dic

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
    data = Data()