import torch
from torch import nn
import bcolz
import numpy as np

def load_embeddings(folder_path):
    folder_path = folder_path.rstrip('/')
    words = bcolz.carray(rootdir='%s/words' % folder_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings' % folder_path, mode='r')
    return words, embeddings


class Net(nn.Module):
    def __init__(self,wordNum,args):
        super(Net,self).__init__()
        if(args.use_embedding):
            words, embeddings = load_embeddings(args.embedding_dir)
            embeddings = np.array(embeddings)
            self.embeding = nn.Embedding(embeddings.shape[0],embeddings.shape[1])
            self.embeding.weight = nn.Parameter(torch.FloatTensor(embeddings))
            for param in self.embeding.parameters():
                param.requires_grad = False
        else:
            self.embeding = nn.Embedding(wordNum, args.EmbedSize)
        self.args = args
        self.biLSTM = nn.LSTM(
            args.EmbedSize,
            args.hiddenSize,
            num_layers= args.hiddenNum,
            batch_first=True,
            bidirectional=True,
            dropout= 0.5
        )
        self.linear = nn.Linear(args.hiddenSize*2,args.outSize)
    def forward(self, x):
        x = self.embeding(x)
        x, (hn,cn) = self.biLSTM(x)
        x = x.contiguous().view((-1,self.args.hiddenSize*2))
        x = self.linear(x)
        return x