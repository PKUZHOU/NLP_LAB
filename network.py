import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self,wordNum,args):
        super(Net,self).__init__()
        self.embeding = nn.Embedding(wordNum, args.EmbedSize)
        self.args = args
        self.biLSTM = nn.LSTM(
            args.EmbedSize,
            args.hiddenSize,
            num_layers= args.hiddenNum,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(args.hiddenSize*2,args.hiddenSize//2)
        self.linear2 = nn.Linear(args.hiddenSize//2,args.outSize)
    def forward(self, x):
        x = self.embeding(x)
        x, (hn,cn) = self.biLSTM(x)
        out_size = x.size(0)*x.size(1)
        x = x.contiguous().view((out_size,-1))
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x,dim=1)

        return x