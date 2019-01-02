import torch
from torch.autograd import Variable
from data_util import Data
import argparse
from network import Net
from torch.utils.data import DataLoader
import torch.optim as optim
import pickle

torch.manual_seed(1)
criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
def adjust_learning_rate(optimizer, decay_rate=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def train(args):
    data = Data(use_embedding=args.use_embedding)
    loader = DataLoader(data,batch_size=args.batch,shuffle=True,num_workers=4)
    word_2_idx = data.wv_dict
    total_words = len(word_2_idx)
    net = Net(total_words,args)
    net.train()
    plot_loss = []
    if(args.GPU):
        net = net.cuda()
    if(args.optimizer=='sgd'):
        opt = optim.SGD(net.parameters(),lr = args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif (args.optimizer=='adam'):
        opt = optim.Adam(net.parameters(),lr=args.lr)
    for epoch in range(args.epoches):
        if(epoch in [3,5,7]):
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])
        for batch_indx,batch in enumerate(loader):
            x,y = batch
            x = Variable(x)
            y = Variable(y.contiguous().view((-1)))
            if(args.GPU):
                x = x.cuda()
                y = y.cuda()
            pred = net(x)
            loss  = criterion(pred,y)

            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
            opt.step()
            opt.zero_grad()
            if(batch_indx%100==0):
                print("epoch: ",epoch,"batch: ",batch_indx,"loss: ",(float(loss.cpu())))
                plot_loss.append(float(loss.cpu()))
        with open("model/"+str(epoch)+'.pkl','wb') as f:
            torch.save(net.state_dict(),f)
    with open('data/loss_plot.pkl','wb') as f:
        pickle.dump(plot_loss,f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", type=bool, default=True)
    parser.add_argument("--time_step", type=int, default=32)
    parser.add_argument("--EmbedSize",type=int,default=64)
    parser.add_argument("--hiddenSize", type=int, default=256)
    parser.add_argument("--hiddenNum", type=int, default=2)
    parser.add_argument("--outSize", type=int, default=3)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--use_embedding", type=bool, default=True)
    parser.add_argument("--embedding_dir", type=str, default='data/embedding/zh_char.64')
    args = parser.parse_args()
    train(args)