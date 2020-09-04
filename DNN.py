import torch as t
import torchvision as tv
import numpy as np
import time
# from  data_fetch import *
from torch.utils.data import Dataset
import pandas as pd


# 超参数
EPOCH = 50000
BATCH_SIZE = 20
DOWNLOAD_MNIST = True   # 下过数据的话, 就可以设置成 False
N_TEST_IMG = 10          # 到时候显示 5张图片看效果, 如上图一


class MyDataset(Dataset):
    
    # 初始化
    def __init__(self):
        # # 读入数据
        # filelist = file_list('./runfile')
        # data_base = data_construct(filelist)
        self.parms = pd.read_csv('parms.csv').to_numpy()
        self.target = pd.read_csv('target.csv').to_numpy()
        
    # 返回df的长度
    def __len__(self):
        return len(self.parms)
    
    # 获取第idx+1列的数据
    def __getitem__(self, idx):
        return t.tensor(self.parms[idx], dtype=t.float32), t.tensor(self.target[idx], dtype=t.float32)



class DNN(t.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        

        train_data = MyDataset()

        # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
        self.train_loader = t.utils.data.DataLoader(
            dataset=train_data, 
            batch_size=BATCH_SIZE,
            shuffle=True)

        # self.test_loader = t.utils.data.DataLoader(
        #     dataset=test_data, 
        #     batch_size=BATCH_SIZE,
        #     shuffle=True) 
            

        self.dnn = t.nn.Sequential(
            t.nn.Linear(7,32),
            t.nn.Sigmoid(),
            t.nn.Linear(32,64),
            t.nn.Sigmoid(),
            t.nn.Linear(64,256),
            t.nn.Sigmoid(),
            t.nn.Linear(256,512),
            t.nn.Sigmoid(),
            t.nn.Linear(512,128),
            t.nn.Sigmoid(),
            t.nn.Linear(128,32),
            t.nn.Dropout(0.2),
            t.nn.Sigmoid(),
            t.nn.Linear(32,16),
            t.nn.ReLU(),
            t.nn.Linear(16,5),
        )

        self.lr = 0.001
        self.loss = t.nn.L1Loss()
        self.opt = t.optim.Adam(self.parameters(), lr = self.lr)

    def forward(self,x):

        nn1 =x# t.tensor(x, dtype=t.float32)
        #print(nn1.shape)
        out = self.dnn(nn1)
        #print(out.shape)
        return(out)

def train():
    use_gpu =  t.cuda.is_available()
    model = DNN()
    if(use_gpu):
        model.cuda()
    print(model)
    loss = model.loss
    opt = model.opt
    dataloader = model.train_loader
    # testloader = model.test_loader

    
    for e in range(EPOCH):
        step = 0
        ts = time.time()
        train_loss = 0
        total = 0
        for (x, y) in (dataloader):

            model.train()# train model dropout used
            step += 1
            # print("x=%s"%(x))
            # print("y=%s"%(y))
            b_x =x# t.tensor(x, dtype=t.float32)#t.tensor(x)#.view(-1,7)   # batch x, shape (batch, 28*28)
            #print(b_x.shape)
            b_y =y# t.tensor(y, dtype=t.float32)#t.tensor(y)#.view(-1,7)
            if(use_gpu):
                b_x = b_x.cuda()
                b_y = b_y.cuda()
            out = model(b_x)
            losses = loss(out,b_y)
            train_loss += losses.detach().item()
            total+=1
            # print("loss= %s"%(losses))
            opt.zero_grad()
            losses.backward()
            opt.step()
        print('epoch=  %s losses=  %s'%(e,train_loss))
        if(e%1000 == 0):
            t.save(model.state_dict(), './model_%s.pkl'%(e)) 
            # if(step%100 == 0):
            #     if(use_gpu):
            #         print(e,step,losses.data.cpu().numpy())
            #     else:
            #         print(e,step,losses.data.numpy())
                
            #     model.eval() # train model dropout not use
            #     for (tx,ty) in testloader:
            #         t_x = tx   # batch x, shape (batch, 28*28)
            #         t_y = ty
            #         if(use_gpu):
            #             t_x = t_x.cuda()
            #             t_y = t_y.cuda()
            #         t_out = model(t_x)
            #         if(use_gpu):
            #             acc = (np.argmax(t_out.data.cpu().numpy(),axis=1) == t_y.data.cpu().numpy())
            #         else:
            #             acc = (np.argmax(t_out.data.numpy(),axis=1) == t_y.data.numpy())

            #         print(time.time() - ts ,np.sum(acc)/1000)
            #         ts = time.time()
            #         break#只测试前1000个
            


    t.save(model, './model.pkl')  # 保存整个网络
    t.save(model.state_dict(), './model_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
    #加载参数的方式
    """net = DNN()
    net.load_state_dict(t.load('./model_params.pkl'))
    net.eval()"""
    #加载整个模型的方式
    # net = t.load('./model.pkl')
    # net.cpu()
    # net.eval()
    # for (tx,ty) in testloader:
    #     t_x = tx   # batch x, shape (batch, 28*28)
    #     t_y = ty

    #     t_out = net(t_x)
    #     #acc = (np.argmax(t_out.data.CPU().numpy(),axis=1) == t_y.data.CPU().numpy())
    #     acc = (np.argmax(t_out.data.numpy(),axis=1) == t_y.data.numpy())

    #     print(np.sum(acc)/1000)

if __name__ == "__main__":
    train()