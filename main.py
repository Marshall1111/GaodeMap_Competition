import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.nn.functional as F
import argparse



class DD(Dataset): 
    def __init__(self, h5file_name, transform=None,target_transform=None):
        super(DD,self).__init__()
                               #对继承自父类的属性进行初始化

        self.f_img = h5py.File('./data/images.h5', 'r')
        self.f_label = h5py.File('./data/labels.h5', 'r')

        self.imgs = self.f_img["img"]
        self.labels = self.f_label["label"]


        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img = self.imgs[index]

        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img) 

        return img,label

    def __len__(self):
    	return len(self.imgs)


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)



class cnn_lstm(nn.Module):
    def __init__(self, time_step, input_channel):
        super(cnn_lstm, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel*time_step, 64, 8, 5, 1),  # [64, 24, 24]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(2, 2, 0),      # [64, 12, 12]

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),      # [128, 6, 6]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0)       # [256, 3, 3]
        )

        self.lstm = nn.Sequential(
            nn.LSTM( input_size=256*8*3, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True),                
            
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(100, 3),
        )

    def forward(self, x):
        out = self.cnn(x)
        #print(out.shape)
        out, state = self.lstm(out.view(4, 5, -1))

        #print(out.shape)
        out = self.fc(out)
        return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gapde Map Traffic model trainer')
    parser.add_argument('--cuda', type=int, default=0, help='is use cuda to traing or not.')
    args = parser.parse_args()

    # is use the cuda for training or not
    if args.cuda != 0:
        isUseCuda = True
    else:
        isUseCuda = False


    # load the dataset
    dd = DD('camera_0712.h5')
    train_loader = DataLoader(dataset=dd, batch_size=4, shuffle=True, num_workers=0)

    # define the model for training
    if args.cuda != 0:
        model = cnn_lstm(5,3).cuda()
    else:
        model = cnn_lstm(5,3)
        
    print(model)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # model paras
    epoch = 100
    batch_sz = 4
    tm_step = 5
    img_channel = 3
    img_width = 1280
    img_height = 720


    # main func
    for ep in range(epoch):

        # start the training mode
        model.train()

        # 
        for batch_idx, (data, target) in enumerate(train_loader):#batch_idx是enumerate（）函数自带的索引，从0开始
            # data.size():[64, 1, 28, 28]
            # target.size():[64]
            
            # convert data to specific data shape
            data = data.view(batch_sz, img_channel*tm_step, img_height, img_width)

            # img normalization, (-1, 1)
            data = data/127.5 - 1
            
            # caculate the output
            if isUseCuda:
                output = model(data.cuda())
            else:
                output = model(data)

            # normalization the label format with array
            label = torch.zeros(batch_sz, 3)
            for i in range(4):
                idx = int(target[i])
                #print(idx)
                label[i][idx] = int(1)
            label = label.long()

            # caculate the model loss value
            optimizer.zero_grad()   # 所有参数的梯度清零

            if isUseCuda:
                loss = cost(output, label.cuda())
            else:
                loss = cost(output, label)
            

            # update model parameters               
            loss.backward()         #即反向传播求梯度
            optimizer.step()        #调用optimizer进行梯度下降更新参数


        print(loss)

        # save the mode parameters
        if ep%10 == 0:
            torch.save(model.state_dict(), './models/model.pth')
            print("model has been saved!")
