""" testing whether no lstm can perform in a way such that it fits the problem of 
5 softmax single snap shot to 2 (x,y) dimension without any interploation 


result: problem is solved. run it for yourself to check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SJtools.copilot.targetPredictor.dataset import MyDataset
import argparse
import tqdm
import matplotlib.pyplot as plt

# create a simple net
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10, 84)  # 5*5 from image dimension
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 10)
        self.fc8 = nn.Linear(10, 10)
        self.fcn = nn.Linear(10, 2)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fcn(x)
        return x



class SnapTargetNet(nn.Module):

    def __init__(self):
        super(SnapTargetNet, self).__init__()
        self.fc1 = nn.Linear(5, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fcn = nn.Linear(10, 2)

        self.fc1_bn = nn.BatchNorm1d(84)
        self.fc2_bn = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = torch.flatten(x, 1) 
        # x = F.relu(self.fc1_bn(self.fc1(x)))
        # x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fcn(x)
        return x

# model = Net()

# losses = []
# data_x = torch.rand(5,10)
# data_y = torch.rand(5,2)
# for i,x in enumerate(data_x):
#     data_y[i,1] = 0.85 if (x.sum() > 5) else 0
#     data_y[i,0] = 0 if (x.sum() > 5) else -0.85

    
# learning_rate = 0.001
# loss_function = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# # print target predictor's optimizer hyper parameter
# print("hyper param of optimizer:")
# for param_group in optimizer.param_groups:
#     for k,v in param_group.items():
#         if k != "params":
#             print(k,v)

# # train
# torch.set_printoptions(sci_mode=False)
# epoch = 0


def findargmaxy(data_x):
    data_y = torch.argmax(data_x,dim=1)
    new = []
    targets = [(-0.85,0.0),
                (0.85,0.0),
                (0.0,0.85),
                (0.0,-0.85),
                (0.0,0.0),
    ]
    newlist = [targets[i] for i in data_y]
    return torch.tensor(newlist)
        



def plot3D(y,output,L2dist,plotN=1000,printResult=True):
    plotSeparate = True
    printResult = printResult

    skip = len(y) // plotN
    ydata = list(range(plotN))
    xdata = output[::skip,0]
    zdata = output[::skip,1]
    x_data = y[::skip,0]
    z_data = y[::skip,1]
    L2dists = L2dist[::skip]


    if not plotSeparate:
        ax = plt.axes(projection='3d')


        # Data for three-dimensional scattered points (together)
        ydata = list(range(plotN))
        ax.scatter3D(xdata, ydata, zdata, c=ydata,cmap='Greens');

        ax.scatter3D(x_data, ydata, z_data, c=ydata, cmap='Reds');
        plt.show()

    else:
        # second plot
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        surf = ax.scatter3D(xdata, ydata, zdata, c=ydata,cmap='Greens');
        ax.set_zlim(-1.01, 1.01)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.scatter3D(x_data, ydata, z_data, c=ydata, cmap='Reds');


        # ax = fig.add_subplot(1, 3, 3)
        # ax.plot(ydata, L2dists);

        plt.show()

    if printResult:
        for (ox,oy),(rx,ry) in zip(output,y):
            print(ox,rx,oy,ry)



if __name__ == "__main__":

    # hyper parameters
    parser = argparse.ArgumentParser(description="Specify Target LSTM Architecture")
    parser.add_argument("-sequence_length",type=int,default=1)
    parser.add_argument("-train_epoch_size",type=int,default=100000)
    parser.add_argument("-eval_epoch_size",type=int,default=1000)
    parser.add_argument("-num_epochs",type=int,default=10)
    parser.add_argument("-save_train_data_name",type=str, default='')
    parser.add_argument("-save_eval_data_name",type=str, default='')
    parser.add_argument("-ignoreStillState", default=True, action='store_true')
    parser.add_argument("-dont_ignoreStillState", dest='ignoreStillState', action='store_false')
    parser.add_argument("-batch_size",type=int,default=64)
    parser.add_argument("-lr", default=0.0003, type=float, help="base learning rate")
    args = parser.parse_args()


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    train_dataset = MyDataset(sequence_length=1,epoch_size=args.train_epoch_size,save_path=args.save_train_data_name,device=device,ignoreStillState=args.ignoreStillState)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataset = MyDataset(sequence_length=args.sequence_length,epoch_size=args.eval_epoch_size,save_path=args.save_eval_data_name,device=device,ignoreStillState=args.ignoreStillState)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

    model = SnapTargetNet()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    for i_epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        num_batches = len(train_loader)
        for data_x, data_y in tqdm.tqdm(train_loader):
            data_x = data_x.reshape(data_x.shape[::2])
            data_y = findargmaxy(data_x)

            pred_y = model(data_x)
            loss = loss_function(pred_y, data_y)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        train_losses.append(total_train_loss / num_batches)

        model.eval()
        total_eval_loss = 0
        num_batches = len(eval_loader)
        with torch.no_grad():
            for data_x, data_y in eval_loader:
                data_x = data_x.reshape(data_x.shape[::2])
                pred_y = model(data_x)
                data_y = findargmaxy(data_x)
                loss = loss_function(pred_y, data_y)
                total_eval_loss += loss.item()
        eval_loss = total_eval_loss / num_batches

        print(f"epoch #{i_epoch} / Loss: {train_losses[-1]} / eval Loss: {eval_loss}")

    # evaluate model
    model.eval()
    real_y_all = []
    pred_y_all = []
    with torch.no_grad():
        for data_x, data_y in eval_loader:
            data_x = data_x.reshape(data_x.shape[::2])
            pred_y = model(data_x)
            data_y = findargmaxy(data_x)

            real_y_all.append(data_y)
            pred_y_all.append(pred_y)

    real_y_all = torch.cat(real_y_all)
    pred_y_all = torch.cat(pred_y_all)

    L2dist = (((pred_y_all-real_y_all)**2).sum(axis=1)**0.5)

    plot3D(real_y_all,pred_y_all,L2dist,plotN=100,printResult=False)

    


# while True:
#     epoch+=1
#     data_x = torch.rand(5,10)
#     data_y = torch.rand(5,2)
#     for i,x in enumerate(data_x):
#         data_y[i,1] = 85 if (x.sum() > 5) else -85
#         data_y[i,0] = 85 if (x.sum() > 5) else -85

#     pred_y = model(data_x)
#     loss = loss_function(pred_y, data_y)
#     losses.append(loss.item())

#     model.zero_grad()
#     loss.backward()

#     optimizer.step()
#     print(epoch,losses[-1],pred_y,data_y)