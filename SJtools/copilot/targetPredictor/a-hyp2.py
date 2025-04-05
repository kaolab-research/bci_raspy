""" testing whether lstm can perform in a way such that it fits the problem of 
5 softmax classification better than x,y MSE

result: unknown
"""


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SJtools.copilot.targetPredictor.dataset import MyDataset
import argparse
import tqdm
import matplotlib.pyplot as plt
import wandb
import yaml
from torch.optim import lr_scheduler 
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# this model is same as LSTM model but does not use linear x,y to solve problem. instead it solves it with (classification)
class LSTMCLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=5, device=None):
        super(LSTMCLNet, self).__init__()
        self.device = device
        print(self.device)

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.sm = nn.Softmax(dim=1)

        self.initHC()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()


        if self.device is not None:
            c0 = c0.to(device=self.device)
            h0 = h0.to(device=self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # out.size() --> batch_size, seq_size, features
        # out[:, -1, :] --> batch_size, features --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        out = self.sm(out)

        # out.size() --> batch_size, output_dim
        return out


    def predict(self,x):
        # predict using internal h and c (not for training)

        with torch.no_grad():
            out, (hn, cn) = self.lstm(x, (self.h0, self.c0))
            self.h0 = hn 
            self.c0 = cn

            # out.size() --> batch_size, seq_size, features
            # out[:, -1, :] --> batch_size, features --> just want last time step hidden states! 
            out = self.fc(out[:, -1, :]) 

        # out.size() --> batch_size, output_dim
        return out

    def initHC(self):
        # init H and C for prediction

        # Initialize hidden state with zeros
        self.h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

        # Initialize cell state
        self.c0 = torch.zeros(self.num_layers, 1, self.hidden_dim)

        if self.device is not None:
            self.c0 = self.c0.to(device=self.device)
            self.h0 = self.h0.to(device=self.device)

    def reset(self):
        self.initHC()





def findclassificationy(data_y):
    targets = [(-0.85,0.0),
                (0.85,0.0),
                (0.0,0.85),
                (0.0,-0.85),
                (0.0,0.0),
    ]
    newlist = torch.zeros((data_y.shape[0],5))
    for y_i,y in enumerate(data_y):
        for i in range(5):
            if targets[i][0] == y[0] and targets[i][1] == y[1]:
                newlist[y_i][i] = 1
                break
    
    return newlist
        


if __name__ == "__main__":

    # hyper parameters
    parser = argparse.ArgumentParser(description="Specify Target LSTM Architecture")
    parser.add_argument("-sequence_length",type=int,default=128) # consider 250 (each episode is 500?)
    parser.add_argument("-train_epoch_size",type=int,default=100000)
    parser.add_argument("-eval_epoch_size",type=int,default=1000)
    parser.add_argument("-num_epochs",type=int,default=10)
    parser.add_argument("-save_train_data_name",type=str, default='')
    parser.add_argument("-save_eval_data_name",type=str, default='')
    parser.add_argument("-ignoreStillState", default=True, action='store_true')
    parser.add_argument("-dont_ignoreStillState", dest='ignoreStillState', action='store_false')
    parser.add_argument("-batch_size",type=int,default=64)
    parser.add_argument("-lr", default=0.0003, type=float, help="base learning rate")
    parser.add_argument("-hidden_dim",type=int,default=32)
    parser.add_argument("-num_layers",type=int,default=5)
    parser.add_argument("-model_name",type=str,default="a-hyp2-DEFAULT")
    parser.add_argument("-test", default=False, action='store_true')
    parser.add_argument("-wandb", default=True, action='store_true')
    parser.add_argument("-no_wandb", dest='wandb', action='store_false')
    parser.add_argument("-lr_scheduler", type=str, default="constant", help="constant / reducelronplateau / linear / exp")
    args = parser.parse_args()


    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    train_dataset = MyDataset(sequence_length=args.sequence_length,epoch_size=args.train_epoch_size,save_path=args.save_train_data_name,device=device,ignoreStillState=args.ignoreStillState)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataset = MyDataset(sequence_length=args.sequence_length,epoch_size=args.eval_epoch_size,save_path=args.save_eval_data_name,device=device,ignoreStillState=args.ignoreStillState)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

    model = LSTMCLNet(input_dim=5, hidden_dim=args.hidden_dim, num_layers=args.num_layers,device=device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler == "constant":
        scheduler = None
    if args.lr_scheduler == "reducelronplateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)  # default patience is 10 and factor is 0.1
    # if args.lr_scheduler == "linear":
    #     scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01)
    # if args.lr_scheduler == "exp":
    #     lmbda = lambda epoch: 0.95
    #     scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    

    if args.test:
        folderPath = './SJtools/copilot/targetPredictor/models/'
        PATH = folderPath + args.model_name + ".pt"
        model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    model.to(device)

    # TEST CODE
    if args.test:
        total_eval_loss = 0
        data_y_all = []
        pred_y_all = []
        num_batches = len(eval_loader)
        with torch.no_grad():
            for data_x, data_y in eval_loader:
                data_y = findclassificationy(data_y)
                data_y = data_y.to(device)
                pred_y = model(data_x)
                loss = loss_function(pred_y, data_y)
                total_eval_loss += loss.item()
                data_y_all.append(data_y)
                pred_y_all.append(pred_y)
        eval_loss = total_eval_loss / num_batches
        data_y_all = torch.cat(data_y_all)
        pred_y_all = torch.cat(pred_y_all)


        # PR

        # graph
        data_y_all = torch.argmax(data_y_all,axis=1)
        pred_y_all = torch.argmax(pred_y_all,axis=1)
        print(data_y_all.shape)
        print(pred_y_all.shape)
        print('Precision:',precision_score(data_y_all.tolist(), pred_y_all.tolist(),labels=[0,1,2,3,4],average="weighted"))	
        print('Recall:',recall_score(data_y_all.tolist(), pred_y_all.tolist(),labels=[0,1,2,3,4],average="weighted"))

        d = data_y_all == pred_y_all
        print(f'correctly got {d.sum()} out of {len(d)}. it is {d.sum()/len(d)}')

        n = 30
        indices = np.arange(n)

        width = np.min(np.diff(indices))/3
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(indices-width,data_y_all[:n],width,label='label',align='edge')
        ax.bar(indices,pred_y_all[:n],width,label='pred',align='edge')

        ax.set_xlabel('Test histogram')
        plt.legend(loc='best')
        plt.show()

        print()


    # TRAIN CODE
    if not args.test:
        # wandb
        if args.wandb:
            run = wandb.init(
                project="temp_exp", 
                config = {"architecture":"lstm"},
                entity="aaccjjt",
                )
            wandb.config.update(args)

        train_losses = []
        for i_epoch in range(args.num_epochs):
            model.train()
            total_train_loss = 0
            num_batches = len(train_loader)
            for data_x, data_y in tqdm.tqdm(train_loader):
                data_y = findclassificationy(data_y)
                data_y = data_y.to(device)
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
                    data_y = findclassificationy(data_y)
                    data_y = data_y.to(device)
                    pred_y = model(data_x)
                    loss = loss_function(pred_y, data_y)
                    total_eval_loss += loss.item()
            eval_loss = total_eval_loss / num_batches

            print(f"epoch #{i_epoch} / Loss: {train_losses[-1]} / eval Loss: {eval_loss}")
            if args.wandb: wandb.log({'train_loss': train_losses[-1], 'eval_loss': eval_loss})

            # scheduler
            if scheduler is not None:
                scheduler.step(eval_loss)   # update the learning rate scheduler


        # save model
        folderPath = './SJtools/copilot/targetPredictor/models/'
        fileName = args.model_name
        PATH = folderPath+fileName+".pt"
        torch.save(model.state_dict(), PATH)
        if args.wandb: wandb.save(PATH)

        yamldata = {
        'model': 'LSTMCLNet',
        'path': '',
        'hyperparameters': {
            'input_dim':5,
            'output_dim':5,
            'hidden_dim':args.hidden_dim,
            'num_layers':args.num_layers,}
        }

        yamldata['path'] = PATH
        YAML_PATH = folderPath+fileName+".yaml"
        with open(YAML_PATH, 'w') as file:
            documents = yaml.dump(yamldata, file)
        if args.wandb: wandb.save(YAML_PATH)

