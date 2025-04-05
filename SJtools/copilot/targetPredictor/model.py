import torch.nn as nn
import torch
import torch.nn.functional as F


# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device=None):
        super(LSTM, self).__init__()
        self.device = device

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.initHC()

    def forward(self, x):
        # (1,1,5) = (batch,seq,features)

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

        # out.size() --> batch_size, output_dim
        return out


    def predict(self,x):
        # predict using internal h and c (not for training)
        # (1,1,5) = (batch,seq,features)

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

        
class LSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_lstms=1,device=None):
        super(LSTM2, self).__init__()
        self.device = device

        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Number of hidden layers
        self.num_layers = num_layers
        self.num_lstms = num_lstms

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstms = []
        lstm0 = nn.LSTM(input_dim, hidden_dim, self.num_layers, batch_first=True)
        self.lstms.append(lstm0)
        
        # add n-1 lstms
        for _ in range(num_lstms-1):
            lstm = nn.LSTM(hidden_dim, hidden_dim, self.num_layers, batch_first=True)
            self.lstms.append(lstm)

        if self.device is not None:
            for i in range(num_lstms):
                self.lstms[i] = self.lstms[i].to(self.device)
            # print(h[0].is_cuda,c[0].is_cuda,x.is_cuda,) 
            # print("asfasdfas",next(lstm.parameters()).is_cuda)

        self.relu = nn.ReLU()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # for prediction
        self.initHC()


    def forward(self, x):
        # (1,1,5) = (batch,seq,features)

        # Initialize hidden state with zeros
        # Initialize cell state
        h,c= [],[]
        for i in range(self.num_lstms):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            h.append(h0)
            c.append(c0)

        if self.device is not None:
            for i in range(self.num_lstms):
                c[i] = c[i].to(device=self.device)
                h[i] = h[i].to(device=self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        

        out, (hn, cn) = self.lstms[0](x, (h[0].detach(), c[0].detach()))
        for i in range(1,self.num_lstms):
            out = self.relu(out)
            out, (hn, cn) = self.lstms[i](out, (h[i].detach(), c[i].detach()))

        # out.size() --> batch_size, seq_size, features
        # out[:, -1, :] --> batch_size, features --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 

        # out.size() --> batch_size, output_dim
        return out

    def predict(self,x):
        # predict using internal h and c (not for training)
        # (1,1,5) = (batch,seq,features)

        with torch.no_grad():
            out, (hn, cn) = self.lstms[0](x, (self.h[0], self.c[0]))
            self.h[0] = hn; self.c[0] = cn 

            for i in range(1,self.num_lstms):
                out = self.relu(out)
                out, (hn, cn) = self.lstms[i](out, (self.h[i], self.c[i]))
                self.h[i] = hn; self.c[i] = cn 

            # out.size() --> batch_size, seq_size, features
            # out[:, -1, :] --> batch_size, features --> just want last time step hidden states! 
            out = self.fc(out[:, -1, :]) 

        # out.size() --> batch_size, output_dim
        return out

    def initHC(self):
        # init H and C for prediction

        self.h,self.c = [],[]
        for i in range(self.num_lstms):
            h0 = torch.zeros(self.num_layers, 1, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, 1, self.hidden_dim)
            self.h.append(h0)
            self.c.append(c0)

        if self.device is not None:
            for i in range(self.num_lstms):
                self.c[i] = self.c[i].to(device=self.device)
                self.h[i] = self.h[i].to(device=self.device)

    def reset(self):
        self.initHC()

"""  @@@@@@@@@@@@@@@@@@@@@@@@@@@@  LSTMFCS  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """
        
# single LSTM model with many fc layers at the end
class LSTMFCS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, fc_dim=[10,10], device=None):
        super(LSTMFCS, self).__init__()
        self.device = device

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc_dim = fc_dim
        if len(self.fc_dim) < 1: 
            print("wrong number of fc_dim (len must be greater than 1). see default param")
            exit(1)

        self.fc1 = nn.Linear(hidden_dim, fc_dim[0])
        self.fcs = []
        for i in range(1,len(fc_dim)):
            fc = nn.Linear(fc_dim[i-1], fc_dim[i])
            self.fcs.append(fc)
        self.fcn = nn.Linear(fc_dim[-1], 2)


        if self.device is not None:
            for i in range(len(self.fcs)):
                self.fcs[i] = self.fcs[i].to(self.device)

        self.initHC()

    def forward(self, x):
        # (1,1,5) = (batch,seq,features)
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
        x = F.relu(self.fc1(out[:, -1, :]))
        for fc in self.fcs: x = F.relu(fc(x))
        x = self.fcn(x)

        # out.size() --> batch_size, output_dim

        return x


    def predict(self,x):
        # (1,1,5) = (batch,seq,features)
        # predict using internal h and c (not for training)

        with torch.no_grad():
            out, (hn, cn) = self.lstm(x, (self.h0, self.c0))
            self.h0 = hn 
            self.c0 = cn

            # out.size() --> batch_size, seq_size, features
            # out[:, -1, :] --> batch_size, features --> just want last time step hidden states! 
            x = F.relu(self.fc1(out[:, -1, :]))
            for fc in self.fcs: x = F.relu(fc(x))
            x = self.fcn(x)
        # out.size() --> batch_size, output_dim
        return x

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

"""  @@@@@@@@@@@@@@@@@@@@@@@@@@@@  NN  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """
        
# single NN model with certain lookback

class NN(nn.Module):

    def __init__(self, input_dim, sequence_length, output_dim):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim * sequence_length, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fcn = nn.Linear(10, output_dim)

        # self.fc1_bn = nn.BatchNorm1d(84)
        # self.fc2_bn = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = torch.flatten(x, 1) 
        # x = F.relu(self.fc1_bn(self.fc1(x)))
        # x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fcn(x)
        return x

    def predict(self,x):
        # assume you have seq_dim long images in the past that you can look to features x seq_dim
        # pop the oldes and add the newest one to the front and predict
        # reset will zero out all the values in feataures x seq_dim
        # x = (1,1,5) = (batch,seq,features)

        # print(x.shape) # (1,1,5)

        self.seq_data = torch.cat((self.seq_data[:,1:],x),dim=1)

        with torch.no_grad():
            out = self.forward(self.seq_data)
        
        return out

    def reset(self):
        self.seq_data = torch.zeros((1,self.sequence_length,self.input_dim))

class CNN(nn.Module):

    # def __init__(self, input_dim, sequence_length, output_dim, fc_dim=[10,10], device=None):
    #     super(CNN, self).__init__()
    #     self.fc1 = nn.Linear(input_dim * sequence_length, 84)
    #     self.fc2 = nn.Linear(84, 10)
    #     self.fc3 = nn.Linear(10, 10)
    #     self.fcn = nn.Linear(10, output_dim)

    #     self.fc1_bn = nn.BatchNorm1d(84)
    #     self.fc2_bn = nn.BatchNorm1d(10)
        
    # def forward(self, x):
    #     x = torch.flatten(x, 1) 
    #     # x = F.relu(self.fc1_bn(self.fc1(x)))
    #     # x = F.relu(self.fc2_bn(self.fc2(x)))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = self.fcn(x)
    #     return x

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) #inputchannel, outputchannel, filtersize
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x