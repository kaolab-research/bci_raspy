from random import randint
from tokenize import Double
import torch
import tqdm
# torch.manual_seed(1)
from torch.utils.data import DataLoader
from SJtools.copilot.targetPredictor.dataset import MyDataset
from SJtools.copilot.targetPredictor.datasetFromRealData import MyRealDataset
from SJtools.copilot.targetPredictor.model import LSTM,LSTM2,LSTMFCS,NN
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os.path import exists
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

useCuda = False


def eval_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            loss = loss_function(output, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches

    print(f"test loss: {avg_loss}")
    return avg_loss, output, y


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

    parser = argparse.ArgumentParser(description="Specify Target LSTM Architecture")
    parser.add_argument("-hidden_dim",type=int,default=32)
    parser.add_argument("-sequence_length",type=int,default=128)
    parser.add_argument("-num_layers",type=int,default=2)
    parser.add_argument("-num_lstms",type=int,default=0)
    parser.add_argument("-epoch_size",type=int,default=5000)
    parser.add_argument("-model_name",type=str,default="LSTM")
    parser.add_argument("-plot", default=True, action='store_true')
    parser.add_argument("-no_plot", dest='plot', action='store_false')
    parser.add_argument("-eval_data_name",type=str, default='')
    parser.add_argument("-model_type",type=str,default="LSTM_POS1",help='LSTM_POS1 /  LSTM_POS2 / LSTMFCS')
    parser.add_argument("-print", default=False, action='store_true')
    parser.add_argument("-ignoreStillState", default=True, action='store_true')
    parser.add_argument("-dont_ignoreStillState", dest='ignoreStillState', action='store_false')
    parser.add_argument("-fc_dim", default=None, nargs='+',help='only for LSTMFCS')
    parser.add_argument("-real_data_path", nargs='+', default=[],help='if you want to use real data, then provide path for real data and we will use to create data generator')
    parser.add_argument("-include_cursor_pos", default=False, action='store_true')
    parser.add_argument("-n_skip_real_trials", default=5, type=int, help='how many real trials to skip')
    parser.add_argument("-softmax_type", default='complex', type=str,help="choose between (simple, complex, two_peak) for synthetic softmax")
    args = parser.parse_args()

    # preprocess argparse
    if args.fc_dim is not None:
        args.fc_dim = [int(x) for x in args.fc_dim]

    # use yaml data file if it exists
    folderPath = './SJtools/copilot/targetPredictor/models/'
    yamlpath = folderPath + args.model_name + ".yaml"
    if exists(yamlpath):
        print("yaml found: using yaml's hyperparameters")
        with open(yamlpath) as yaml_file:
            yaml_data = yaml.load(yaml_file, Loader=Loader)
            if 'model' in yaml_data: args.model_type = yaml_data['model']
            yaml_hyp = yaml_data["hyperparameters"]
            if 'hidden_dim' in yaml_hyp: args.hidden_dim = yaml_hyp['hidden_dim']
            if 'num_layers' in yaml_hyp: args.num_layers = yaml_hyp['num_layers']
            if 'num_lstms' in yaml_hyp: args.num_lstms = yaml_hyp['num_lstms']
            if 'epoch_size' in yaml_hyp: args.epoch_size = yaml_hyp['epoch_size']
            if 'fc_dim' in yaml_hyp: args.fc_dim = yaml_hyp['fc_dim']
            if 'sequence_length' in yaml_hyp: args.sequence_length = yaml_hyp['sequence_length']
    
    # exit()

    if len(args.real_data_path) > 0:
        test_dataset = MyRealDataset(args.real_data_path,save_path=args.eval_data_name,ignoreStillState=args.ignoreStillState,include_cursor_pos=args.include_cursor_pos,n_skip_real_trials=args.n_skip_real_trials,truncated_epoch_size=args.epoch_size)
    else:
        test_dataset = MyDataset(epoch_size=args.epoch_size,save_path=args.eval_data_name,ignoreStillState=args.ignoreStillState,softmax_type=args.softmax_type)
    batch_size = len(test_dataset)
    
    # dataset and model
    PATH = folderPath + args.model_name + ".pt"
    input_dim, output_dim, seq_dim = test_dataset.info()
    if args.model_type == "LSTM_POS1":
        print("using LSTM_POS1")
        model = LSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers)    
    elif args.model_type == "LSTM_POS2":
        print("using LSTM_POS2")
        # print(args.num_layers,args.num_lstms)
        model = LSTM2(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers,num_lstms=args.num_lstms)
    elif args.model_type == "LSTMFCS":
        print("using LSTMFCS with:",args.fc_dim)
        model = LSTMFCS(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers, fc_dim=args.fc_dim)
    elif args.model_type == "NN":
        print("using NN")
        model = NN(input_dim=input_dim, sequence_length=args.sequence_length, output_dim=output_dim)
    
    model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    if useCuda: model.cuda()

    # optimizer and loss param
    loss_fn = torch.nn.MSELoss()
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_loss, output, y = eval_model(test_loader, model, loss_fn)
    L2dist = (((output-y)**2).sum(axis=1)**0.5)

    print(f"Avg distance deviation:{L2dist.mean()}")
    if args.plot: plot3D(y,output,L2dist,printResult=args.print)