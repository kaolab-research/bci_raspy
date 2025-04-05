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
import argparse
import wandb
import yaml
from torch.optim import lr_scheduler 

if torch.cuda.is_available():
    useCuda = True
    device = torch.device('cuda:2')
else:
    useCuda = False
    device = None

#ref: https://www.crosstab.io/articles/time-series-pytorch-lstm

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in tqdm.tqdm(data_loader):

        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss


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
    print(f"Val loss: {avg_loss}")
    return avg_loss


##################### LSTM PARAMETER

if __name__ == "__main__":


    # hyper parameters
    parser = argparse.ArgumentParser(description="Specify Target LSTM Architecture")
    parser.add_argument("-batch_size",type=int,default=32)
    parser.add_argument("-hidden_dim",type=int,default=32)
    parser.add_argument("-num_layers",type=int,default=5)
    parser.add_argument("-num_lstms",type=int,default=0)
    parser.add_argument("-sequence_length",type=int,default=128)
    parser.add_argument("-train_epoch_size",type=int,default=10000)
    parser.add_argument("-eval_epoch_size",type=int,default=1000)
    parser.add_argument("-num_epochs",type=int,default=500)
    parser.add_argument("-lr", help="learning rate for training", type=float, default=0.001)
    parser.add_argument("-model_name",type=str,default="LSTM")
    parser.add_argument("-model_type",type=str,default="LSTM_POS1",help='LSTM_POS1 /  LSTM_POS2 / LSTMFCS')
    parser.add_argument("-plot",default=False, action='store_true')
    parser.add_argument("-save_train_data_name",type=str, default='')
    parser.add_argument("-save_eval_data_name",type=str, default='')
    parser.add_argument("-wandb", default=True, action='store_true')
    parser.add_argument("-no_wandb", dest='wandb', action='store_false')
    parser.add_argument("-ignoreStillState", default=True, action='store_true')
    parser.add_argument("-dont_ignoreStillState", dest='ignoreStillState', action='store_false')
    parser.add_argument("-fc_dim", default=None, nargs='+',help='only for LSTMFCS')
    parser.add_argument("-lr_scheduler", type=str, default="constant", help="constant / reducelronplateau / linear / exp")
    parser.add_argument("-real_data_path", nargs='+', default=[],help='if you want to use real data, then provide path for real data and we will use to create data generator')
    parser.add_argument("-real_data_train_val_ratio", type=float, default=0.8,help='train_val ratio for real data (only used when real_data_path is set)')
    parser.add_argument("-include_cursor_pos", default=False, action='store_true')
    parser.add_argument("-n_skip_real_trials", default=5, type=int, help='how many real trials to skip')
    parser.add_argument("-softmax_type", default='complex', type=str,help="choose between (simple, complex, two_peak) for synthetic softmax")
    parser.add_argument("-dataset_copilot",default=None,help='if you want a specific copilot to generate the data, use their file name here')
    parser.add_argument("-show_dataset_generation", default=False, action='store_true')
    parser.add_argument("-randomInitCursorPosition", default=False, action='store_true',help='if you want synthetic env to use rnadom cursor position at the beginning of the game')
    args = parser.parse_args()

    # preprocess argparse
    if args.fc_dim is not None:
        args.fc_dim = [int(x) for x in args.fc_dim]

    # wandb
    if args.wandb:
        run = wandb.init(
            project="Target-Predictor", 
            config = {"architecture":"lstm"},
            entity="aaccjjt",
            )
        wandb.config.update(args)

    # dataset and model
    if len(args.real_data_path) > 0:
        full_dataset = MyRealDataset(args.real_data_path,sequence_length=args.sequence_length,save_path=args.save_train_data_name,device=device,ignoreStillState=args.ignoreStillState,include_cursor_pos=args.include_cursor_pos,n_skip_real_trials=args.n_skip_real_trials)
        train_size = int(args.real_data_train_val_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        input_dim, output_dim, seq_dim = full_dataset.info()
    else:
        if args.dataset_copilot is None:
            copilot_param = None
        else:
            copilot_param={"model":"PPO","alpha":1,"target_predictor":"truth","target_predictor_input":"softmax_pos"}
        train_dataset = MyDataset(sequence_length=args.sequence_length,epoch_size=args.train_epoch_size,save_path=args.save_train_data_name,device=device,ignoreStillState=args.ignoreStillState,include_cursor_pos=args.include_cursor_pos,softmax_type=args.softmax_type,dataset_copilot=args.dataset_copilot,copilot_param=copilot_param,show_dataset_generation=args.show_dataset_generation,randomInitCursorPosition=args.randomInitCursorPosition)
        eval_dataset = MyDataset(sequence_length=args.sequence_length,epoch_size=args.eval_epoch_size,save_path=args.save_eval_data_name,device=device,ignoreStillState=args.ignoreStillState,include_cursor_pos=args.include_cursor_pos,softmax_type=args.softmax_type,dataset_copilot=args.dataset_copilot,copilot_param=copilot_param,show_dataset_generation=args.show_dataset_generation,randomInitCursorPosition=args.randomInitCursorPosition)
        input_dim, output_dim, seq_dim = train_dataset.info()

    if args.model_type == "LSTM_POS1":
    # if args.num_lstms == 0:
        model = LSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers, device=device)    
        yamldata = {
            'model': 'LSTM_POS1',
            'path': '',
            'hyperparameters': {
                'input_dim':input_dim,
                'output_dim':output_dim,
                'hidden_dim':args.hidden_dim,
                'num_layers':args.num_layers,}
        }
    elif args.model_type == "LSTM_POS2":
        model = LSTM2(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers,num_lstms=args.num_lstms, device=device)
        yamldata = {
            'model': 'LSTM_POS2',
            'path': '',
            'hyperparameters': {
                'input_dim':input_dim,
                'output_dim':output_dim,
                'num_lstms':args.num_lstms,
                'hidden_dim':args.hidden_dim,
                'num_layers':args.num_layers,}
        }
    elif args.model_type == "LSTMFCS":
        model = LSTMFCS(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers, device=device,fc_dim=args.fc_dim)
        yamldata = {
            'model': args.model_type,
            'path': '',
            'hyperparameters': {
                'input_dim':input_dim,
                'output_dim':output_dim,
                'num_lstms':args.num_lstms,
                'hidden_dim':args.hidden_dim,
                'num_layers':args.num_layers,
                'fc_dim':args.fc_dim
                }
        }

    elif args.model_type == "NN":
        model = NN(input_dim=input_dim, sequence_length=args.sequence_length, output_dim=output_dim)
        yamldata = {
            'model': args.model_type,
            'path': '',
            'hyperparameters': {
                'input_dim':input_dim,
                'output_dim':output_dim,
                'sequence_length':args.sequence_length,
                }
        }
        model.to(device=device)
    else:
        print(f"Error: No such model:{args.model_type}")
        exit(1)

    # save copilot model param if used
    if args.dataset_copilot is not None:
        yamldata["note_copilot_used"] = args.dataset_copilot
        yamldata["note_copilot_param"] = copilot_param
    print(yamldata)

    if useCuda: model.to(device)

    # optimizer and loss param
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler == "constant":
        scheduler = None
    if args.lr_scheduler == "reducelronplateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)  # default patience is 10 and factor is 0.1
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

    # train begins
    hist = []
    num_epochs = args.num_epochs
    lowest_loss = torch.inf
    for ix_epoch in range(num_epochs):
        print(f"--------- Epoch {ix_epoch} / {num_epochs}")
        train_loss = train_model(train_loader, model, loss_fn, optimizer=optimizer)
        eval_loss = eval_model(test_loader, model, loss_fn)
        hist.append([train_loss,eval_loss])
        if args.wandb: wandb.log({'train_loss': train_loss, 'eval_loss': eval_loss})
        
        # scheduler
        if scheduler is not None:
            scheduler.step(eval_loss)   # update the learning rate scheduler

        # save best eval
        if eval_loss < lowest_loss:
            lowest_loss = eval_loss

            folderPath = './SJtools/copilot/targetPredictor/models/'
            fileName = args.model_name
            BESTPATH = folderPath+fileName+"-best.pt"
            torch.save(model.state_dict(), BESTPATH)
            print("new best eval model saved")


    # save model
    folderPath = './SJtools/copilot/targetPredictor/models/'
    fileName = args.model_name
    PATH = folderPath+fileName+".pt"
    torch.save(model.state_dict(), PATH)
    if args.wandb: wandb.save(PATH)
    if args.wandb: wandb.save(BESTPATH)

    # save hyperparameters
    yamldata['path'] = PATH
    YAML_PATH = folderPath+fileName+".yaml"
    with open(YAML_PATH, 'w') as file:
        documents = yaml.dump(yamldata, file)
    if args.wandb: wandb.save(YAML_PATH)

    # load model
    # if args.num_lstms == 0:
    if args.model_type == "LSTM_POS1":
        model = LSTM(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers, device=device)    
    elif args.model_type == "LSTM_POS2":
        model = LSTM2(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers,num_lstms=args.num_lstms, device=device)
    elif args.model_type == "LSTMFCS":
        model = LSTMFCS(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers, device=device,fc_dim=args.fc_dim)
    elif args.model_type == "NN":
        model = NN(input_dim=input_dim, sequence_length=args.sequence_length, output_dim=output_dim)
    model.load_state_dict(torch.load(PATH))

    if args.plot:
        plt.plot(hist, label="Training loss")
        plt.legend(["train","val"])
        plt.show()
        
    if args.wandb: run.finish()