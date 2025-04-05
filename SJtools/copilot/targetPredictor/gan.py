import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SJtools.copilot.targetPredictor.dataset import MyDataset
import argparse
import tqdm
import wandb
import yaml

# referred: https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/gan.ipynb#scrollTo=PifoVX3brVj8

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device=None):
        super(Generator, self).__init__()
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


        # self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        # self.deconv1_bn = nn.BatchNorm2d(d*8)
        # self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        # self.deconv2_bn = nn.BatchNorm2d(d*4)
        # self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        # self.deconv3_bn = nn.BatchNorm2d(d*2)
        # self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        # self.deconv4_bn = nn.BatchNorm2d(d)
        # self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # def forward(self, input):
    #     # x = F.relu(self.deconv1(input))
    #     x = F.relu(self.deconv1_bn(self.deconv1(input)))
    #     x = F.relu(self.deconv2_bn(self.deconv2(x)))
    #     x = F.relu(self.deconv3_bn(self.deconv3(x)))
    #     x = F.relu(self.deconv4_bn(self.deconv4(x)))
    #     x = torch.tanh(self.deconv5(x))

    #     return x

class Discriminator(nn.Module):
    def __init__(self, d=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d, 5)
        self.fc1_bn = nn.BatchNorm1d(5)
        self.fc2 = nn.Linear(5, 5)
        self.fc2_bn = nn.BatchNorm1d(5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, input):

        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = torch.sigmoid(self.fc3(x))

        return x
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Specify Target LSTM Architecture")
    parser.add_argument("-batch_size",type=int,default=32)
    parser.add_argument("-hidden_dim",type=int,default=32)
    parser.add_argument("-num_layers",type=int,default=5)
    parser.add_argument("-num_lstms",type=int,default=0)
    parser.add_argument("-sequence_length",type=int,default=128)
    parser.add_argument("-train_epoch_size",type=int,default=10000)
    parser.add_argument("-eval_epoch_size",type=int,default=1000)
    parser.add_argument("-num_epochs",type=int,default=500)
    parser.add_argument("-lr", help="learning rate for training", type=float, default=2e-4)
    parser.add_argument("-model_name",type=str,default="GAN_G")
    parser.add_argument("-plot",type=bool, default=False)
    parser.add_argument("-save_train_data_name",type=str, default='')
    parser.add_argument("-save_eval_data_name",type=str, default='')
    parser.add_argument("-wandb", default=True, action='store_true')
    parser.add_argument("-no_wandb", dest='wandb', action='store_false')
    parser.add_argument("-ignoreStillState", default=True, action='store_true')
    parser.add_argument("-dont_ignoreStillState", dest='ignoreStillState', action='store_false')
    args = parser.parse_args()

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset(sequence_length=args.sequence_length,epoch_size=args.train_epoch_size,save_path=args.save_train_data_name,device=device,ignoreStillState=args.ignoreStillState)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


    # wandb
    if args.wandb:
        run = wandb.init(
            project="Target-Predictor-GAN", 
            config = {"architecture":"lstm"},
            entity="aaccjjt",
            )
        wandb.config.update(args)


    input_dim, output_dim, seq_dim = train_dataset.info()
    generator = Generator(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, num_layers=args.num_layers,device=device)
    discriminator = Discriminator()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    num_params_gen = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    num_params_disc = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('Number of parameters for generator: %d and discriminator: %d' % (num_params_gen, num_params_disc))

    num_epochs = args.num_epochs # 20
    learning_rate = args.lr # 2e-4

    # GAN training can be unstable. In this case, the strong momentum
    # for the gradient prevents convergence. One possible explanation is that the
    # strong momentum does not allow the two players in the adversarial game to react
    # to each other quickly enough. Decreasing beta1 (the exponential decay for the
    # gradient moving average in [0,1], lower is faster decay) from the default 0.9
    # to 0.5 allows for quicker reactions.
    gen_optimizer = torch.optim.Adam(params=generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    disc_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # set to training mode
    generator.train()
    discriminator.train()

    gen_loss_avg = []
    disc_loss_avg = []




    print('Training ...')
    for epoch in range(num_epochs):
        gen_loss_avg.append(0)
        disc_loss_avg.append(0)
        num_batches = 0
        
        # for image_batch, _ in train_loader:
        for X, y in tqdm.tqdm(train_loader):
            
            # get dataset image and create real and fake labels for use in the loss
            X = X.to(device)
            label_real = torch.ones(X.size(0), device=device)
            label_fake = torch.zeros(X.size(0), device=device)

            # generate a batch of images from samples of the latent prior
            fake_image_batch = generator(X)
            
            # train discriminator to correctly classify real and fake
            # (detach the computation graph of the generator and the discriminator,
            # so that gradients are not backpropagated into the generator)
            real_pred = discriminator(y).squeeze()
            fake_pred = discriminator(fake_image_batch.detach()).squeeze()
            disc_loss = 0.5 * (
                F.binary_cross_entropy(real_pred, label_real) +
                F.binary_cross_entropy(fake_pred, label_fake))
            
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            
            # train generator to output an image that is classified as real
            fake_pred = discriminator(fake_image_batch).squeeze()
            gen_loss = F.binary_cross_entropy(fake_pred, label_real)
            
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()
            
            gen_loss_avg[-1] += gen_loss.item()
            disc_loss_avg[-1] += disc_loss.item()
            num_batches += 1
            
        gen_loss_avg[-1] /= num_batches
        disc_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average loss generator vs. discrim.: %f vs. %f' %
            (epoch+1, num_epochs, gen_loss_avg[-1], disc_loss_avg[-1]))

        if args.wandb: wandb.log({'gen_loss': gen_loss_avg[-1], 'disc_loss': disc_loss_avg[-1]})

    
    print("saving model")
    
    # save model
    folderPath = './SJtools/copilot/targetPredictor/models/'
    fileName = args.model_name
    PATH = folderPath+fileName+".pt"
    torch.save(generator.state_dict(), PATH)
    if args.wandb: wandb.save(PATH)


    folderPath = './SJtools/copilot/targetPredictor/models/'
    fileName = args.model_name + "D"
    PATH = folderPath+fileName+".pt"
    torch.save(discriminator.state_dict(), PATH)
    if args.wandb: wandb.save(PATH)


    # save hyperparameters
    yamldata = {
    'model': 'LSTM_POS1',
    'path': '',
    'hyperparameters': {
        'input_dim':input_dim,
        'output_dim':output_dim,
        'hidden_dim':args.hidden_dim,
        'num_layers':args.num_layers,}
    }
    yamldata['path'] = PATH
    YAML_PATH = folderPath+fileName+".yaml"
    with open(YAML_PATH, 'w') as file:
        documents = yaml.dump(yamldata, file)
    if args.wandb: wandb.save(YAML_PATH)