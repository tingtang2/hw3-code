import sys

import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm

from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd

from model import VAE
torch.manual_seed(11182022)

class notMNISTDataset(Dataset):
    '''
        torch dataset class for easy batching of notMNIST dataset
    '''
    def __init__(self, img_data, labels):
        self.img_data = img_data 
        self.labels = labels 

    def __getitem__(self, idx):
        return self.img_data[idx]/255.0, self.labels[idx] # set images values to be between 0 and 1

    def __len__(self):
        return len(self.labels)

def create_dataloaders(device, batch_size=128):
    img_data = np.load('/home/tingchen/notMNIST_small/images.npy')
    label_data = np.load('/home/tingchen/notMNIST_small/labels_singular.npy')
    
    X_train, X_val, y_train, y_val = train_test_split(img_data, label_data, test_size=0.20, random_state=42)
    print(f'train set size: {X_train.shape}, val set size: {X_val.shape}')

    # set up dataset objects
    train_dataset = notMNISTDataset(torch.from_numpy(X_train.astype(np.float32)).to(device), torch.from_numpy(y_train).to(device))
    valid_dataset = notMNISTDataset(torch.from_numpy(X_val.astype(np.float32)).to(device), torch.from_numpy(y_val).to(device))

    # set up data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)

    return train_loader, valid_loader

def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for i, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        
        x_hat = model(x)
        loss = criterion(x_hat, x) - model.kl
        
        loss.backward()
        running_loss += loss.item()

        optimizer.step()

    return running_loss/(len(loader) * loader.batch_size) 

def plot_latent(model, loader):
    model.eval()

    labels = [] 
    z_s = []   
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            z, mu, sigma = model(x, encode=True)
            z_s.append(z.cpu().detach().numpy())
            
            labels += y.cpu().numpy().tolist()

    z_s = np.vstack(z_s)
    pd_dict = {'first dimension': z_s[:, 0], 'second dimension': z_s[:, 1], 'labels': labels}

    df = pd.DataFrame(pd_dict)
    df['labels'] = df['labels'].astype(str)
    
    fig = px.scatter(df, x='first dimension', y='second dimension', color='labels', title='Training set projected into learned latent space')
    fig.write_html('latent_space.html')



def plot_reconstructed(model, r0=(-5, 10), r1=(-10, 5), n=12, device=None):
    w = 28
    img = np.zeros((n*w, n*w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = model(z, decode=True)
            x_hat = x_hat.reshape(w, w).cpu().detach().numpy()

            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat

    fig = px.imshow(img, color_continuous_scale=px.colors.sequential.Electric)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.write_html('reconstruction.html')


def plot_line(x, y, x_label, y_label, title):
    df = pd.DataFrame({y_label: y, x_label: x})
    fig = px.line(df, x=x_label, y=y_label, title=title)
    fig.write_html(f'{x_label}vs{y_label}.html')

def compute_predictive_ELBO(model, loader, criterion) -> float:
    predictive_ELBO = 0.0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x_hat = model(x)
            ELBO_batch = -criterion(x_hat, x) + model.kl
            
            predictive_ELBO += ELBO_batch.item()


    return predictive_ELBO/(len(loader) * loader.batch_size) 


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    train_loader, valid_loader = create_dataloaders(device)
    
    vae = VAE(n_latent_dims=2).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), amsgrad=True)
    criterion = MSELoss(reduction='sum')

    ELBO = []
    predictive_ELBO = []

    for epoch in trange(200):
        ELBO.append(-train(vae, train_loader, optimizer, criterion))
        predictive_ELBO.append(compute_predictive_ELBO(vae, valid_loader, criterion))
        print(f'ELBO: {ELBO[-1]}, predictive ELBO: {predictive_ELBO[-1]}')
    
    # plot_line(np.arange(len(ELBO)), ELBO, 'Iterations', 'ELBO', 'ELBO as a function of iterations')
    # plot_line(np.arange(len(predictive_ELBO)), predictive_ELBO, 'Iterations', 'Predictive ELBO', 'Predictive ELBO as a function of iterations')

    # plot both predictive and training elbo
    # df = pd.DataFrame({'epoch': 2 * [i for i in range(len(ELBO))], 'ELBO': ELBO + predictive_ELBO, 'type': ['training' for i in range(len(ELBO))] + ['predictive' for i in range(len(ELBO))]})
    # fig = px.line(df, x='epoch', y='ELBO', color='type', title='ELBOs as a Functions of Iteration')
    # fig.write_html('epochs_vs_elbos.html')


    # plot_latent(vae, train_loader)
    plot_reconstructed(vae, r0=(-4, 4), r1=(-4, 4), device=device)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())