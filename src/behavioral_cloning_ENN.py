import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import  ENN, PNN
from torch.utils.tensorboard import SummaryWriter

def get_dummy_data():
    x_train = np.random.randint(0, 2, n * (52 + 2 + 318)).reshape(n, -1)
    x_train = torch.from_numpy(x_train).type(torch.float32)

    hands = torch.from_numpy(np.array(
        [np.random.choice(list(range(0, 52)), 13, replace=False) for i in range(n)]
    ))
    y_train = torch.nn.functional.one_hot(hands, -1).sum(axis=1)

    b_train = torch.nn.functional.one_hot(
        torch.from_numpy(np.random.choice(list(range(0, 37)), n, replace=True)), -1)

    return x_train, y_train, b_train


def get_file_data():
    ddd = pd.read_pickle('INTDATA.pkl').values
    x_train = ddd[:, :372]
    y_train = ddd[:, -52:]
    b_train = ddd[:, 372]

    x_train = torch.from_numpy(x_train).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    b_train = torch.nn.functional.one_hot( torch.from_numpy( b_train ).long(), -1 )

    return x_train, y_train, b_train

if __name__=='__main__':
    lr = 1e-5
    batch_size = 32
    num_epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # x_train, y_train, b_train = get_dummy_data()
    x_train, y_train, b_train = get_file_data()
    
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    model_enn = ENN().to(device)
    optimizer = torch.optim.Adam(model_enn.parameters(), lr=lr)
    writer = SummaryWriter(log_dir='.')
    for i_epoch in range(num_epochs):
        print(i_epoch)
        losses = []
        for x_b, y_b in tqdm(dataloader_train):
            x_b, y_b = x_b.to(device), y_b.to(device)
            logit_b = model_enn(x_b)
            # loss = -(y_b * logit_b.log() + (1-y_b) * (1-logit_b).log()).mean() 
            loss = torch.nn.functional.binary_cross_entropy(logit_b, y_b)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("Loss/train", loss, i_epoch)
            losses.append(loss.detach().numpy())
        print(np.mean(losses))
        torch.save(model_enn.state_dict(), f"model_enn_{i_epoch}.data")

    
    dataset_train = torch.utils.data.TensorDataset(x_train, b_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    model_pnn = PNN().to(device)
    optimizer = torch.optim.Adam(model_pnn.parameters(), lr=lr)
    for i_epoch in range(num_epochs):
        print(i_epoch)
        for x_b, b_b in tqdm(dataloader_train):
            x_b, b_b = x_b.to(device), b_b.to(device)
            enn_b = model_enn(x_b).detach()
            x2_b = torch.concat([x_b, enn_b], dim=1)
            logit_b = model_pnn(x2_b)
            loss = torch.nn.functional.cross_entropy(logit_b, b_b.type(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss)
