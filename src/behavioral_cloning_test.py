import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from utils import ENN, PNN


def get_dummy_data():
    x_train = np.random.randint(0, 2, n * (52 + 2 + 318)).reshape(n, -1)
    x_train = torch.from_numpy(x_train).type(torch.float32)

    hands = torch.from_numpy(np.array(
        [np.random.choice(list(range(0, 52)), 13, replace=False) for i in range(n)]
    ))
    y_train = torch.nn.functional.one_hot(hands, -1).sum(axis=1)
    b_train = torch.from_numpy(np.random.choice(list(range(0, 37)), n, replace=True))
    # b_train = torch.nn.functional.one_hot(b_train, -1)

    return x_train, y_train, b_train


def get_file_data():
    ddd = pd.read_pickle('../INTDATA.pkl').values
    x_train = ddd[:, :372]
    y_train = ddd[:, -52:]
    b_train = ddd[:, 372]

    x_train = torch.from_numpy(x_train).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32)
    b_train = torch.from_numpy( b_train ).long()
    # b_train = torch.nn.functional.one_hot( b_train , -1 )

    return x_train, y_train, b_train

if __name__=='__main__':
    lr = 1e-4
    batch_size = 32
    num_epochs = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # x_train, y_train, b_train = get_dummy_data()
    x_train, y_train, b_train = get_file_data()
    
    model_enn = ENN().to(device)
    model_enn.load_state_dict(torch.load('model_cache/model_372_52_e5/model_enn_19.data'))
    dataset_train = torch.utils.data.TensorDataset(x_train, b_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    model_pnn = PNN().to(device)
    optimizer = torch.optim.Adam(model_pnn.parameters(), lr=lr)
    class_weights = np.ones(38)
    class_weights[0] = .1
    class_weights = torch.from_numpy(class_weights).type(torch.float32)
    writer = SummaryWriter(log_dir='.')
    for i_epoch in range(num_epochs):
        print(i_epoch)
        for x_b, b_b in tqdm(dataloader_train):
            x_b, b_b = x_b.to(device), b_b.to(device)
            enn_b = model_enn(x_b).detach()
            x2_b = torch.concat([x_b, enn_b], dim=1)
            logit_b = model_pnn(x2_b)
            loss = torch.nn.functional.cross_entropy(logit_b, b_b, weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss)
        df = pd.DataFrame({'pred': model_pnn(torch.cat([x_train, model_enn(x_train)], axis=1)).detach().numpy().argmax(axis=1), 'target': b_train})
        accuracy = df.query('pred==target').shape[0]/df.shape[0]
        accuracy_nonpass = df.query('(pred==target) and (pred!=0)').shape[0]/df.query('pred!=0').shape[0]
        print(f"Accuracy = {df.query('pred==target').shape[0]/df.shape[0] * 100:.2f} %")
        print(f"Accuracy (nonpass) = {df.query('(pred==target) and (pred!=0)').shape[0]/df.query('pred!=0').shape[0] * 100:.2f} %")
        torch.save(model_pnn.state_dict(), f"model_pnn_{i_epoch}.data")
        writer.add_scalar("Loss/train", loss, i_epoch)
        writer.add_scalar("Accuracy/train", accuracy, i_epoch)
        writer.add_scalar("Accuracy_nonpass/train", accuracy_nonpass, i_epoch)
