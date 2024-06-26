import time
import os
from pathlib import Path
import h5py

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data

from coag_kernels import *

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.linewidth'] = 1.25



class ToyModel(nn.Module):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.input_size = 100
        self.hidden_size = 275
        self.output_size = self.input_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            num_layers = 1,
                            dtype=torch.float64
                            )
        self.out = nn.Linear(self.hidden_size, self.output_size, dtype=torch.float64)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.out(x)
        return x

def create_dataset(dataset, lookback):
    """ Transform a time series into a prediction dataset.
        Source: https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        
        Args:
            dataset: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    X,y = np.array(X), np.array(y)
    return torch.tensor(X), torch.tensor(y)

def train(filename, output_dir):
    f = h5py.File(filename, "r")
    # t      = f["simple_dataset"][:,0]
    series = f["simple_dataset"][:,1:]
    print(len(series))

    train_size = int(len(series) * 0.67)
    # test_size = len(series) - train_size
    train,test = series[:train_size], series[train_size:]

    lookback = 1
    X_train , y_train = create_dataset(train, lookback)
    X_test , y_test = create_dataset(test, lookback)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_train.dtype, y_train.dtype)
    print(X_test.dtype, y_test.dtype)

    model = ToyModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=8)

    n_epochs = 10000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 50 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    torch.save(model.state_dict(), output_dir+"/toymodel.pt")

    return


def test(model_dir, best=False):
    """
    """
    best_model = os.path.join(model_dir, "best/toymodel.pt")
    test_model = os.path.join(model_dir, "toymodel.pt")

    model = ToyModel()
    if best:
        model.load_state_dict(torch.load(best_model))
        print("Testing best model.")
    else:
        model.load_state_dict(torch.load(test_model))
    model.eval()

    m  = np.logspace(-12,4,100)
    t  = np.logspace(-9,3,100)
    analytical = np.zeros((100,100))
    lstm_results = np.zeros((100,100))

    for i,x in enumerate(t):
        analytical[i] = solution_constant_kernel(m,1.,1.,x)
    analytical[analytical<1e-30] = 1e-30

    initial = analytical[0]
    lstm_results[0] = np.log10(initial)

    with torch.no_grad():
        for i in range(65,100):
            lstm_results[i] = model(torch.tensor(np.log10(analytical[i-1]).reshape(1,-1)))
            # lstm_results[i] = model(torch.tensor(lstm_results[i-1].reshape(1,-1)))

    lstm_results = 10**(lstm_results)
    lstm_results[lstm_results<1e-30] = 0.

    for idx in range(65,100):
        fig, ax = plt.subplots(1,1, figsize=(7,5))

        ax.loglog(m, analytical[idx], c="k", lw=1, ls="-.")
        ax.loglog(m, lstm_results[idx], c="b")

        # ax.set_xlim(m[0, 0], m[0, -1])
        ax.set_ylim(1.e-30, 1.e3)
        ax.set_xlabel(r"$m$", math_fontfamily='dejavuserif')
        ax.set_ylabel(r"$N\,\left(m,t\right)\,\cdot\,m^2$", math_fontfamily='dejavuserif')
        ax.set_title(f"Neural Network -- Constant Kernel -- t = {idx} / 100", math_fontfamily='dejavuserif')
        fig.tight_layout()

        imgname = f"plots/toymodel_t{idx}.png"
        plt.savefig(os.path.join(model_dir,imgname))
        plt.close()
    # plt.show()


    return


if __name__ == "__main__":
    start = time.time()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = str(Path(src_dir).parents[0])
    data_dir = os.path.join(main_dir, "data")
    model_dir = os.path.join(main_dir, "models")

    filename = os.path.join(data_dir, "simple_dataset.h5")

    # train(filename, model_dir)
    test(model_dir)

    end = time.time()
    print(f"Elapsed time: {end-start} s")