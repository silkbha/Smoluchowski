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
from plotting import *



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
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.input_size, self.hidden_size, dtype=torch.float64),
        #     nn.Linear(self.hidden_size, 2*self.hidden_size, dtype=torch.float64),
        #     nn.Linear(2*self.hidden_size, self.hidden_size, dtype=torch.float64),
        # )
        self.out = nn.Linear(self.hidden_size, self.output_size, dtype=torch.float64)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        # x = self.mlp(x)
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



def train(filename, output_dir, kernel, retrain=False, best=False):
    f = h5py.File(filename, "r")
    # t      = f["simple_dataset"][:,0]
    series = f["simple_dataset"][:,1:]
    print(len(series))

    test_size = int(len(series) - 2)
    # test_size = len(series) - train_size
    train,test = series[:], series[test_size:]

    lookback = 1
    X_train , y_train = create_dataset(train, lookback)
    X_test , y_test = create_dataset(test, lookback)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_train.dtype, y_train.dtype)
    print(X_test.dtype, y_test.dtype)

    model = ToyModel()
    if retrain:        
        if best:
            best_model = os.path.join(output_dir, f"best/toymodel_{kernel}.pt")
            model.load_state_dict(torch.load(best_model))
            print("Retraining best model.")
        else:
            test_model = os.path.join(output_dir, f"toymodel_{kernel}.pt")
            model.load_state_dict(torch.load(test_model))
            print("Retraining model.")


    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=128)

    n_epochs = 50000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 250 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    torch.save(model.state_dict(), output_dir+f"/toymodel_{kernel}.pt")

    return



def test(model_dir, kernel, best=False, plot=True):
    """
    """
    start = time.time()

    best_model = os.path.join(model_dir, f"best/toymodel_{kernel}.pt")
    test_model = os.path.join(model_dir, f"toymodel_{kernel}.pt")

    model = ToyModel()
    if best:
        model.load_state_dict(torch.load(best_model))
        print("Testing best model.")
    else:
        model.load_state_dict(torch.load(test_model))
        print("Testing model.")
    model.eval()

    m  = np.logspace(-12,4,100)
    analytical = np.zeros((100,100))
    lstm_results = np.zeros((100,100))
    lstm_onestep = np.zeros((100,100))

    if kernel == "constant":
        t = np.logspace(-9,3,100)
        for i,x in enumerate(t):
            analytical[i] = solution_constant_kernel(m,1.,1.,x)
    elif kernel == "linear":
        t  = np.logspace(0,1.2,100)
        for i,x in enumerate(t):
            analytical[i] = solution_linear_kernel(m,1.,1.,x)

    analytical[analytical<1e-30] = 1e-30

    initial = analytical[0]
    lstm_results[0] = np.log10(initial)
    lstm_onestep[0] = np.log10(initial)

    with torch.no_grad():
        for i in range(1,100):
            lstm_onestep[i] = model(torch.tensor(np.log10(analytical[i-1]).reshape(1,-1)))
            lstm_results[i] = model(torch.tensor(lstm_results[i-1].reshape(1,-1)))

    lstm_results = 10**(lstm_results)
    lstm_results[lstm_results<1e-30] = 0.

    lstm_onestep = 10**(lstm_onestep)
    lstm_onestep[lstm_onestep<1e-30] = 0.

    end = time.time()
    print(f"Elapsed time: {end-start} s")

    if plot:
        plot_everything(m,analytical,lstm_results, kernel, model_dir, onestep=False)
        plot_set(m,analytical,lstm_results, kernel, model_dir, onestep=False)
        
        plot_everything(m,analytical,lstm_onestep, kernel, model_dir, onestep=True)
        plot_set(m,analytical,lstm_onestep, kernel, model_dir, onestep=True)

    return



if __name__ == "__main__":
    start = time.time()

    kernel = "linear"

    src_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = str(Path(src_dir).parents[0])
    data_dir = os.path.join(main_dir, "data")
    model_dir = os.path.join(main_dir, "models")

    filename = os.path.join(data_dir, f"simple_dataset_{kernel}.h5")

    # train(filename, model_dir, kernel, retrain=True, best=True)
    test(model_dir, kernel, best=True, plot=False)

    end = time.time()
    print(f"Total elapsed time: {end-start} s")