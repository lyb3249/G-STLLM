import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def metric(pred, true):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / (true + 1e-5)))
    mspe = np.mean(np.square((pred - true) / (true + 1e-5)))
    return mae, mse, rmse, mape, mspe


def time_features(df_stamp, timeenc=0, freq='h'):
    df_stamp['month'] = df_stamp['timestamp'].dt.month
    df_stamp['day'] = df_stamp['timestamp'].dt.day
    df_stamp['weekday'] = df_stamp['timestamp'].dt.weekday
    df_stamp['hour'] = df_stamp['timestamp'].dt.hour
    return df_stamp.drop(['timestamp'], axis=1).values.astype(np.float32)