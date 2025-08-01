import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import time_features

class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='S', target='PM25', scale=True, timeenc=0, freq='h'):
        assert size is not None
        self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_pickle(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = {'train': 0, 'val': num_train - self.seq_len, 'test': len(df_raw) - num_test - self.seq_len}
        border2s = {'train': num_train, 'val': num_train + num_vali, 'test': len(df_raw)}

        border1 = border1s[self.flag]
        border2 = border2s[self.flag]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data.iloc[border1s['train']:border2s['train']].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data
        self.data_y = data
        self.data_stamp = df_raw[['timestamp']][border1:border2]
        self.data_stamp = time_features(self.data_stamp, timeenc=self.timeenc, freq=self.freq)

        self.border1 = border1
        self.border2 = border2

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len

def data_provider(args, flag):
    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features='S',
        target='PM25',
        scale=True,
        timeenc=0,
        freq='h')

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(flag=='train'))
    return data_loader