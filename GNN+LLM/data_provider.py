import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils import time_features

class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 target_station='부산북항', target='PM25', scale=True, timeenc=0, freq='h'):
        assert size is not None
        self.seq_len, self.label_len, self.pred_len = size

        self.target = target  # ex: 'PM25'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.target_station = target_station

        # __init__ 마지막에 함수를 불러와서 실제 데이터 로딩 및 전처리 작업 시작
        self.__read_data__()

    def __read_data__(self):
        # 데이터 불러옴
        df_raw = pd.read_pickle(os.path.join(self.root_path, self.data_path))

        # Pivot PM25 values: [timestamp x station] format
        df_pivot = df_raw.pivot(index='timestamp', columns='측정소명', values=self.target)
        self.station_list = df_pivot.columns.tolist()
        self.station2idx = {name: i for i, name in enumerate(self.station_list)}
        target_idx = self.station2idx[self.target_station] # 부산북항은 index 13


        # 추출: 위도, 경도 정적 정보
        station_meta = df_raw.drop_duplicates('측정소명')[['측정소명', 'latitude', 'longitude']]
        station_meta = station_meta.set_index('측정소명').loc[self.station_list]
        self.node_coords = station_meta[['latitude', 'longitude']].values  # shape: [N, 2]

        # 데이터 분할
        num_train = int(len(df_pivot) * 0.7)
        num_test = int(len(df_pivot) * 0.2)
        num_vali = len(df_pivot) - num_train - num_test

        border1s = {
            'train': 0,
            'val': num_train - self.seq_len,
            'test': len(df_pivot) - num_test - self.seq_len
        }
        border2s = {
            'train': num_train,
            'val': num_train + num_vali,
            'test': len(df_pivot)
        }

        border1 = border1s[self.flag]
        border2 = border2s[self.flag]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_pivot.iloc[border1s['train']:border2s['train']].values)
            data = self.scaler.transform(df_pivot.values)
        else:
            data = df_pivot.values

        self.data_x = data  # [T, N]
        self.data_y = data
        self.data_stamp = df_pivot.index.to_frame(index=False)[border1:border2]
        self.data_stamp = time_features(self.data_stamp, timeenc=self.timeenc, freq=self.freq)

        self.border1 = border1
        self.border2 = border2
        self.target_idx = target_idx

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, :]         # [seq_len, N]
        seq_y = self.data_y[r_begin:r_end, :]         # [label+pred_len, N]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.target_idx, self.node_coords

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len

# Datatset_Custom 클래스를 좀 더 편리하게 사용하기 위하 래퍼
def data_provider(args, flag):
    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        target_station=args.target_station,
        target='PM25',
        scale=True,
        timeenc=0,
        freq='h')

    # 모든 split(train, val, test)에 대해 args 설정
    args.target_idx = dataset.target_idx
    args.node_coords = dataset.node_coords

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(flag == 'train'))
    return data_loader