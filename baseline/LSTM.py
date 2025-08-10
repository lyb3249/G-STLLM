import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from utils import EarlyStopping, metric
import os

# ==== 1. 데이터 로드 ====
data_path = r"C:\Users\user\Desktop\YBL\GNN+LLM 항만 대기오염 예측\data\최종피클파일\22_pollutant_신선대.pkl"
df = pd.read_pickle(data_path)

# PM25만 사용
target_col = "PM25"
values = df[target_col].values.astype(np.float32)

# ==== 2. 데이터셋 정의 ====
class SeqDataset(Dataset):
    def __init__(self, data, label_len, pred_len):
        self.data = data
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.label_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq_x = self.data[idx: idx + self.label_len]
        seq_y = self.data[idx + self.label_len: idx + self.label_len + self.pred_len]
        return torch.tensor(seq_x).unsqueeze(-1), torch.tensor(seq_y).unsqueeze(-1)

label_len = 24   # 예: 6시간(15분 간격이면 24 step)
pred_len = 24     # 예: 1.5시간
dataset = SeqDataset(values, label_len, pred_len)

train_size = int(len(dataset) * 0.7)
val_size = int(len(dataset) * 0.2)
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==== 3. LSTM 모델 정의 ====
class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_len=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)             # (batch, seq_len, hidden)
        out = out[:, -1, :]               # 마지막 타임스텝 출력
        out = self.fc(out)                # (batch, pred_len)
        return out.unsqueeze(-1)          # (batch, pred_len, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecast(pred_len=pred_len).to(device)

# ==== 4. 학습 설정 ====
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # val_loss 최소화가 목표
    factor=0.5,       # lr을 절반으로 줄임
    patience=3       # 3번 연속 개선 없으면 발동
)

early_stopping = EarlyStopping(patience=5, verbose=True, path=f"./checkpoints/LSTM.pt")

# ==== 5. 학습 루프 ====
for epoch in range(20):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}")

# ==== 6. 테스트 ====
model.load_state_dict(torch.load(f"./checkpoints/LSTM.pt"))
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        predictions.append(output.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

print("테스트 MSE:", np.mean((predictions - actuals)**2))