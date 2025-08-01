import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from data_provider import data_provider
from utils import EarlyStopping, metric

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--llm_model', type=str, default='GPT2')
parser.add_argument('--llm_layers', type=int, default=4)
parser.add_argument('--llm_dim', type=int, default=768)
parser.add_argument('--enc_in', type=int, default=1)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=24)
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--prompt_domain', action='store_true')
parser.add_argument('--content', type=str, default='부산 북항 대기오염 예측을 위한 시계열 입력')
parser.add_argument('--root_path', type=str, default='./data')
parser.add_argument('--data_path', type=str, default='bukhang_single_station.pkl')
args = parser.parse_args()

setting = f"{args.task_name}_{args.llm_model}_{args.seq_len}_{args.pred_len}"

train_loader = data_provider(args, flag='train')
vali_loader = data_provider(args, flag='val')
test_loader = data_provider(args, flag='test')

model = Model(args).to(args.device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
early_stopping = EarlyStopping(patience=5, verbose=True, path=f"./checkpoints/{setting}.pt")

for epoch in range(args.train_epochs):
    model.train()
    train_loss = []
    for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(args.device), batch_y_mark.to(args.device)

        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            batch_x_mark, batch_y_mark = batch_x_mark.to(args.device), batch_y_mark.to(args.device)
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
            val_loss.append(loss.item())

    print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.4f} | Val Loss: {np.mean(val_loss):.4f}")
    early_stopping(np.mean(val_loss), model)
    if early_stopping.early_stop:
        break

# Load best model and test
model.load_state_dict(torch.load(f"./checkpoints/{setting}.pt"))
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(args.device), batch_y_mark.to(args.device)
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        preds.append(outputs.cpu().numpy())
        trues.append(batch_y[:, -args.pred_len:, :].cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
mae, mse, rmse, mape, mspe = metric(preds, trues)
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}")