import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from data_provider import data_provider
from utils import EarlyStopping, metric
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, BertModel, BertTokenizer, BertConfig, LlamaModel, LlamaTokenizer, LlamaConfig
from baseline.Dlinear import Dlinear  # Dlinear import 추가
import os

os.makedirs("./checkpoints", exist_ok=True)

import random

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for result file naming')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--task_name', type=str, default='long_term_forecast')
parser.add_argument('--llm_model_dir', type=str, default='/ERC/models/gpt2/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e')
parser.add_argument('--llm_model', type=str, default='GPT2')
parser.add_argument('--llm_layers', type=int, default=4)
parser.add_argument('--llm_dim', type=int, default=768)
parser.add_argument('--enc_in', type=int, default=1)
parser.add_argument('--d_model', type=int, default =64)
parser.add_argument('--d_ff', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=12)
parser.add_argument('--pred_len', type=int, default=12) 
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--prompt_domain', action='store_true')
parser.add_argument('--content', type=str, default='부산 북항 대기오염 예측을 위한 시계열 입력')
parser.add_argument('--root_path', type=str, default='/ERC/data')
parser.add_argument('--data_path', type=str, default='pollutant.pkl')
parser.add_argument('--target_station', type=str, default='부산북항')
parser.add_argument('--model', type=str, default='GNNLLM')
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--num_class', type=int, default=1)
args = parser.parse_args()

fix_seed(args.seed)

# 데이터 준비
train_loader = data_provider(args, flag='train')
vali_loader = data_provider(args, flag='val')
test_loader = data_provider(args, flag='test')

if args.model == 'GNNLLM' : 
    model = GNNLLM(args).to(args.device)
elif args.model == 'TimeLLM' :
    model = TimeLLM(args).to(args.device)
elif args.model == 'Dlinear':
    model = Dlinear(args).to(args.device)  # args가 configs 역할을 함
else:
    raise ValueError(f"Unsupported model: {args.model}")
    
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
checkpoint_name = f"{args.task_name}_{args.model}_{args.llm_model}_seed{args.seed}_bs{args.batch_size}_sl{args.seq_len}_pl{args.pred_len}.pt"
early_stopping = EarlyStopping(patience=7, verbose=True, path=f"./checkpoints/{checkpoint_name}")

target_idx = args.target_idx

for epoch in range(args.train_epochs):
    model.train()
    train_loss = []
    for batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ in train_loader:
        batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(args.device), batch_y_mark.to(args.device)

        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        loss = criterion(outputs[:, :, target_idx], batch_y[:, -args.pred_len:, target_idx])
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ in vali_loader:
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            batch_x_mark, batch_y_mark = batch_x_mark.to(args.device), batch_y_mark.to(args.device)
            outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            loss = criterion(outputs[:, :, target_idx], batch_y[:, -args.pred_len:, target_idx])
            val_loss.append(loss.item())

    print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_loss):.4f} | Val Loss: {np.mean(val_loss):.4f}")
    early_stopping(np.mean(val_loss), model)
    if early_stopping.early_stop:
        break

# Load best model and test
model.load_state_dict(torch.load(f"./checkpoints/{checkpoint_name}"))
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ in test_loader:
        batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
        batch_x_mark, batch_y_mark = batch_x_mark.to(args.device), batch_y_mark.to(args.device)
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
        preds.append(outputs[:, :, target_idx].cpu().numpy())
        trues.append(batch_y[:, -args.pred_len:, target_idx].cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
mae, mse, rmse, mape, mspe = metric(preds, trues)
print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}")

import pandas as pd
import os

def save_result_to_excel(args, mae, rmse, mape):
    filename = f'results_gpu{args.gpu_id}.xlsx'
    sheet = args.model  # 예: TimeLLM, GNNLLM 등

    row = {
        'model': args.model,
        'llm_layers': args.llm_layers,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'patch_len': args.patch_len,
        'stride': args.stride,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

    if os.path.exists(filename):
        try:
            existing = pd.read_excel(filename, sheet_name=sheet)
            df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet, index=False)
        except ValueError:
            # 시트가 없는 경우 새로 저장
            with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
                pd.DataFrame([row]).to_excel(writer, sheet_name=sheet, index=False)
    else:
        # 파일이 없는 경우 새로 작성 (mode='w', if_sheet_exists 생략)
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
            pd.DataFrame([row]).to_excel(writer, sheet_name=sheet, index=False)

# → 마지막 성능 출력 아래에 추가
save_result_to_excel(args, mae, rmse, mape)