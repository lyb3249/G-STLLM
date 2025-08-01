import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from model import Model  # 너가 만든 Time-LLM 모델 파일


def get_args():
    parser = argparse.ArgumentParser()

    # Task 설정
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--llm_model', type=str, default='GPT2', choices=['GPT2', 'LLAMA', 'BERT'])
    parser.add_argument('--llm_dim', type=int, default=768)  # GPT2 기준

    # PatchEmbedding
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=64)

    # LLM 관련
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    # 데이터 관련
    parser.add_argument('--enc_in', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # Prompt
    parser.add_argument('--prompt_domain', type=int, default=1)
    parser.add_argument('--content', type=str,
                        default='This is air pollution monitoring data including SO2, CO, O3, NO2, PM10, PM2.5.')

    # 파일 경로
    parser.add_argument('--data_path', type=str, default='/Users/bini/Library/Mobile Documents/com~apple~CloudDocs/Industrial_LAB/GNN+LLM 항만 대기오염 예측/data/pollutant_adj_sensor_impute.pkl')
    parser.add_argument('--save_path', type=str, default='checkpoints/best_model.pth')

    return parser.parse_args()


def load_real_data(filepath, seq_len, pred_len):
    df = pd.read_pickle(filepath)
    df = df.sort_values(by='timestamp')

    features = ['SO2', 'CO', 'O3', 'NO2', 'PM10', 'PM25']
    df = df[features].dropna()

    df = (df - df.mean()) / df.std()

    data = df.values
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pred_len])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def train():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    x_all, y_all = load_real_data(args.data_path, args.seq_len, args.pred_len)

    val_size = int(len(x_all) * args.val_ratio)
    train_size = len(x_all) - val_size
    train_x, val_x = torch.utils.data.random_split(x_all, [train_size, val_size])
    train_y, val_y = torch.utils.data.random_split(y_all, [train_size, val_size])

    train_loader = DataLoader(TensorDataset(train_x.dataset[train_x.indices], train_y.dataset[train_y.indices]),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x.dataset[val_x.indices], val_y.dataset[val_y.indices]),
                            batch_size=args.batch_size, shuffle=False)

    # 2. Model
    model = Model(args).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # 3. Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            dec_inp = torch.zeros_like(batch_y).to(device)

            optimizer.zero_grad()
            out = model(batch_x, None, dec_inp, None)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 4. Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                dec_inp = torch.zeros_like(val_y).to(device)
                out = model(val_x, None, dec_inp, None)
                loss = criterion(out, val_y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 5. Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f">>> Saved best model to {args.save_path}")


if __name__ == '__main__':
    train()
