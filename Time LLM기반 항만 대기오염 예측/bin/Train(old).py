import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import *  # 너가 만든 Time-LLM 모델 파일
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    # Task 설정
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--llm_model', type=str, default='GPT2', choices=['GPT2', 'LLAMA', 'BERT'])
    parser.add_argument('--llm_dim', type=int, default=768)  # GPT2: 768, LLAMA7B: 4096

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
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)

    # Prompt
    parser.add_argument('--prompt_domain', type=int, default=0)
    parser.add_argument('--content', type=str, default='This is electricity transformer data.')

    return parser.parse_args()


def generate_dummy_data(num_samples=256, seq_len=96, pred_len=96, num_features=7):
    x = torch.randn(num_samples, seq_len, num_features)
    y = torch.randn(num_samples, pred_len, num_features)
    return x, y


def train():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 데이터 준비
    x, y = generate_dummy_data(num_samples=512, seq_len=args.seq_len, pred_len=args.pred_len, num_features=args.enc_in)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)

    # 2. 모델 준비
    model = Model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 3. 학습 루프
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            dec_inp = torch.zeros_like(batch_y).to(device)

            optimizer.zero_grad()
            out = model(batch_x, None, dec_inp, None)  # forward()의 네 인자
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.6f}")


if __name__ == '__main__':
    train()