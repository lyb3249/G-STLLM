import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from torch import Tensor
from torch_geometric.nn import GATConv

# 정규화를 위한 class
class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

# 입력 시퀀스의 각 포인트를 고정된 차원의 벡터로 임베딩
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # conv 연산을 이용하여 임베딩을 만들 때 자기 자신과 바로 양옆의 이웃 정보를 함께 고려
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# end 부분을 마지막 원소 값으로 복제하여 채우는 역할
class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output

# 시계열 데이터를 patch 단위로 분할하고, embedding 처리
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1] # 복원을 위해 변수 수 저장
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars

# patch와 text prototypes를 attention을 통해 patch embeddings 생성
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

# LLM의 output을 받아 최종 예측값 생성
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# 그래프 모듈 추가
class GraphModule(nn.Module):
    def __init__(self, node_feat_dim, gnn_dim, edge_index):
        super(GraphModule, self).__init__()
        self.gnn = GATConv(node_feat_dim, gnn_dim)
        self.edge_index = edge_index  # shape: [2, E]

    def forward(self, node_input):  # node_input: [B, N, F]
        batch_graph_embeddings = []
        for b in range(node_input.size(0)):
            h = self.gnn(node_input[b], self.edge_index.to(node_input.device))  # [N, D]
            h_graph = h.mean(dim=0)  # Graph-level pooling → [D] "일단은 부산북항 노드를 출력하는게 아니라 pooling 했음"
            batch_graph_embeddings.append(h_graph)
        return torch.stack(batch_graph_embeddings)  # [B, D]

# 거리기반 인접행렬 생성함수
def build_edge_index(coords, k=4):
    """
    coords: numpy array of shape [N, 2] (latitude, longitude)
    k: number of nearest neighbors
    return: torch.LongTensor of shape [2, E] for PyG GATConv
    """
    from sklearn.metrics.pairwise import haversine_distances
    from math import radians

    coords_rad = np.radians(coords)  # 위경도를 라디안으로 변환
    dist_matrix = haversine_distances(coords_rad)  # [N, N], 단위: radians

    edge_index = []
    for i in range(coords.shape[0]):
        nearest = np.argsort(dist_matrix[i])[1:k+1]  # 자기 자신 제외 후 k개 선택
        for j in nearest:
            edge_index.append([i, j])

    edge_index = torch.LongTensor(edge_index).T  # [2, E] 형태로 변환
    return edge_index

# GNN+LLM 모델
class GNNLLM(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(GNNLLM, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len # 앞으로 예측할 기간
        self.seq_len = configs.seq_len # 이전에 보는 기간
        self.d_ff = configs.d_ff #함수의 차원
        self.top_k = 5 #가장 높은 상관관계를 가지는 lag top_K개
        self.d_llm = configs.llm_dim #llm의 차원(LLama7b:4096; GPT2-small:768; BERT-base:768)
        self.patch_len = configs.patch_len #patch의 길이
        self.stride = configs.stride #patch를 다시 생성하는데 몇개의 Timestamp를 건너뛸건지

        # GNN 구성
        self.node_coords = torch.tensor(configs.node_coords, dtype=torch.float32)  # [N, 2]
        edge_index = build_edge_index(self.node_coords.numpy(), k=4)
        self.graph_module = GraphModule(node_feat_dim=3, gnn_dim=self.d_llm, edge_index=edge_index)

        #HuggingFace 모델 허브에서 LLAMA, GPT2, BERT 중 하나를 로드
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_model_dir)
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            self.llm_model = GPT2Model.from_pretrained(configs.llm_model_dir, config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_model_dir)
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        #토크나이저 설정 및 PAD 토큰 처리(eos_token이 있으면 그걸 padding tokend으로 사용하고 없으면 만들어주기)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        #LLM은 얼린 상태로 설정
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 자연어 프롬프트 처음에 도메인에 대해 설명할 내용
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'This dataset contains hourly air pollution data collected from the Busan Bukhang monitoring station.'


        self.dropout = nn.Dropout(configs.dropout)

        # Patch embedding 정의(패치로 쪼개고, 각 패치를 임베딩하는 모듈)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # LLM 임베딩 관련 정의
        self.word_embeddings = self.llm_model.get_input_embeddings().weight #word_embedding 행렬을 불러옴
        self.vocab_size = self.word_embeddings.shape[0] # 안에 있는 단어의 개수 확인. 즉 차원 정보 저장
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens) #차원 축소(vocab_size -> num_tokens)

        # Reprogramming Layer 정의
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # patch 개수 및 출력 head 정의 (출력 FlattenHead에서 input 차원 계산용)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        #표준화 및 역표준화
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # target 관측소 index
        self.target_idx = configs.target_idx

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()

        # GNN 처리
        x_node = x_enc.permute(0, 2, 1)          # [B, N, T]
        x_summary = x_node.mean(dim=-1)          # [B, N]
        static = self.node_coords.unsqueeze(0).expand(B, -1, -1).to(x_enc.device)  # [B, N, 2]
        node_input = torch.cat([x_summary.unsqueeze(-1), static], dim=-1)          # [B, N, 3]
        node_input = node_input.float()
        graph_embed = self.graph_module(node_input)                                # [B, d_llm].

        # Target 관측소 시계열만 사용
        x_target = x_node[:, self.target_idx, :]  # [B, T]

        # Prompt 생성
        min_values = x_target.min(dim=1)[0]
        max_values = x_target.max(dim=1)[0]
        medians = x_target.median(dim=1).values
        lags = self.calcute_lags(x_target.unsqueeze(1))  # [B, 1, T]
        trends = x_target.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(B):
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
                f"Input statistics: min value {min_values[b].item()}, max value {max_values[b].item()}, "
                f"median value {medians[b].item()}, the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags[b].tolist()}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device))  # [B, P, d_llm]

        # Patch & Reprogramming (target 관측소만)
        x_target = x_target.unsqueeze(1)  # [B, 1, T]
        enc_out, n_vars = self.patch_embedding(x_target.float())
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # LLM 입력 구성
        graph_token = graph_embed.unsqueeze(1)  # [B, 1, d_llm]
        llama_input = torch.cat([graph_token, prompt_embeddings,  enc_out], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_input).last_hidden_state[:, :, :self.d_ff]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2)
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1)

        return self.normalize_layers(dec_out, 'denorm')

    # def calcute_lags(self, x_enc):
    #     print("x_enx.shape before FFT:",x_enc.shape)
    #     q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    #     print("q_fft.shape before FFT:",q_fft.shape)
    #     k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    #     res = q_fft * torch.conj(k_fft)
    #     print("res.shape before FFT:",res.shape)
    #     corr = torch.fft.irfft(res, dim=-1)
    #     mean_value = torch.mean(corr, dim=1)
    #     _, lags = torch.topk(mean_value, self.top_k, dim=-1)
    #     return lags

    def calcute_lags(self, x_enc):  # x_enc: [B, 1, T]
        B, C, T = x_enc.shape
    
        x = x_enc.squeeze(1)  # [B, T]
    
        if x.shape[1] < 2:
            print("T too short for FFT")
            return torch.zeros((B, self.top_k), dtype=torch.long, device=x.device)
    
        q_fft = torch.fft.rfft(x, dim=-1)
        k_fft = torch.fft.rfft(x, dim=-1)
        res = q_fft * torch.conj(k_fft)
    
        corr = torch.fft.irfft(res, dim=-1)  # [B, T]
        mean_value = corr  # 이미 평균된 벡터
    
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

class TimeLLM(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(TimeLLM, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len # 앞으로 예측할 기간
        self.seq_len = configs.seq_len # 이전에 보는 기간
        self.d_ff = configs.d_ff #함수의 차원
        self.top_k = 5 #가장 높은 상관관계를 가지는 lag top_K개
        self.d_llm = configs.llm_dim #llm의 차원(LLama7b:4096; GPT2-small:768; BERT-base:768)
        self.patch_len = configs.patch_len #patch의 길이
        self.stride = configs.stride #patch를 다시 생성하는데 몇개의 Timestamp를 건너뛸건지

        #HuggingFace 모델 허브에서 LLAMA, GPT2, BERT 중 하나를 로드
        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(configs.llm_model_dir)
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True

            self.llm_model = GPT2Model.from_pretrained(configs.llm_model_dir, config=self.gpt2_config)
            self.tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_model_dir)
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        #토크나이저 설정 및 PAD 토큰 처리(eos_token이 있으면 그걸 padding tokend으로 사용하고 없으면 만들어주기)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        #LLM은 얼린 상태로 설정
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # 자연어 프롬프트 처음에 도메인에 대해 설명할 내용
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'This dataset contains hourly air pollution data collected from the Busan Bukhang monitoring station.'


        self.dropout = nn.Dropout(configs.dropout)

        # Patch embedding 정의(패치로 쪼개고, 각 패치를 임베딩하는 모듈)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        # LLM 임베딩 관련 정의
        self.word_embeddings = self.llm_model.get_input_embeddings().weight #word_embedding 행렬을 불러옴
        self.vocab_size = self.word_embeddings.shape[0] # 안에 있는 단어의 개수 확인. 즉 차원 정보 저장
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens) #차원 축소(vocab_size -> num_tokens)

        # Reprogramming Layer 정의
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # patch 개수 및 출력 head 정의 (출력 FlattenHead에서 input 차원 계산용)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        #표준화 및 역표준화
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # target 관측소 index
        self.target_idx = configs.target_idx

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()

        # GNN 처리
        x_node = x_enc.permute(0, 2, 1)          # [B, N, T]

        # Target 관측소 시계열만 사용
        x_target = x_node[:, self.target_idx, :]  # [B, T]

        # Prompt 생성
        min_values = x_target.min(dim=1)[0]
        max_values = x_target.max(dim=1)[0]
        medians = x_target.median(dim=1).values
        lags = self.calcute_lags(x_target.unsqueeze(1))  # [B, 1, T]
        trends = x_target.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(B):
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
                f"Input statistics: min value {min_values[b].item()}, max value {max_values[b].item()}, "
                f"median value {medians[b].item()}, the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags[b].tolist()}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                    max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_enc.device))  # [B, P, d_llm]

        # Patch & Reprogramming (target 관측소만)
        x_target = x_target.unsqueeze(1)  # [B, 1, T]
        enc_out, n_vars = self.patch_embedding(x_target.float())
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        llama_input = torch.cat([prompt_embeddings, enc_out], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_input).last_hidden_state[:, :, :self.d_ff]
        dec_out = dec_out.reshape(-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2)
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1)

        return self.normalize_layers(dec_out, 'denorm')

    def calcute_lags(self, x_enc):  # x_enc: [B, 1, T]
        B, C, T = x_enc.shape

        x = x_enc.squeeze(1)  # [B, T]

        if x.shape[1] < 2:
            print("T too short for FFT")
            return torch.zeros((B, self.top_k), dtype=torch.long, device=x.device)

        q_fft = torch.fft.rfft(x, dim=-1)
        k_fft = torch.fft.rfft(x, dim=-1)
        res = q_fft * torch.conj(k_fft)

        corr = torch.fft.irfft(res, dim=-1)  # [B, T]
        mean_value = corr  # 이미 평균된 벡터

        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

