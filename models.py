import torch
from torch import nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
import numpy as np
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
from os.path import join, expanduser, basename, splitext, basename, exists
import os
from glob import glob
from sklearn.cluster import KMeans
import random

mgc_dim = 180  # メルケプストラム次数　？？
lf0_dim = 3  # 対数fo　？？ なんで次元が３？
vuv_dim = 1  # 無声or 有声フラグ　？？
bap_dim = 15  # 発話ごと非周期成分　？？

duration_linguistic_dim = 438  # question_jp.hed で、ラベルに対する言語特徴量をルールベースで記述してる
acoustic_linguisic_dim = 442  # 上のやつ+frame_features とは？？
duration_dim = 1
acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim  # aoustice modelで求めたいもの

mgc_start_idx = 0
lf0_start_idx = 180
vuv_start_idx = 183
bap_start_idx = 184

windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]


device = "cuda" if torch.cuda.is_available() else "cpu"

hidden_num = 511


class VQVAE(nn.Module):
    def __init__(
        self, bidirectional=True, num_layers=2, num_class=2, z_dim=1, dropout=0.15, input_linguistic_dim=acoustic_linguisic_dim
    ):
        super(VQVAE, self).__init__()
        self.num_layers = num_layers
        self.num_direction = 2 if bidirectional else 1
        self.num_class = num_class
        self.quantized_vectors = nn.Embedding(
            num_class, z_dim
        )  # torch.tensor([[i]*z_dim for i in range(nc)], requires_grad=True)
        # self.quantized_vectors.weight.data.uniform_(0, 1)
        self.quantized_vectors.weight = nn.init.normal_(
            self.quantized_vectors.weight, 0.1, 0.001
        )

        self.z_dim = z_dim

        self.fc11 = nn.Linear(
            input_linguistic_dim + acoustic_dim, input_linguistic_dim + acoustic_dim - 2
        )
        self.lstm1 = nn.LSTM(
            input_linguistic_dim + acoustic_dim,
            hidden_num,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )  # 入力サイズはここできまる
        self.fc2 = nn.Linear(self.num_direction * hidden_num + 2, z_dim)
        ##ここまでエンコーダ

        self.fc12 = nn.Linear(
            input_linguistic_dim + z_dim, input_linguistic_dim + z_dim - 2
        )
        self.lstm2 = nn.LSTM(
            input_linguistic_dim + z_dim,
            hidden_num,
            num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc3 = nn.Linear(self.num_direction * hidden_num + 2, 1)

    def choose_quantized_vector(self, z, epoch):  # zはエンコーダの出力
        error = torch.sum((self.quantized_vectors.weight - z) ** 2, dim=1)
        min_index = torch.argmin(error).item()

        return self.quantized_vectors.weight[min_index]

    def quantize_z(self, z_unquantized, epoch):
        z = torch.zeros(z_unquantized.size(), requires_grad=True).to(device)

        for i in range(z_unquantized.size()[0]):
            z[i] = (
                z_unquantized[i]
                + self.choose_quantized_vector(z_unquantized[i].reshape(-1), epoch)
                - z_unquantized[i].detach()
            )

        return z

    def encode(self, linguistic_f, acoustic_f, mora_index, tokyo):
        labels = torch.cat([torch.ones([linguistic_f.size()[0], 1, 1]), torch.zeros([linguistic_f.size()[0], 1, 1])], dim=2) if not tokyo else torch.cat([torch.zeros([linguistic_f.size()[0], 1, 1]), torch.ones([linguistic_f.size()[0], 1, 1])], dim=2)

        x = torch.cat([linguistic_f, acoustic_f], dim=1)
        x = self.fc11(x)
        x = F.relu(x)
        
        out, hc = self.lstm1(torch.cat([x.view(x.size()[0], 1, -1), labels.to(device)], dim=2))
        out_forward = out[:, :, :hidden_num][mora_index]
        mora_index_for_back = np.concatenate([[0], mora_index[:-1] + 1])
        out_back = out[:, :, hidden_num:][mora_index_for_back]
        out = torch.cat([out_forward, out_back], dim=2)

        h1 = F.relu(out)


        return self.fc2(torch.cat([h1, labels.to(device)[:mora_index_for_back.shape[0]]], dim=2)) #ここはモーラ単位しかない

    def init_codebook(self, codebook):
        self.quantized_vectors.weight = codebook

    def decode(self, z, linguistic_features, mora_index, tokyo):
        labels = torch.cat([torch.ones([linguistic_features.size()[0], 1, 1]), torch.zeros([linguistic_features.size()[0], 1, 1])], dim=2) if not tokyo else torch.cat([torch.zeros([linguistic_features.size()[0], 1, 1]), torch.ones([linguistic_features.size()[0], 1, 1])], dim=2).float().to(device)

        z_tmp = torch.tensor(
            [[0] * self.z_dim] * linguistic_features.size()[0],
            dtype=torch.float32,
            requires_grad=True,
        ).to(device)

        for i, mora_i in enumerate(mora_index):
            prev_index = 0 if i == 0 else int(mora_index[i - 1])
            z_tmp[prev_index : int(mora_i)] = z[i]

        x = torch.cat(
            [
                linguistic_features,
                z_tmp.view(-1, self.z_dim),
            ],
            dim=1,
        )

        x = self.fc12(x)
        x = F.relu(x)

        h3, (h, c) = self.lstm2(torch.cat([x.view(x.size()[0], 1, -1), labels.to(device)], dim=2))
        h3 = F.relu(h3)

        return self.fc3(torch.cat([h3, labels.to(device)], dim=2))  # torch.sigmoid(self.fc3(h3))

    def forward(self, linguistic_features, acoustic_features, mora_index, epoch, tokyo=False):
        z_not_quantized = self.encode(
            linguistic_features, acoustic_features, mora_index, tokyo=tokyo
        )
        # print(z_not_quantized)
        z = self.quantize_z(z_not_quantized, epoch)

        return self.decode(z, linguistic_features, mora_index, tokyo=tokyo), z, z_not_quantized
