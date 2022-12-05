
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.PE import PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model.GTEncoder import Encoder


class Embed2GraphByProduct(nn.Module):

    def __init__(self, input_dim, roi_num=264):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        m = torch.einsum('ijk,ipk->ijp', x, x)
        m = self.sigmoid(m)

        return m

class Embed2GraphByAttention(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, embed_dim)
        self.fc_K = torch.nn.Linear(input_dim, embed_dim)


        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, Q, K):


        K = self.fc_K(K)
        Q = self.fc_Q(Q)

        Q = self.norm(Q)
        K = self.norm(K)


        m = torch.matmul(Q, K.permute(0, 2, 1))

        # 除以每个头尾数的平凡根，做数值缩放
        # m /= d ** 0.5

        m = torch.softmax(m, dim=-1)
        return m

class GCNPredictor(nn.Module):

    def __init__(self, node_input_dim, roi_num=360, pool_ratio = 0.7, gcn_layer=3):
        super().__init__()
        self.num_layers_gcn = gcn_layer
        inner_dim = roi_num
        self.roi_num = roi_num
        self.pool_ratio = pool_ratio
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.fc_shot_cut = nn.Sequential(
                    nn.Linear(inner_dim, 16),
                    nn.LeakyReLU(negative_slope=0.2),
                )
        for i in range(self.num_layers_gcn):
            if i == self.num_layers_gcn - 1:
                gcn = nn.Sequential(
                    nn.Linear(inner_dim, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, 16),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                norm = torch.nn.BatchNorm1d(16)
            else:
                gcn = nn.Sequential(
                    nn.Linear(inner_dim, inner_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                norm = torch.nn.BatchNorm1d(inner_dim)


            self.gcns.append(gcn)
            self.norms.append(norm)

        self.cls = torch.nn.Parameter(torch.zeros(1, 16))

        self.bn = torch.nn.BatchNorm1d(16)

        self.pool = Encoder(input_dim=16, num_head=4, embed_dim=8, is_cls=True)

        self.fcn = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
    def forward(self, m, node_feature):
        bz, node_num = m.shape[0], m.shape[1]

        # 图卷积操作
        x_clone = node_feature.clone()
        x_clone[:, :, :] = 0
        x = node_feature
        for layer in range(self.num_layers_gcn):
            x = torch.einsum('ijk,ijp->ijp', m, x)
            if layer == self.num_layers_gcn - 1:
                x = self.gcns[layer](x) + self.fc_shot_cut(x_clone)
            else:
                x = self.gcns[layer](x) + x_clone
            x = x.reshape((bz * self.roi_num, -1))
            x = self.norms[layer](x)
            x = x.reshape((bz, self.roi_num, -1))
            x_clone = x.clone()

        # 加入cls token
        self.cls = self.cls.to(device)
        x_in = torch.empty(bz, node_num + 1, 16).to(device)
        for i in range(bz):
            x_in[i] = torch.cat((self.cls, x[i]), 0)  #按维数0拼接（竖着拼）
        # 池化
        out, cor_matrix = self.pool(x_in)

        cor = cor_matrix.clone()
        cor = cor[:, :, 0, :].to(device)
        # 计算score
        score = cor[:, 0, :]
        for i in range(3):
            score += cor[:, i + 1, :]

        score = score[:, 1:]
        sc = score.clone()
        score, rank = score.sort(dim=-1, descending=True)

        l = int(node_num * self.pool_ratio)
        x_p = torch.empty(bz, l, 16).to(device)
        # 保留重要结点
        x = out[:, 1:, :]
        score = score[:, :l]
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        for i in range(x.shape[0]):
            x_p[i] = x[i, rank[i, :l], :]

        x_p = torch.matmul(score, x_p).squeeze(1)


        # 拉平
        x = x_p.view(bz, -1)
        x = self.fcn(x)
        return x, torch.sigmoid(sc), cor_matrix


class GraphTransformer(nn.Module):

    def __init__(self, model_config, roi_num=200, node_feature_dim=360, time_series=512, pool_ratio = 0.7):
        super().__init__()

        self.extract = Encoder(input_dim=time_series, num_head=4, embed_dim=model_config['embedding_size'])



        self.emb2graph = Embed2GraphByProduct(model_config['embedding_size'], roi_num=roi_num)
        # self.emb2graph = Embed2GraphByAttention(input_dim=time_series, embed_dim=model_config['embedding_size'])

        self.predictor = GCNPredictor(node_feature_dim, roi_num=roi_num, pool_ratio = pool_ratio, gcn_layer=model_config['gcn_layer'])

        self.bn = torch.nn.BatchNorm1d(roi_num)
        self.pe = PositionalEncoding(d_model=roi_num, roi_num=roi_num)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(time_series, 8),
            torch.nn.Sigmoid(),
                                          )
        self.bn1 = torch.nn.BatchNorm1d(16)

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, t, nodes, pseudo):
        m, _ = self.extract(t)
        m = self.fc(m)
        m = self.emb2graph(m)


        bz, _, _ = m.shape

        edge_variance = torch.mean(torch.var(m.reshape((bz, -1)), dim=1))

        return self.predictor(m, nodes), m, edge_variance























        # self.tfs = nn.ModuleList()
        # self.tf_norm = nn.ModuleList()
        # for i in range(self.num_layers_tf):
        #     tf = Encoder(input_dim=32, num_head=4, embed_dim=8, is_cls=True)
        #     norm = torch.nn.BatchNorm1d(8)
        #     self.tfs.append(tf)
        #     self.tf_norm.append(norm)

        # for layer in range(self.num_layers_tf - 1):
        #     x = x.reshape((bz * (self.roi_num + 1), -1))
        #     x = self.tf_norm[layer](x)
        #     x = x.reshape((bz, self.roi_num + 1, -1))
        #     x, cor_matrix = self.tfs[layer + 1](x)

        # x = x[:, 0, :]
        # x = self.bn(x)