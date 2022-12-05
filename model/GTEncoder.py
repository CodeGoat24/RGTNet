import torch.nn
import torch

class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(embed_dim, input_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1)
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)

    def forward(self, x):
        x_clone = x.clone()

        x = self.norm(x)


        out = self.fc(x)
        out = out + x_clone
        return out

# 注意力计算函数
def attention(Q, K, V, is_cls):

    l = Q.shape[2]
    d = Q.shape[3]
    num_head = Q.shape[1]


    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头尾数的平凡根，做数值缩放
    score /= d ** 0.5


    if is_cls:
        score[:, :, :, 0] = -float('inf')
    cor_matrix = score.clone()

    score = torch.softmax(score, dim=-1)

    score = torch.matmul(score, V)

    score = score.permute(0, 2, 1, 3).reshape(-1, l, num_head*Q.shape[3])

    return score, cor_matrix


class MultiHead(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim, is_cls):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, embed_dim)
        self.fc_K = torch.nn.Linear(input_dim, embed_dim)
        self.fc_V = torch.nn.Linear(input_dim, embed_dim)
        self.is_cls = is_cls
        self.num_head = num_head

        self.out_fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 2*embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2*embed_dim, input_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
                                          )

        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)
        self.norm1 = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, Q, K, V):

        b = Q.shape[0]
        len = Q.shape[1]

        clone_Q = Q.clone()


        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)


        Q = Q.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)


        score, cor_matrix = attention(Q, K, V, self.is_cls)
        score = self.norm1(score)

        score = self.out_fc(score)
        score = clone_Q + score
        return score, cor_matrix

class EncoderLayer(torch.nn.Module):
    def __init__(self,input_dim, num_head, embed_dim, is_cls):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead(input_dim, num_head, embed_dim, is_cls)


    def forward(self, x):
        score, cor_matrix = self.mh(x, x, x)


        return score, cor_matrix

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim, is_cls = False):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(input_dim, num_head, embed_dim, is_cls)

    def forward(self, x):
        x, cor_matrix = self.layer(x)

        return x, cor_matrix