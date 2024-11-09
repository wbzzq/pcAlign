
from scipy.spatial.transform import Rotation as R
import os
import copy
import math
import json
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from util import transform_point_cloud, npmat2euler
from PSE import PSE_module
import open3d as o3d
import itertools
import dcputil
import csv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#获取点云中每对点构成的三角形的三条边的长度，并对每个三角形的边长进行排序（从小到大）。
#pcd：点云的坐标，形状为 [B, Number_points, 3]，其中 B 是批次大小，Number_points 是每个点云中的点数，每个点有三个坐标值（x, y, z）。
#pairs：形成三角形的点对，形状为 [B, Number_points, Number_pairs, 3]，其中 Number_pairs 是每个点的配对数。
#result：形状为 [B, Number_points, Number_pairs, 3]
def getTri(pcd,pairs):
    B,N,N_p,_ = pairs.shape
    result = torch.zeros((B*N, N_p, 3), dtype=torch.float32)
    temp = (torch.arange(B) * N).reshape(B,1,1,1).repeat(1,N,N_p,3).cuda()
    pairs = pairs + temp
    pcd = pcd.reshape(-1,3)
    pairs = pairs.reshape(-1,N_p,3)
    result[:,:,0] = (torch.sum(((pcd[pairs[:,:,0],:]-pcd[pairs[:,:,1],:])**2),dim=-1))
    result[:,:,1] = (torch.sum(((pcd[pairs[:,:,1],:]-pcd[pairs[:,:,2],:])**2),dim=-1))
    result[:,:,2] = (torch.sum(((pcd[pairs[:,:,0],:]-pcd[pairs[:,:,2],:])**2),dim=-1))
    result = result.reshape(B,N,N_p,3)
    result, _ = torch.sort(result,dim=-1,descending=False)
    return result

#找出 DISTANCE_THRESHOLD 以外的 k 个最近点
#x:  点云[B, 3, Number_points]
#k:  最近点的个数
#idx：最近点的索引
def knn_tri(x,k):
    """
    Function: find the k nearest points outside DISTANCE_THRESHOLD
    Param:
        x:  point clouds [B, 3, Number_points]
        k:  The number of points
    Return:
        idx: the index of k nearest points
    """
    DISTANCE_THRESHOLD = -0.1
    x = x.transpose(1,2)
    distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
    mask = distance > DISTANCE_THRESHOLD
    distance[mask] = float('-inf')
    idx = distance.topk(k=k,dim=-1)[1]
    return idx

#判断源点云和目标点云对应关系的有效性，weight矩阵中的值越接近1，对应关系越有效
def GMCCE(src,tgt,Sharp=30,Threshold=0.6,k=10,eps=1e-6):
    """
    Function: judge whether the correspondence is inlier or not
    Param:
        src: source point clouds [B, 3, Number_points]
        tgt: target point clouds [B, 3, Number_points]
        corres: predicted correspondences [B, Number_points, 1]
        k:  The number of pairs is k x (k - 1) // 2
    """
    index = torch.arange(src.shape[2]).reshape(1,-1,1)
    index = index.repeat(src.shape[0],1,k*(k-1)//2).unsqueeze(-1).cuda()
    idx = knn_tri(src,k)
    pairs = list(itertools.combinations(torch.arange(0,k),2))
    idx_pairs = idx[:,:,pairs]
    src_T_pairs = torch.cat((index,idx_pairs),dim=-1)
    src = src.transpose(-1,-2)
    length_src = getTri(src,src_T_pairs)
    length_tgt = getTri(tgt,src_T_pairs) + eps
    loss = torch.sum((length_src - length_tgt)**2,dim=-1)
    loss = loss/torch.sum((length_src + length_tgt)**2,dim=-1)
    loss,_ = torch.sort(loss,dim=-1,descending=False)
    loss = loss[:,:,:k]
    loss = torch.sqrt(loss + eps)
    loss = torch.mean(loss,dim=-1)
    Min_loss,_ = torch.min(loss,dim=-1)
    Min_loss = Min_loss.reshape(-1,1)
    loss = loss - Min_loss

    weight = 2 * torch.sigmoid(-Sharp * (loss.cuda()))
    weight[weight <= Threshold] = 0
    weight[weight > Threshold] = 1

    return weight

#在目标点云中搜索与源点云中的每个点距离在 search_voxel_size 内的点，来获取两个点云之间的匹配索引。
#通过 KD 树搜索在目标点云中找到与源点云中的每个点匹配的点对。通过限制搜索半径和匹配点数目，可以有效地找到两个点云之间的对应关系
def get_matching_indices(source, target, search_voxel_size, K=None):
    """
    Function: get matching indices
    Param:
        source, target：point clouds [B, Number_points, 3]
        trans：transformation
        search_voxel_size：the points within search_voxel_size are considered as inliers
    Return
        match_inds: {i,j}
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def clones(module, N):
    """
    Function: clone the module N times
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#注意力机制：Q、K、V那个计算公式
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


#x:  点云[B, 3, Number_points]
#k:  最近点的个数
#idx：最近点的索引
def knn(x, k):
    x = x.transpose(1,2)
    distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
    idx = distance.topk(k=k, dim=-1)[1]
    return idx

#计算循环一致性，即源点云到目标点云的trans和目标点云到源点云的trans是否一致
def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):

    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, n_emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(n_emb_dims, n_emb_dims//2),
                                nn.LayerNorm(n_emb_dims//2),
                                nn.LeakyReLU(),
                                nn.Linear(n_emb_dims//2, n_emb_dims//4),
                                nn.LayerNorm(n_emb_dims//4),
                                nn.LeakyReLU(),
                                nn.Linear(n_emb_dims//4, n_emb_dims//8),
                                nn.LayerNorm(n_emb_dims//8),
                                nn.LeakyReLU())
        self.proj_rot = nn.Linear(n_emb_dims//8, 4)
        self.proj_trans = nn.Linear(n_emb_dims//8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.n_ff_dims = args.n_ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

#计算源特征嵌入和目标特征嵌入之间的差异，并基于这种特征差异residual生成一个Temperature
class TemperatureNet(nn.Module):
    def __init__(self, args):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.temp_factor = args.temp_factor
        self.nn = nn.Sequential(nn.Linear(self.n_emb_dims, 128),
                                nn.LayerNorm(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 128),
                                nn.LayerNorm(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 128),
                                nn.LayerNorm(128),
                                nn.LeakyReLU(),
                                nn.Linear(128, 1),
                                nn.LeakyReLU())
        self.feature_disparity = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src_embedding = src_embedding.mean(dim=2)
        tgt_embedding = tgt_embedding.mean(dim=2)
        residual = torch.abs(src_embedding-tgt_embedding)

        self.feature_disparity = residual

        Temperature = torch.clamp(self.nn(residual), 1.0/self.temp_factor, 1.0*self.temp_factor)
        return Temperature, residual

#计算两个点集之间的旋转矩阵和位移向量，使得一个点集通过该变换后尽可能地与另一个点集对齐。该函数采用加权的方式，使某些点在计算中占有更大的权重。
#w: 形状为 (N) 的张量，表示每个点的权重。
#eps: 一个小的浮点数，用于避免除零错误。
def weighted_procrustes(X, Y, w, eps):
    """
    X: torch tensor N x 3
    Y: torch tensor N x 3
    w: torch tensor N
    """
    assert len(X) == len(Y)
    W1 = torch.abs(w).sum()
    w = w.reshape(-1,1).repeat(1,3)
    w_norm = w / (W1 + eps)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)

    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
    U, D, V = Sxy.svd()
    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t())).float()
    t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
    return R, t


def visualize_correspondence(src, src_corr):
    """
    src: torch.Tensor of shape [N, 3] (source point cloud)
    src_corr: torch.Tensor of shape [N, 3] (corresponding points in tgt point cloud)
    """
    src_np = src.cpu().numpy()  # Convert to numpy array for visualization
    src_corr_np = src_corr.cpu().numpy()  # Convert to numpy array for visualization

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot source points in red
    ax.scatter(src_np[:, 0], src_np[:, 1], src_np[:, 2], c='r', label='Source points')

    # Plot corresponding target points in blue
    ax.scatter(src_corr_np[:, 0], src_corr_np[:, 1], src_corr_np[:, 2], c='b', label='Corresponding points')

    # Draw lines between corresponding points
    for i in range(src_np.shape[0]):
        ax.plot([src_np[i, 0], src_corr_np[i, 0]],
                [src_np[i, 1], src_corr_np[i, 1]],
                [src_np[i, 2], src_corr_np[i, 2]], 'g-', linewidth=0.5)

    # Set labels and legend
    plt.legend()
    plt.title('Source and Corresponding Points')
    plt.show()


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.corres_mode = args.corres_mode
        self.alpha = args.alpha
        self.Sharp = args.GMCCE_Sharp
        self.Threshold = args.GMCCE_Thres
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.temperature = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        self.my_iter = torch.ones(1)
        self.last_tem = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, num_dims, num_points = src.size()
        temperature = input[4].view(batch_size, 1, 1)
        is_corr = input[5]

        if self.corres_mode == True and is_corr == True:
            R = []
            T = []
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(10000 * scores, dim=2)
            _, corres = torch.max(scores,dim=-1)
            corr_tgt = torch.matmul(scores,tgt.transpose(1,2))
            corres = corres.reshape(corres.shape[0],-1,1)
            weight = GMCCE(src,corr_tgt,Sharp=self.Sharp,Threshold=self.Threshold)
            weight = weight.unsqueeze(-1)
            src = src.transpose(1,2).contiguous()
            tgt = tgt.transpose(1,2).contiguous()
            for i in range(src.shape[0]):
                src_corr = tgt[i][corres[i]].squeeze()
                r,t = weighted_procrustes(src[i].cpu(),src_corr.cpu(),weight[i].detach().cpu(),1e-7)
                R.append(r)
                T.append(t)
            R = torch.stack(R, dim=0).cuda()
            T = torch.stack(T, dim=0).cuda()
            return R, T, corres, weight
        else:
            R = []
            T = []
            d_k = src_embedding.size(1)
            feature_scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k) #计算src_embedding和tgt_embedding的相似度 torch.Size([1, N, N])
            feature_scores = torch.softmax(temperature * feature_scores, dim=2) #torch.Size([1, N, N])
            # distances = torch.cdist(src.transpose(1,2).contiguous(), tgt.transpose(1,2).contiguous())  # [B, N_src, N_tgt]
            # #将距离转换为得分，例如使用高斯函数
            # geo_scores = torch.exp(-temperature * distances)  # temperature 调节尺度，影响得分的分散性
            # idx = torch.arange(src.shape[2]).reshape(1,-1).repeat(src.shape[0],1)
            # feature_scores = feature_scores / feature_scores.sum(dim=2, keepdim=True)
            # geo_scores = geo_scores / geo_scores.sum(dim=2, keepdim=True)
            # scores = self.alpha * feature_scores + (1 - self.alpha) * geo_scores  # alpha 是一个可调的混合系数
            #weight, corres = torch.max(scores,dim=-1)
            weight, corres = torch.max(feature_scores,dim=-1)
            corres = corres.reshape(corres.shape[0],-1,1)  #torch.Size([2, 500, 1])
            weight = weight.unsqueeze(-1)
            src = src.transpose(1,2).contiguous()
            tgt = tgt.transpose(1,2).contiguous()

            #identity_matrix = torch.eye(3, device='cuda:0')

            for i in range(src.shape[0]):
                src_corr = tgt[i][corres[i]].squeeze() #torch.Size([500, 3])
                #visualize_correspondence(src[i], src_corr)
                #print(type(src_corr), src_corr.shape)
                r,t = weighted_procrustes(src[i].cpu(),src_corr.cpu(),weight[i].detach().cpu(),1e-7)
                #R.append(identity_matrix)
                R.append(r)
                T.append(t)
            R = torch.stack(R, dim=0).cuda()
            T = torch.stack(T, dim=0).cuda()
            return R, T, corres, weight


class Position_encoding(nn.Module):
    def __init__(self,len,ratio):
        super(Position_encoding,self).__init__()
        self.PE = nn.Sequential(
            nn.Linear(3,len * ratio),
            nn.Sigmoid(),
            nn.Linear(len * ratio,len),
            nn.ReLU()
        )
    def forward(self,x):
        return self.PE(x)

#将原始输入特征与注意力权重逐通道相乘，增强重要特征的表示。
class SE_Block(nn.Module):
    def __init__(self,ch_in,reduction=16):
        super(SE_Block,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in,ch_in//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in//reduction,ch_in,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x*y.expand_as(x)

class T_prediction(nn.Module):
    def __init__(self, args):
        super(T_prediction, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.Position_encoding = Position_encoding(args.n_emb_dims,8)
        self.SE_Block = SE_Block(ch_in=args.n_emb_dims)
        self.emb_nn = PSE_module(embed_dim=args.n_emb_dims,token_dim=args.token_dim)
        self.attention = Transformer(args=args)
        self.temp_net = TemperatureNet(args)
        self.head = SVDHead(args=args)

 
    def forward(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr = self.predict_embedding(*input)
        rotation_ab, translation_ab, corres_ab, weight_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature,is_corr)
        rotation_ba, translation_ba, corres_ba, weight_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature,is_corr)
        return rotation_ab.cuda(), translation_ab.cuda(), rotation_ba.cuda(), translation_ba.cuda(), feature_disparity, corres_ab, weight_ab

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        is_corr = input[2]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)
        src_encoding = self.Position_encoding(src.transpose(1,2)).transpose(1,2).contiguous()
        tgt_encoding = self.Position_encoding(tgt.transpose(1,2)).transpose(1,2).contiguous()

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding+src_encoding, tgt_embedding+tgt_encoding)
        src_embedding = self.SE_Block(src_embedding+src_embedding_p)  # torch.Size([1, 512, 500])
        tgt_embedding = self.SE_Block(tgt_embedding+tgt_embedding_p)  # torch.Size([1, 512, 500])
        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
            pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def corres_correct(matches,corres): 
    B,N,_ = corres.shape
    idx_src = torch.arange(N).reshape(1,-1,1).repeat(B,1,1)
    corres = torch.cat([idx_src,corres],dim=-1).int()
    is_correct = []
    for match,corres_iter in zip(matches,corres):
        is_correct_iter = find_correct_correspondence(torch.tensor(match).int().unsqueeze(0),corres_iter.unsqueeze(0),N)
        is_correct.extend(is_correct_iter)
    return np.array(is_correct)

#可视化正确的对应关系
def visualize_matches(pcd0, pcd1, matches):
    """
    Visualize two point clouds (pcd0, pcd1) and their corresponding matches.
    """
    # 将点云分配颜色
    pcd0.paint_uniform_color([1, 0, 0])  # 红色
    pcd1.paint_uniform_color([0, 1, 0])  # 绿色

    # 创建线条集合，用于展示匹配关系
    lines = []
    for match in matches:
        for idx0, idx1 in match:
            lines.append([idx0, len(pcd0.points) + idx1])

    # 合并两个点云
    merged_pcd = pcd0 + pcd1

    # 创建线条集合
    line_set = o3d.geometry.LineSet()
    line_set.points = merged_pcd.points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 0, 1])  # 线条颜色为蓝色

    # 可视化点云及其匹配
    o3d.visualization.draw_geometries([pcd0, pcd1, line_set])

def find_matches(src,tgt,rotation,translation):
    src = transform_point_cloud(src, rotation, translation)
    matches = []
    for pointcloud1,pointcloud2 in zip(src,tgt):
        pcd0 = make_open3d_point_cloud(pointcloud1.cpu().numpy().T)
        pcd1 = make_open3d_point_cloud(pointcloud2.cpu().numpy().T)
        match = get_matching_indices(pcd0, pcd1, 3)
        #visualize_matches(pcd0,pcd1,matches)
        matches.append(match)
    return matches

def _hash(arr, M=None):
    if len(arr.shape) == 2:
        N, D = arr.shape
    elif len(arr.shape) == 1:
        N = arr.shape[0]
        D = 1
    else:
        raise ValueError("Unsupported array shape")
    # if isinstance(arr, np.ndarray):
    #     N, D = arr.shape
    # else:
    #     N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if D == 1:
            # 处理一维数组
            hash_vec += arr * (M**d)
        else:
            # 处理二维数组
            hash_vec += arr[:, d] * (M**d)
    return hash_vec

def find_correct_correspondence(pos_pairs, pred_pairs, hash_seed=None):
    assert len(pos_pairs) == len(pred_pairs)
    if hash_seed is None:
        #assert len(len_batch) == len(pos_pairs)
        return
    corrects = []
    for i, pos_pred in enumerate(zip(pos_pairs, pred_pairs)):
        pos_pair, pred_pair = pos_pred
        if isinstance(pos_pair, torch.Tensor):
            pos_pair = pos_pair.numpy()
        if isinstance(pred_pair, torch.Tensor):
            pred_pair = pred_pair.numpy()

        _hash_seed = hash_seed

        pos_keys = _hash(pos_pair, _hash_seed)
        pred_keys = _hash(pred_pair, _hash_seed)

        corrects.append(np.isin(pred_keys, pos_keys, assume_unique=False))
    return np.hstack(corrects)

def square_distance(src, dst):
     # Expand dimensions for broadcasting
    src_expand = src.unsqueeze(3)  # (B, 3, N, 1)
    dst_expand = dst.unsqueeze(2)  # (B, 3, 1, M)
    
    # Compute squared Euclidean distance
    dist_sq = torch.sum((src_expand - dst_expand) ** 2, dim=1)  # (B, N, M)
    return dist_sq

def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu()
    return np.array(arr)


def compute_discrimination_loss(src_points, tgt_points, corres, is_correct, weight_ab, margin=0.05, alpha=0.5):
    """
    Args:
        src_points (torch.Tensor): 源点云 [B, 3, N]，B为批次大小，N为点的数量
        tgt_points (torch.Tensor): 目标点云 [B, 3, N]
        corres (torch.Tensor): 对应点的索引 [B, N, 1]，模型预测的点对索引
        is_correct (torch.Tensor): 匹配是否正确的标签 [B*N]，展平后的张量
        weight_ab (torch.Tensor): 模型预测的置信度 [B, N, 1]
        margin (float): 距离阈值，如果点对的物理距离大于这个值则会受到惩罚
        alpha (float): 惩罚因子，控制距离误差对总损失的影响
    Returns:
        torch.Tensor: discrimination loss
    """
    
    # 初始化损失
    B, _, N = src_points.shape  # 获取批次大小和点的数量
    loss = 0.0
    
    # 将is_correct展平到形状 [B, N]
    is_correct = is_correct.view(B, N)
    
    # 遍历每个批次中的点云对
    for i in range(B):
        # 获取当前批次的源点和目标点
        src = src_points[i]  # [3, N]
        tgt = tgt_points[i]  # [3, N]
        
        # 获取当前批次的正确性标签和对应关系
        correct = is_correct[i]  # [N]
        corres_idx = corres[i].squeeze(-1)  # [N]，去掉最后一维，将索引展平成 [N]
        
        # 获取对应点的坐标
        matched_tgt_points = tgt[:, corres_idx]  # [3, N]，通过索引找到源点对应的目标点
        
        # 计算源点和对应目标点之间的欧氏距离
        distances = torch.norm(src - matched_tgt_points, dim=0)  # [N]，每个点的距离
        
        # 对正确匹配的点，计算距离损失，如果距离大于 margin，则受到惩罚
        distance_loss = torch.clamp(distances - margin, min=0.0)
        
        # 获取当前批次的权重，去掉最后一维
        weight = weight_ab[i].squeeze(-1)  # [N]
        
        # 计算加权的总损失，基于匹配的置信度和距离误差
        # 对于匹配错误的点也要进行惩罚
        loss_per_point = correct * (distance_loss + alpha * (1 - weight)) + torch.logical_not(correct) * weight

        
        # 累积当前批次的损失
        loss += torch.mean(loss_per_point)
    
    return loss / B  # 平均化损失



class DIT(nn.Module):
    def __init__(self, args):
        super(DIT, self).__init__()
        # self.writer = SummaryWriter(log_dir=r'./checkpoints/{%s}/tensorboards/', args.exp_name)
        self.writer = SummaryWriter(log_dir=r'./checkpoints/{}/tensorboards/'.format(args.exp_name))
        self.num_iters = args.n_iters
        self.logger = Logger(args)
        self.discount_factor = args.discount_factor
        self.discrimination_loss = args.discrimination_loss
        self.T_prediction = T_prediction(args)
        self.model_path = args.model_path
        self.cycle_consistency_loss = args.cycle_consistency_loss
        self.crit = nn.BCELoss()

        if torch.cuda.device_count() > 1:
            self.T_prediction = nn.DataParallel(self.T_prediction)

    def forward(self, *input):
        rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab,weight_ab = self.T_prediction(*input)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab,weight_ab

    def _train_one_batch(self, pcd, Transform, opt):
        opt.zero_grad()
        src, tgt, rotation_ab, translation_ab = pcd['src'], pcd['tgt'], Transform['R_ab'], Transform['T_ab']
        batch_size = src.size(0)
        device = src.device
        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotation_ab_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_cycle_consistency_loss = 0
        total_discrimination_loss = 0
        total_mse_loss = 0
        total_rotation_loss = 0
        total_translation_loss = 0
        accuracy = 0
        losses = {}
        Transforms_Pred = {}

        corres_gt = find_matches(src,tgt,rotation_ab,translation_ab)

        for i in range(self.num_iters):
            is_corr = 1
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
            feature_disparity, corres_ab ,weight_ab = self.forward(src, tgt, is_corr)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ba_pred_i
            
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            rotation_loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity)
            translation_loss = F.mse_loss(translation_ab_pred, translation_ab)
            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            total_mse_loss += loss
            is_correct = torch.tensor(corres_correct(corres_gt,corres_ab.cpu())).squeeze() #torch.Size([1000])
            accuracy = is_correct.sum()/is_correct.shape[0]
            if self.discrimination_loss != 0:  
               discrimination_loss = self.discrimination_loss * self.crit((weight_ab).reshape(-1,1).squeeze().cpu(), is_correct.to(torch.float)) * self.discount_factor ** i
            #    discrimination_loss = compute_discrimination_loss(
            #     src_points=src,
            #     tgt_points=tgt,
            #     corres=corres_ab,
            #     is_correct=is_correct.cuda(),
            #     weight_ab=weight_ab,
            #     margin=3,
            #     alpha=0.5
            # ) * self.discount_factor ** i
               total_discrimination_loss += discrimination_loss
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i, translation_ba_pred_i) \
                                     * self.cycle_consistency_loss * self.discount_factor**i  
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss = total_loss + loss + cycle_consistency_loss + discrimination_loss 
            #total_loss = total_loss + loss + cycle_consistency_loss
            total_rotation_loss += rotation_loss
            total_translation_loss += translation_loss
        clip_val = torch.tensor(0.1, device=src.device, dtype=torch.float32)
        dist_src = torch.minimum(torch.min(torch.sqrt(square_distance(src, tgt)), dim=-1)[0], clip_val)
        dist_ref = torch.minimum(torch.min(torch.sqrt(square_distance(tgt, src)), dim=-1)[0], clip_val)
        chamfer_dist = torch.mean(dist_src, dim=-1) + torch.mean(dist_ref, dim=-1)
        chamfer_dist = chamfer_dist.mean().item()
        total_loss.backward()
        opt.step()

        losses['total_loss'] = total_loss.item()
        losses['cycle'] = total_cycle_consistency_loss
        losses['acc'] = accuracy
        losses['discrimination'] = total_discrimination_loss.item()
        losses['mse'] = total_mse_loss.item()
        losses['trans'] = total_translation_loss.item()
        losses['rot'] = total_rotation_loss.item()
        losses['chamfer_distance'] = chamfer_dist
        Transforms_Pred['R_ab_pred'] = rotation_ab_pred
        Transforms_Pred['T_ab_pred'] = translation_ab_pred

        return losses, Transforms_Pred

    def _test_one_batch(self, pcd, Transform):
        src, tgt, rotation_ab, translation_ab = pcd['src'], pcd['tgt'], Transform['R_ab'], Transform['T_ab']
        batch_size = src.size(0)
        device = src.device
        identity = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_cycle_consistency_loss = 0
        total_discrimination_loss = 0
        total_mse_loss = 0
        accuracy = 0
        losses = {}
        Transforms_Pred = {}

        corres_gt = find_matches(src,tgt,rotation_ab,translation_ab)
        for i in range(self.num_iters):
            is_corr = (i != 0)
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
            feature_disparity, corres_ab,weight_ab = self.forward(src, tgt, is_corr)
        
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ba_pred_i

            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            total_mse_loss += loss
            discrimination_loss = 0
            is_correct = torch.tensor(corres_correct(corres_gt,corres_ab.cpu())).squeeze()
            accuracy = is_correct.sum()/is_correct.shape[0]
            if self.discrimination_loss != 0:
                discrimination_loss = self.discrimination_loss * self.crit((weight_ab).reshape(-1,1).squeeze().cpu(), is_correct.to(torch.float)) * self.discount_factor ** i
            #     discrimination_loss = compute_discrimination_loss(
            #     src_points=src,
            #     tgt_points=tgt,
            #     corres=corres_ab,
            #     is_correct=is_correct.cuda(),
            #     weight_ab=weight_ab,
            #     margin=3,
            #     alpha=0.5
            # ) * self.discount_factor ** i
                total_discrimination_loss += discrimination_loss
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i, translation_ba_pred_i) \
                                     * self.cycle_consistency_loss * self.discount_factor ** i
            #total_cycle_consistency_loss += cycle_consistency_loss
            total_loss = total_loss + loss + cycle_consistency_loss + discrimination_loss 
        clip_val = torch.tensor(0.1, device=src.device, dtype=torch.float32)
        dist_src = torch.minimum(torch.min(torch.sqrt(square_distance(src, tgt)), dim=-1)[0], clip_val)
        dist_ref = torch.minimum(torch.min(torch.sqrt(square_distance(tgt, src)), dim=-1)[0], clip_val)
        chamfer_dist = torch.mean(dist_src, dim=-1) + torch.mean(dist_ref, dim=-1)
        chamfer_dist = chamfer_dist.mean().item()
        losses['total_loss'] = total_loss.item()
        losses['cycle'] = total_cycle_consistency_loss
        losses['acc'] = accuracy
        losses['discrimination'] = total_discrimination_loss.item()
        losses['mse'] = total_mse_loss.item()
        losses['chamfer_distance'] = chamfer_dist
        Transforms_Pred['R_ab_pred'] = rotation_ab_pred
        Transforms_Pred['T_ab_pred'] = translation_ab_pred
        return losses, Transforms_Pred


    def Compute_metrics(self, avg_losses, Transforms):
        # print(Transforms['R_ab'].shape)
        # print(Transforms['T_ab'].shape)
        # print(Transforms['R_ab_pred'].shape)
        # print(Transforms['T_ab_pred'].shape)
        concatenated = dcputil.concatenate(dcputil.inverse(Transforms['R_ab'], Transforms['T_ab']),
                                        np.concatenate([Transforms['R_ab_pred'], Transforms['T_ab_pred'].unsqueeze(-1)], axis=-1))
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = np.mean((torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).numpy())
        residual_transmag = np.mean((concatenated[:, :, 3].norm(dim=-1)).numpy())

        rotations_ab = Transforms['R_ab'].numpy()
        translations_ab = Transforms['T_ab'].numpy()
        rotations_ab_pred = Transforms['R_ab_pred'].numpy()
        translations_ab_pred = Transforms['T_ab_pred'].numpy()

        r_ab_mae = np.abs(Transforms['euler_ab'] - Transforms['euler_ab_pred'])
        t_ab_mae = np.abs(translations_ab - translations_ab_pred)
        cur_acc = np.mean((r_ab_mae <= 1)*(t_ab_mae <= 0.0002))
        r_ab_mse = np.mean((Transforms['euler_ab'] - Transforms['euler_ab_pred']) ** 2,axis=1)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(r_ab_mae, axis=1)
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2, axis=1)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(t_ab_mae,axis=1)
        r_ab_r2_score = r2_score(Transforms['euler_ab'], Transforms['euler_ab_pred'])
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)
        
        info = {#'arrow': 'A->B',
                'epoch': avg_losses['epoch'],
                'elapsed_time': avg_losses['elapsed_time'],
                'stage': avg_losses['stage'],
                'loss': avg_losses['avg_loss'],
                #'cycle_consistency_loss': avg_losses['avg_cycle'],
                #'dis_loss': avg_losses['avg_discrimination'],
                #'mse_loss': avg_losses['avg_mse'],
                'r_ab_mse': r_ab_mse.mean(),
                'r_ab_rmse': r_ab_rmse.mean(),
                'r_ab_mae': r_ab_mae.mean(),
                't_ab_mse': t_ab_mse.mean(),
                't_ab_rmse': t_ab_rmse.mean(),
                't_ab_mae': t_ab_mae.mean(),
                'chamfer_distance': avg_losses['chamfer_distance'],
                #'r_ab_r2_score': r_ab_r2_score,
                #'t_ab_r2_score': t_ab_r2_score,
                'corres_accuracy':avg_losses['avg_acc'].item(),
                'r_ab_mie': residual_rotdeg,
                't_ab_mie': residual_transmag,
                'cur_acc':cur_acc}
        self.logger.write(info)
        return info

    def _train_one_epoch(self, epoch, train_loader, opt, args):
        self.train()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        total_chamfer_distance = 0.0  # 初始化 Chamfer 距离
        total_corres_accuracy = 0.0
        total_cycle_consistency_loss = 0.0
        #total_scale_consensus_loss = 0.0
        total_mse_loss = 0.0
        total_rotation_loss = 0.0
        total_translation_loss = 0.0
        total_discrimination_loss = 0.0
        avg_losses = {}
        Transforms = {}
        vis = False
        start = time.time()
        batch_idx = 0
        for pcd, Transform in tqdm(train_loader):
            for key in pcd.keys():
                pcd[key] = pcd[key].cuda() 
            for key in Transform.keys():
                Transform[key] = Transform[key].cuda() 
            losses, Transform_pred = self._train_one_batch(pcd, Transform, opt)
            batch_size = pcd['src'].size(0)
            num_examples += batch_size

            total_mse_loss += losses['mse'] * batch_size
            total_rotation_loss += losses['rot']*batch_size
            total_translation_loss += losses['trans']*batch_size
            total_loss = total_loss + losses['total_loss'] * batch_size
            total_cycle_consistency_loss = total_cycle_consistency_loss + losses['cycle'] * batch_size
            #total_scale_consensus_loss = total_scale_consensus_loss + losses['scale'] * batch_size
            total_corres_accuracy += losses['acc'] * batch_size
            total_discrimination_loss += losses['discrimination'] * batch_size
            total_chamfer_distance += losses['chamfer_distance'] * batch_size  # 累加 Chamfer 距离

            #使用tensorboard记录损失
            self.writer.add_scalar('Loss/total_loss', losses['total_loss'], epoch * len(train_loader) + batch_idx)
            self.writer.add_scalar('Loss/rot_loss', losses['rot'], epoch * len(train_loader) + batch_idx)
            self.writer.add_scalar('Loss/trans_loss', losses['trans'], epoch * len(train_loader) + batch_idx)

            rotations_ab.append(Transform['R_ab'].detach().cpu())
            translations_ab.append(Transform['T_ab'].detach().cpu())
            rotations_ab_pred.append(Transform_pred['R_ab_pred'].detach().cpu())
            translations_ab_pred.append(Transform_pred['T_ab_pred'].detach().cpu())
            eulers_ab.append(Transform['euler_ab'].cpu().numpy())

            # if (batch_idx + 1) % 100 == 0:
            #     save_path = f'./checkpoints/{args.exp_name}/models/train_{epoch}_batch_{batch_idx + 1}.pth'
            #     torch.save(self.state_dict(), save_path)

            batch_idx += 1

        end = time.time()
        avg_losses['avg_loss'] = total_loss / num_examples
        avg_losses['avg_cycle'] = total_cycle_consistency_loss / num_examples
        avg_losses['avg_acc'] = total_corres_accuracy / num_examples
        avg_losses['avg_discrimination'] = total_discrimination_loss / num_examples
        avg_losses['avg_mse'] = total_mse_loss / num_examples
        avg_losses['chamfer_distance'] = total_chamfer_distance/ num_examples
        avg_losses['epoch'] = epoch
        avg_losses['stage'] = 'train'
        avg_losses['elapsed_time'] = (end - start)/num_examples

        Transforms['R_ab'] = torch.cat(rotations_ab, axis=0)
        Transforms['T_ab'] = torch.cat(translations_ab, axis=0)
        Transforms['R_ab_pred'] = torch.cat(rotations_ab_pred, axis=0)
        Transforms['T_ab_pred'] = torch.cat(translations_ab_pred, axis=0)
        Transforms['euler_ab'] = np.degrees(np.concatenate(eulers_ab, axis=0))
        Transforms['euler_ab_pred'] = npmat2euler(Transforms['R_ab_pred'])
        return self.Compute_metrics(avg_losses,Transforms)

    def _test_one_epoch(self, epoch, test_loader):
        start = time.time()
        with torch.no_grad():
            self.eval()
            total_loss = 0
            total_chamfer_distance = 0.0  # 初始化 Chamfer 距离
            rotations_ab = []
            translations_ab = []
            rotations_ab_pred = []
            translations_ab_pred = []
            eulers_ab = []
            num_examples = 0
            total_corres_accuracy = 0.0
            total_cycle_consistency_loss = 0.0
            #total_scale_consensus_loss = 0.0
            total_mse_loss = 0.0
            total_discrimination_loss = 0.0
            avg_losses = {}
            Transforms = {}
            Metrics_mode = {}
            for pcd, Transform in tqdm(test_loader):
                for key in pcd.keys():
                    pcd[key] = pcd[key].cuda() 
                for key in Transform.keys():
                    Transform[key] = Transform[key].cuda() 
                losses, Transform_pred = self._test_one_batch(pcd, Transform)
                batch_size = pcd['src'].size(0)
                num_examples += batch_size
                total_mse_loss += losses['mse'] * batch_size
                total_loss = total_loss + losses['total_loss'] * batch_size
                total_cycle_consistency_loss = total_cycle_consistency_loss + losses['cycle'] * batch_size
                #total_scale_consensus_loss = total_scale_consensus_loss + losses['scale'] * batch_size
                total_corres_accuracy += losses['acc'] * batch_size
                total_discrimination_loss += losses['discrimination'] * batch_size
                total_chamfer_distance += losses['chamfer_distance'] * batch_size  # 累加 Chamfer 距离

                rotations_ab.append(Transform['R_ab'].detach().cpu())
                translations_ab.append(Transform['T_ab'].detach().cpu())
                rotations_ab_pred.append(Transform_pred['R_ab_pred'].detach().cpu())
                translations_ab_pred.append(Transform_pred['T_ab_pred'].detach().cpu())
                eulers_ab.append(Transform['euler_ab'].cpu().numpy())

            end = time.time()
            avg_losses['avg_loss'] = total_loss / num_examples
            avg_losses['avg_cycle'] = total_cycle_consistency_loss / num_examples
            #avg_losses['avg_scale'] = total_scale_consensus_loss / num_examples
            avg_losses['avg_acc'] = total_corres_accuracy / num_examples
            avg_losses['avg_discrimination'] = total_discrimination_loss / num_examples
            avg_losses['avg_mse'] = total_mse_loss / num_examples
            avg_losses['chamfer_distance'] = total_chamfer_distance/ num_examples  
            avg_losses['epoch'] = epoch
            avg_losses['stage'] = 'test'
            avg_losses['elapsed_time'] = (end - start)/num_examples

            Transforms['R_ab'] = torch.cat(rotations_ab, axis=0)
            Transforms['T_ab'] = torch.cat(translations_ab, axis=0)
            Transforms['R_ab_pred'] = torch.cat(rotations_ab_pred, axis=0)
            Transforms['T_ab_pred'] = torch.cat(translations_ab_pred, axis=0)
            Transforms['euler_ab'] = np.degrees(np.concatenate(eulers_ab, axis=0))
            Transforms['euler_ab_pred'] = npmat2euler(Transforms['R_ab_pred'])


        return self.Compute_metrics(avg_losses,Transforms)

    def show(self, test_loader):
        with torch.no_grad():
            self.eval()
            rotations_ab = []
            translations_ab = []
            rotations_ab_pred = []
            translations_ab_pred = []
            for pcd, Transform in tqdm(test_loader):
                for key in pcd.keys():
                    pcd[key] = pcd[key].cuda()
                for key in Transform.keys():
                    Transform[key] = Transform[key].cuda()

                    # 1. 可视化输入的点云数据
                self.visualize_point_cloud(pcd['src'], pcd['tgt'], title='Input Point Clouds')

                _, Transform_pred = self._test_one_batch(pcd, Transform)

                rotations_ab.append(Transform['R_ab'].detach().cpu())
                translations_ab.append(Transform['T_ab'].detach().cpu())
                rotations_ab_pred.append(Transform_pred['R_ab_pred'].detach().cpu())
                translations_ab_pred.append(Transform_pred['T_ab_pred'].detach().cpu())

                # 2. 将预测的变换矩阵应用于源点云，并可视化变换后的源点云
                transformed_source = self.apply_transformation(pcd['src'], Transform_pred)
                self.visualize_point_cloud(transformed_source, pcd['tgt'], title='Transformed Source Point Cloud')
                #break

    def compute_euclidean_distance(self,src, tgt):
        # src: [B, N, 3] - Transformed source point cloud
        # tgt: [B, M, 3] - Target point cloud
        B, N, _ = src.shape
        M = tgt.shape[1]
        
        # Expand and compute pairwise distance between each point in src and tgt
        src_exp = src.unsqueeze(2).expand(B, N, M, 3)
        tgt_exp = tgt.unsqueeze(1).expand(B, N, M, 3)
        
        # Euclidean distance between every pair of points in src and tgt
        dist = torch.norm(src_exp - tgt_exp, dim=-1)  # [B, N, M]
        
        # Find the minimum distance from each source point to the target points
        min_dist, _ = torch.min(dist, dim=2)  # [B, N]
        
        return min_dist

    def show2(self, src, tgt):
        # 1. 可视化输入的点云数据
        self.visualize_point_cloud(src, tgt, title='Input Point Clouds')
        batch_size = src.shape[0]
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        for i in range(self.num_iters):
            is_corr = 0
            rotation_ab_pred_i, translation_ab_pred_i, _, _, _ , _, _= self.forward(src, tgt, is_corr)

            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                          + translation_ab_pred_i

            print(rotation_ab_pred)
            print(translation_ab_pred)
            src_transformed = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
            
            self.visualize_point_cloud(src_transformed, tgt, title='Transformed Source Point Cloud')

            # # 从 [B, 3, N] 转换为 [B, N, 3]
            # src_transformed = src_transformed.permute(0, 2, 1)  # [B, N, 3]
            # tgt = tgt.permute(0, 2, 1)  # 确保 tgt 的维度也是 [B, N, 3]
            #
            #  # 3. 计算应用预测变换后的源点云与目标点云之间的欧氏距离
            # min_distances = self.compute_euclidean_distance(src_transformed, tgt2)  # [B, N]
            #
            # # 4. 打印平均最近点距离
            # avg_min_distance = torch.mean(min_distances)
            # print(f"Iteration {i}: Average nearest point distance = {avg_min_distance.item():.4f}")

    def show3(self, src, tgt, src2, tgt2):
        # 1. 可视化输入的点云数据
        self.visualize_point_cloud(src2, tgt2, title='Original Point Clouds')
        batch_size = src.shape[0]
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        for i in range(self.num_iters):
            is_corr = 0
            rotation_ab_pred_i, translation_ab_pred_i, _, _, _, _, _ = self.forward(src, tgt, is_corr)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            # print(rotation_ab_pred)
            # print(translation_ab_pred)
            src_transformed = transform_point_cloud(src2, rotation_ab_pred, translation_ab_pred)

            #print(src_transformed.shape)
            self.visualize_point_cloud(src_transformed, tgt2, title='Aligned Source Point Cloud')

        return rotation_ab_pred, translation_ab_pred


    def visualize_point_cloud(self, source, target, batch_idx=0, title="Point Cloud"):
        """
        可视化给定批次的source和target点云。
        参数:
        - source: 形状为 [B, 3, N] 的点云数据。
        - target: 形状为 [B, 3, N] 的点云数据。
        - batch_idx: 要可视化的批次索引，默认值为0。
        - title: 窗口标题。
        """
        # 从指定批次索引中提取单个点云
        source_np = source[batch_idx].detach().cpu().numpy()  # 形状 [3, N]
        target_np = target[batch_idx].detach().cpu().numpy()  # 形状 [3, N]

        # 如果数据是 [3, N] 形状，转换为 [N, 3]
        if source_np.shape[0] == 3:
            source_np = source_np.transpose(1, 0)  # 现在是 [N, 3]
        if target_np.shape[0] == 3:
            target_np = target_np.transpose(1, 0)  # 现在是 [N, 3]

        # 确保数据类型为 float32
        source_np = source_np.astype(np.float32)
        target_np = target_np.astype(np.float32)

        source_pcd = o3d.geometry.PointCloud()
        target_pcd = o3d.geometry.PointCloud()

        source_pcd.points = o3d.utility.Vector3dVector(source_np)
        target_pcd.points = o3d.utility.Vector3dVector(target_np)

        source_pcd.paint_uniform_color([1, 0, 0])  # Red for source
        target_pcd.paint_uniform_color([0, 0, 1])  # blue for target

        o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name=title,width=800, height=600)

    def apply_transformation(self, source, Transform_pred):
        # 获取批次大小和点云的数量
        batch_size = source.shape[0]
        num_points = source.shape[2]

        # Apply the predicted rotation and translation to the source point cloud
        R_pred = Transform_pred['R_ab_pred'].detach().cpu().numpy()  # 形状 (B, 3, 3)
        T_pred = Transform_pred['T_ab_pred'].detach().cpu().numpy()  # 形状 (B, 3)

        # 将 source_points 转换为 numpy 数组
        source_points = source.detach().cpu().numpy()  # 形状 (B, 3, N)

        # 初始化存储转换后点云的数组
        transformed_points = np.empty((batch_size, 3, num_points), dtype=np.float32)

        # 对每个批次应用旋转矩阵和平移向量
        for i in range(batch_size):
            transformed_points[i] = R_pred[i] @ source_points[i] + T_pred[i].reshape(3, 1)

        return torch.tensor(transformed_points).cuda()  # 返回形状为 [B, 3, N]

    def save(self, path):
        if torch.cuda.device_count() > 1:
            torch.save(self.T_prediction.module.state_dict(), path)
        else:
            torch.save(self.T_prediction.state_dict(), path)

    def load(self, path):
        self.T_prediction.load_state_dict(torch.load(path))

    def close(self):
        self.writer.close()



class Logger:
    def __init__(self, args):
        self.path = 'checkpoints/' + args.exp_name
        self.fw = open(self.path+'/log', 'a')
        self.fw.write(str(args))
        self.fw.write('\n')
        self.fw.flush()
        print(str(args))
        with open(os.path.join(self.path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def write(self, info):
        #arrow = info['arrow']
        epoch = info['epoch']
        elapsed_time = info['elapsed_time']
        stage = info['stage']
        loss = info['loss']
        #cycle_consistency_loss = info['cycle_consistency_loss']
        #scale_consensus_loss = info['scale_consensus_loss']
        #mse_loss = info['mse_loss']
        #discrimination_loss = info['dis_loss']
        r_ab_mse = info['r_ab_mse']
        r_ab_rmse = info['r_ab_rmse']
        r_ab_mae = info['r_ab_mae']
        t_ab_mse = info['t_ab_mse']
        t_ab_rmse = info['t_ab_rmse']
        t_ab_mae = info['t_ab_mae']
        chamfer_distance = info['chamfer_distance']
        #r_ab_r2_score = info['r_ab_r2_score']
        #t_ab_r2_score = info['t_ab_r2_score']
        r_ab_mie = info['r_ab_mie']  #旋转误差
        t_ab_mie = info['t_ab_mie']  #平移误差
        corres_accuracy = info['corres_accuracy']
        cur_acc = info['cur_acc'] #配准成功率
        text = 'Stage:%s  Epoch: %d, Elapsed_time:%f, Loss: %f, Rot_MSE: %.10f, Rot_RMSE: %.10f, ' \
               'Rot_MAE: %.10f,Trans_MSE: %.10f, Trans_RMSE: %.10f, Trans_MAE: %8.10f, Chamfer_distance: %f, ' \
                'corres_accuracy:%f, r_ab_mie: %.10f, t_ab_mie: %.10f, cur_acc:%f\n' % \
               (stage, epoch, elapsed_time,loss, r_ab_mse, r_ab_rmse, r_ab_mae, t_ab_mse, t_ab_rmse, t_ab_mae, chamfer_distance, corres_accuracy,r_ab_mie,t_ab_mie,cur_acc)
        self.fw.write(text)
        self.fw.flush()
        print(text)

    def close(self):
        self.fw.close()

