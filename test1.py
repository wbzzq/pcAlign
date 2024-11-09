import copy
import os
import gc
import argparse
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import DataLoader
from model import DIT
import trimesh
from scipy.spatial.transform import Rotation
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.distance import minkowski
import csv
import json

def calculate_mse_with_nearest(point_cloud_1, point_cloud_2):
    # 使用 KDTree 查找最近邻
    tree = KDTree(point_cloud_2)
    
    # 将点云 1 中的每个点与点云 2 中的最近邻点配对
    squared_differences = []
    for p in point_cloud_1:
        distance, index = tree.query(p)  # 查找最近邻点
        nearest_point = point_cloud_2[index]
        squared_difference = np.linalg.norm(p - nearest_point)**2
        squared_differences.append(squared_difference)
    
    # 计算均方误差
    mse = np.mean(squared_differences)
    return mse

def calculate_rre(R, R_gt):
    product = np.dot(R.T, R_gt)  # R的转置乘以R_gt
    trace_value = np.trace(product)
    rre = np.arccos((trace_value - 1) / 2)
    return np.degrees(rre)

def read_mat_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 提取 mat 行并去掉 "mat: " 前缀
    mat_line = lines[0].strip().split(": ")[1]
    
    # 将字符串转换为一维数组
    mat_values = np.fromstring(mat_line, sep=' ')
    
    # 转换为 4x4 矩阵
    mat_matrix = mat_values.reshape((4, 4))
    
    return mat_matrix

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp3', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='N',
                        help='control geo_scores and feature_scores')
    parser.add_argument('--n_emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=12, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_iters', type=int, default=1, metavar='N',
                        help='Num of iters to run inference')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='N',
                        help='Discount factor to compute the loss')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--temp_factor', type=float, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--lr', type=float, default=0.00003, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=4327, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize the registration process')
    parser.add_argument('--cycle_consistency_loss', type=float, default=0.1, metavar='N',
                        help='cycle consistency loss')
    parser.add_argument('--discrimination_loss', type=float, default=0.5, metavar='N',
                        help='discrimination loss')
    parser.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor of rotation')
    parser.add_argument('--model_path', type=str, default='models/train_best.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch when training)')
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
                        help='Size of batch when testing)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--n_points', type=int, default=168, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--corres_mode', action='store_true', default=False,
                        help='decide whether use GMCCE or not')
    parser.add_argument('--GMCCE_Sharp', type=float, default=30, metavar='N',
                        help='The Sharp of GMCCE module')
    parser.add_argument('--GMCCE_Thres', type=float, default=0.6, metavar='N',
                        help='The Threshold of GMCCE module')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='decide whether load model or not')
    parser.add_argument('--token_dim', default=64, type=int, metavar='PCT',
                        help='the token dim')

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False #关闭CuDNN库的自动优化功能
    torch.backends.cudnn.deterministic = True #使CuDNN库在确定性模式下运行,这样每次运行时结果都是一样的
    torch.backends.cudnn.enabled = True #启用CuDNN库
    torch.manual_seed(args.seed) #设置PyTorch的随机数生成器的种子
    torch.cuda.manual_seed_all(args.seed) #设置所有GPU的随机数生成器的种子
    np.random.seed(args.seed) #设置NumPy的随机数生成器的种子。

    net = DIT(args).cuda()
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    
    src_path = r'./test/ptcloud_1.ply'
    tgt_path = r'./test/ptcloud_2.ply'
    trans_path = r'./output/1_2.txt'
    num_points = 729
    curvature_sample_ratio = 0

    matrix_ab = read_mat_from_txt(trans_path)
    R_gt = matrix_ab[:3, :3].T
    t_gt = matrix_ab[:3, 3]
    pc1 = o3d.io.read_point_cloud(src_path)
    pc2 = o3d.io.read_point_cloud(tgt_path)

    pc1_distances = np.asarray(pc1.compute_nearest_neighbor_distance())
    pc1_curvatures = 1 / (pc1_distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

    pc2_distances = np.asarray(pc2.compute_nearest_neighbor_distance())
    pc2_curvatures = 1 / (pc2_distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

    # 按曲率从大到小排序，选取曲率较大的点
    num_curvature_points = int(num_points * curvature_sample_ratio)
    num_random_points = num_points - num_curvature_points

    pc1_total_points = len(pc1.points)
    pc2_total_points = len(pc2.points)

    # 确保索引不超过点的数量
    pc1_curvature_indices = np.argsort(-pc1_curvatures)[:num_curvature_points]
    pc2_curvature_indices = np.argsort(-pc2_curvatures)[:num_curvature_points]

    # 随机采样时，排除已被曲率采样的点，确保无重复采样
    remaining_indices = np.setdiff1d(np.arange(pc1_total_points), pc1_curvature_indices)
    random_indices = np.random.choice(remaining_indices, num_random_points, replace=False)
    remaining_indices1 = np.setdiff1d(np.arange(pc2_total_points), pc2_curvature_indices)
    random_indices1 = np.random.choice(remaining_indices1, num_random_points, replace=False)

    # 合并曲率采样和随机采样的点
    all_indices = np.concatenate((pc1_curvature_indices, random_indices))
    all_indices1 = np.concatenate((pc2_curvature_indices, random_indices1))

    pc1_origin = copy.deepcopy(np.asarray(pc1.points))
    pc2_origin = copy.deepcopy(np.asarray(pc2.points))
    
    #src和tgt分别取点
    sampled_pcd = pc1.select_by_index(all_indices)
    sampled_pcd1 = pc2.select_by_index(all_indices1)

     #转化为numpy
    src = np.array(sampled_pcd.points).astype('float32')
    tgt = np.array(sampled_pcd1.points).astype('float32')

    # 转换为 PyTorch 张量
    pc1 = torch.from_numpy(src.T)
    pc2 = torch.from_numpy(tgt.T)
    pc1_origin = torch.from_numpy(pc1_origin.T)
    pc2_origin = torch.from_numpy(pc2_origin.T)

    R,t = net.show3(pc1.unsqueeze(0).cuda(), pc2.unsqueeze(0).cuda(),pc1_origin.unsqueeze(0).cuda(),pc2_origin.unsqueeze(0).cuda())

    pc = o3d.io.read_point_cloud(src_path)
    pcd = o3d.io.read_point_cloud(tgt_path)

    t = t.squeeze(0).detach().cpu().numpy()
    R = R.squeeze(0).detach().cpu().numpy()

    matric = np.eye(4)
    matric[:3, :3] = R
    matric[:3, 3] = t

    pc.transform(matric)
    pc = np.asarray(pc.points).astype('float32')
    pcd = np.asarray(pcd.points).astype('float32')
    
    mse = calculate_mse_with_nearest(pc,pcd)
    rre = calculate_rre(R,R_gt)
    rte = np.linalg.norm(t - t_gt)

    print(mse)
    print(rre)
    print(rte)

if __name__ == '__main__':
    main()

