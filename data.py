import os
import torch
from sklearn.linear_model import RANSACRegressor
from torch.utils.data import Dataset
import trimesh
import torchvision
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import copy
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
from scipy.spatial import cKDTree


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * 2
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = np.random.random(size=(1, 3)) + np.array([[-500, -500, -500]]) * 2 

    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))

    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T

#根据距离找点对
def find_matching_points(src, tgt, num_matches, distance_threshold=0.1):
    # 创建 kd-tree 用于快速最近邻搜索
    tree = cKDTree(tgt.T)

    # 查找 src 中每个点在 tgt 中的最近邻
    distances, indices = tree.query(src.T, k=1)  # k=1 表示找最近的一个点

    # 选择距离小于阈值的点对
    valid_indices = np.where(distances < distance_threshold)[0]

    # 取有效点对
    matched_src = src[:, valid_indices]
    matched_tgt = tgt[:, indices[valid_indices]]

    # 检查有效点对数量
    num_valid_matches = matched_src.shape[1]

    if num_valid_matches < num_matches:
        # 有效点对不足，从源和目标中随机取点
        remaining_num = num_matches - num_valid_matches

        # 随机选择剩余点，避免与有效点对重复
        remaining_indices = np.setdiff1d(np.arange(src.shape[1]), valid_indices)
        random_indices = np.random.choice(remaining_indices, remaining_num, replace=False)

        matched_src = np.concatenate((matched_src, src[:, random_indices]), axis=1)
        matched_tgt = np.concatenate((matched_tgt, tgt[:, indices[random_indices]]), axis=1)
    else:
        # 有效点对超过所需数量，取前 num_matches 个
        matched_src = matched_src[:, :num_matches]
        matched_tgt = matched_tgt[:, :num_matches]

    return matched_src, matched_tgt


def ransac_match(src, tgt, num_matches):

    # 使用 RANSAC 找到最佳匹配
    ransac = RANSACRegressor()  # 设定最小样本比例
    ransac.fit(src.T, tgt.T)  # 转置为 (n_samples, n_features)

    inlier_mask = ransac.inlier_mask_

    matched_src = src[:, inlier_mask]
    matched_tgt = tgt[:, inlier_mask]

    # 如果匹配点对数量不足，随机选择剩余的点
    if matched_src.shape[1] < num_matches:
        additional_indices = np.random.choice(np.arange(src.shape[1]), size=num_matches - matched_src.shape[1],
                                              replace=False)
        matched_src = np.concatenate((matched_src, src[:, additional_indices]), axis=1)
        matched_tgt = np.concatenate((matched_tgt, tgt[:, additional_indices]), axis=1)

    # 如果匹配点对数量超过 num_matches，随机选择指定数量的点
    elif matched_src.shape[1] > num_matches:
        select_indices = np.random.choice(np.arange(matched_src.shape[1]), size=num_matches, replace=False)
        matched_src = matched_src[:, select_indices]
        matched_tgt = matched_tgt[:, select_indices]

    return matched_src, matched_tgt

class PcDataset(Dataset):
    def __init__(self, root_dir,num_points, num_subsampled_points=200,partition='train', curvature_sample_ratio=0.5,rot_factor=256):
        self.root_dir = root_dir
        self.num_points = num_points
        self.partition = partition
        self.rot_factor = rot_factor
        self.curvature_sample_ratio = curvature_sample_ratio
        self.num_subsampled_points = num_subsampled_points
        self.files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.ply'):
                    self.files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        src_mesh = trimesh.load(self.files[idx])
        src_points = src_mesh.vertices.astype('float32')
        indices = np.random.choice(src_points.shape[0], self.num_points, replace=False)
        src = copy.deepcopy(src_points[indices])
        src = src.T
        tgt = copy.deepcopy(src)

        src,tgt = farthest_subsample_points(src, tgt, num_subsampled_points=self.num_subsampled_points)
        if self.partition != 'train':
            np.random.seed(idx)
        anglex = np.random.uniform(-1, 1) * np.pi / self.rot_factor
        angley = np.random.uniform(-1, 1) * np.pi / self.rot_factor
        anglez = np.random.uniform(-1, 1) * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-4, 4)])
        # translation_ab = np.array([np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25), np.random.uniform(-0.25, 0.25)])
        translation_ba = -R_ba.dot(translation_ab)

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        tgt = rotation_ab.apply(tgt.T).T + np.expand_dims(translation_ab, axis=1)
        #寻找对应的点
        #matched_src, matched_tgt = find_matching_points(src, tgt,self.num_subsampled_points,2)

        euler_ab = np.asarray([anglez, angley, anglex]).astype('float32')
        euler_ba = -euler_ab[::-1]

        pcd = {}
        Transform = {}

        pcd['src'] = src.astype('float32')
        pcd['tgt'] = tgt.astype('float32')
        Transform['R_ab'] = R_ab.astype('float32')
        Transform['T_ab'] = translation_ab.astype('float32')
        Transform['euler_ab'] = euler_ab.astype('float32')
        Transform['R_ba'] = R_ba.astype('float32')
        Transform['T_ba'] = translation_ba.astype('float32')
        Transform['euler_ba'] = euler_ba.astype('float32')

        return pcd, Transform

if __name__ == "__main__":
    train_path = r'./traindata/train'
    dataset = PcDataset(train_path, num_points=2000,num_subsampled_points=168)
    print(f'Dataset size: {len(dataset)}')
    pcData, trans = dataset[0]
    print(trans['T_ab'].shape)
    print(pcData['tgt'].shape)
    print(pcData['src'].shape)


    def visualize_point_cloud(source, target, batch_idx=0, title="Point Cloud"):

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

        o3d.visualization.draw_geometries([source_pcd,target_pcd], window_name=title)
        #o3d.visualization.draw_geometries([target_pcd], window_name=title)


    tgt = torch.from_numpy(pcData['tgt'])
    src = torch.from_numpy(pcData['src'])
    visualize_point_cloud(src.unsqueeze(0),tgt.unsqueeze(0))
