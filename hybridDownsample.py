import os
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree
import copy

#FPS采样可以保留点云的整体结构
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


class MyDataset(Dataset):
    def __init__(self, root_dir,num_points, partition='train', curvature_sample_ratio=0.5,rot_factor=4):
        self.root_dir = root_dir
        self.num_points = num_points
        self.curvature_sample_ratio = curvature_sample_ratio
        self.partition = partition
        self.rot_factor = rot_factor
        self.files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.ply'):
                    self.files.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 读取点云
        pcd = o3d.io.read_point_cloud(self.files[idx])
        #pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

        points_np = np.asarray(pcd.points)
        normals_np = np.asarray(pcd.normals)

        # 筛选出 Z 坐标在 (-7, 7) 范围内的点的索引
        x_min, x_max = -4, 4
        y_min, y_max = -4, 4
        z_min, z_max = -5, 5

        # 筛选出同时满足 X、Y、Z 范围内的点的索引
        xyz_filtered_indices = np.where(
                (points_np[:, 0] > x_min) & (points_np[:, 0] < x_max) &  # X 范围
                (points_np[:, 1] > y_min) & (points_np[:, 1] < y_max) &  # Y 范围
                (points_np[:, 2] > z_min) & (points_np[:, 2] < z_max)    # Z 范围
            )[0]

        # 根据筛选索引提取点和法向量
        filtered_points = points_np[xyz_filtered_indices]
        filtered_normals = normals_np[xyz_filtered_indices]

        #print(filtered_points.shape[0])

        #创建新的点云对象
        filtered_pc = o3d.geometry.PointCloud()
        filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pc.normals = o3d.utility.Vector3dVector(filtered_normals)

        # 计算每个点的曲率（通过与邻居点的距离来衡量）
        #distances = np.asarray(pcd.compute_nearest_neighbor_distance())
        distances = np.asarray(filtered_pc.compute_nearest_neighbor_distance())
        curvatures = 1 / (distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

        # 按曲率从大到小排序，选取曲率较大的点
        num_curvature_points = int(self.num_points * self.curvature_sample_ratio)
        num_random_points = self.num_points - num_curvature_points

        #total_points = len(pcd.points)
        total_points = len(filtered_pc.points)
        if total_points < self.num_points:
            raise ValueError(f"{self.files[idx]} has only {total_points} points, but {self.num_points} points were requested.")

        # 确保索引不超过点的数量
        curvature_indices = np.argsort(-curvatures)[:num_curvature_points]

        # 随机采样时，排除已被曲率采样的点，确保无重复采样
        remaining_indices = np.setdiff1d(np.arange(total_points), curvature_indices)
        random_indices = np.random.choice(remaining_indices, num_random_points, replace=False)
        random_indices1 = np.random.choice(remaining_indices, num_random_points, replace=False)
        # 合并曲率采样和随机采样的点
        all_indices = np.concatenate((curvature_indices, random_indices))
        all_indices1 = np.concatenate((curvature_indices, random_indices1))
        assert len(all_indices) == self.num_points, f"Sampled {len(all_indices)} points, but {self.num_points} points were expected."

        # 选择采样的点
        # sampled_pcd = pcd.select_by_index(all_indices)
        # sampled_pcd1 = pcd.select_by_index(all_indices1)
        sampled_pcd = filtered_pc.select_by_index(all_indices)
        sampled_pcd1 = filtered_pc.select_by_index(all_indices1)
        #src和tgt分别取点
        src = np.array(sampled_pcd.points)
        tgt = np.array(sampled_pcd1.points)
        #tgt = copy.deepcopy(src)
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
        translation_ab = np.array([np.random.uniform(-12, 12), np.random.uniform(-12, 12), np.random.uniform(-12, 12)])
        translation_ba = -R_ba.dot(translation_ab)

        src = src.T
        tgt = tgt.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        tgt = rotation_ab.apply(tgt.T).T + np.expand_dims(translation_ab, axis=1)

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




if __name__ == '__main__':
    data_dir = r'./traindata/addedtrain'
    dataset = MyDataset(root_dir=data_dir, num_points=168,rot_factor=64,curvature_sample_ratio=0)
    print(f'Dataset size: {len(dataset)}')
    pcData, trans = dataset[358]
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

        source_pcd.paint_uniform_color([0.9, 0, 0])  # Red for source
        target_pcd.paint_uniform_color([0, 0, 0.9])  # blue for target

        o3d.visualization.draw_geometries([source_pcd,target_pcd], window_name=title)
        #o3d.visualization.draw_geometries([target_pcd], window_name=title)


    tgt = torch.from_numpy(pcData['tgt'])
    src = torch.from_numpy(pcData['src'])
    visualize_point_cloud(src.unsqueeze(0),tgt.unsqueeze(0))