import os
import torch
from torch.utils.data import Dataset
import trimesh
import torchvision
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import copy
from scipy.spatial.transform import Rotation as R

#基于曲率和随机混合采样
def hybrid_downsample(pc,num_points,curvature_sample_ratio=0.5):

    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # 计算每个点的曲率（通过与邻居点的距离来衡量）
    distances = np.asarray(pc.compute_nearest_neighbor_distance())
    curvatures = 1 / (distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

    # 按曲率从大到小排序，选取曲率较大的点
    num_curvature_points = int(num_points * curvature_sample_ratio)
    num_random_points = num_points - num_curvature_points

    total_points = len(pc.points)

    # 确保索引不超过点的数量
    curvature_indices = np.argsort(-curvatures)[:num_curvature_points]

    # 随机采样时，排除已被曲率采样的点，确保无重复采样
    remaining_indices = np.setdiff1d(np.arange(total_points), curvature_indices)
    random_indices = np.random.choice(remaining_indices, num_random_points, replace=False)

    # 合并曲率采样和随机采样的点
    all_indices = np.concatenate((curvature_indices, random_indices))


    # 选择采样的点
    sampled_pcd = pc.select_by_index(all_indices)

    # src和tgt分别取点
    sampled_pcd = np.array(sampled_pcd.points)

    return sampled_pcd


class STDataset(Dataset):
    def __init__(self, root_dir,num_points,curvature_sample_ratio=0.5):
        self.root_dir = root_dir
        self.num_points = num_points
        self.data_pairs = []
    
        for subdir, _, files in os.walk(root_dir):
            sources = sorted([f for f in files if f.endswith('_source.ply')])
            targets = sorted([f for f in files if f.endswith('_sv.ply')])
            gts = sorted([f for f in files if f.endswith('_gt.txt')])
            
            for src_file in sources:
                base_name = src_file.replace('_source.ply', '')
                tgt_file = f"{base_name}_sv.ply"
                gt_file = f"{base_name}_gt.txt"
                
                src_path = os.path.join(subdir, src_file)
                tgt_path = os.path.join(subdir, tgt_file)
                gt_path = os.path.join(subdir, gt_file)
                
                # 确保目标文件和ground truth文件都存在
                if os.path.exists(tgt_path) and os.path.exists(gt_path):
                    self.data_pairs.append((src_path, tgt_path, gt_path))
                else:
                    print(f"警告：{src_file} 缺少对应的目标或ground truth文件")
        
    
    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_path, tgt_path, gt_path = self.data_pairs[idx]
        #print(gt_path)
        pc_src = o3d.io.read_point_cloud(src_path)
        pc_tgt = o3d.io.read_point_cloud(tgt_path)
        sampleed_src = hybrid_downsample(pc_src, self.num_points)
        sampleed_tgt = hybrid_downsample(pc_tgt, self.num_points)

        matrix_ab = np.loadtxt(gt_path)
        R_ab = matrix_ab[:3,:3].T
        translation_ab = matrix_ab[:3,3]

        r_ab = R.from_matrix(R_ab)  # 将旋转矩阵转换为 Rotation 对象
        euler_ab = r_ab.as_euler('zyx', degrees=True)  # 提取欧拉角并转换为角度制

        R_ba = R_ab.T
        translation_ba = -R_ba.dot(translation_ab)

        r_ba = R.from_matrix(R_ba)  # 将旋转矩阵转换为 Rotation 对象
        euler_ba = r_ba.as_euler('zyx', degrees=True)  # 提取欧拉角并转换为角度制

        pc_src = sampleed_src.T
        pc_tgt = sampleed_tgt.T


        pcd = {}
        Transform = {}

        pcd['src'] = pc_src.astype('float32')
        pcd['tgt'] = pc_tgt.astype('float32')
        Transform['R_ab'] = R_ab.astype('float32')
        Transform['T_ab'] = translation_ab.astype('float32')
        Transform['euler_ab'] = euler_ab.astype('float32')
        Transform['R_ba'] = R_ba.astype('float32')
        Transform['T_ba'] = translation_ba.astype('float32')
        Transform['euler_ba'] = euler_ba.astype('float32')

        return pcd, Transform
    



if __name__ == "__main__":
    train_path = r'./traindata/train2'
    dataset = STDataset(train_path, num_points=1000, curvature_sample_ratio=0)
    print(f'Dataset size: {len(dataset)}')
    pcData, trans = dataset[0]
    src = pcData['src']
    tgt = pcData['tgt']
    R_ab = trans['R_ab']
    T_ab = trans['T_ab']
    print(src.shape, tgt.shape)
    print(R_ab.shape, T_ab.shape)
    T_ab = T_ab = T_ab.reshape(3, 1)
    src_transformed = np.dot(R_ab, src) + T_ab

    def visualize_point_cloud(source, target, batch_idx=0, title="Point Cloud"):
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

        o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name=title)

    src_transformed = torch.from_numpy(src_transformed)
    tgt = torch.from_numpy(tgt)
    src = torch.from_numpy(src)
    visualize_point_cloud(src.unsqueeze(0),tgt.unsqueeze(0))
    visualize_point_cloud(src_transformed.unsqueeze(0), tgt.unsqueeze(0))



