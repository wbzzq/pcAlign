import os
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import copy
import open3d as o3d
import torch

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.001):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


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

class publicDataset(Dataset):
    def __init__(self, root_dir,num_points, num_subsampled_points=729, partition='train',
                 gaussian_noise=False, rot_factor=4,down=True):
        super(publicDataset, self).__init__()
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.rot_factor = rot_factor
        self.subsampled = down
        self.files = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.ply'):
                    self.files.append(os.path.join(subdir, file))

    def __getitem__(self, item):
        pointcloud = o3d.io.read_point_cloud(self.files[item])
        pointcloud = np.asarray(pointcloud.points[:self.num_points]).astype('float32')
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform(-1,1) * np.pi / self.rot_factor
        angley = np.random.uniform(-1,1) * np.pi / self.rot_factor
        anglez = np.random.uniform(-1,1) * np.pi / self.rot_factor
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
        translation_ab = np.array([np.random.uniform(-13, 13), np.random.uniform(-13, 13),
                                   np.random.uniform(-13, 13)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T
        pointcloud2 = copy.deepcopy(pointcloud1)

        if self.gaussian_noise != 0:
            pointcloud1 = jitter_pointcloud(pointcloud1,clip=self.gaussian_noise)
            pointcloud2 = jitter_pointcloud(pointcloud2,clip=self.gaussian_noise)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                num_subsampled_points=self.num_subsampled_points)
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud2.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex]).astype('float32')
        euler_ba = -euler_ab[::-1]

        index = np.arange(pointcloud2.shape[-1])
        index = np.random.permutation(index)
        pointcloud2 = pointcloud2.T
        pointcloud2 = pointcloud2[index]
        pointcloud2 = pointcloud2.T
       
        pcd = {}
        Transform = {}

        pcd['src'] = pointcloud1.astype('float32')
        pcd['tgt'] = pointcloud2.astype('float32')
        Transform['R_ab'] = R_ab.astype('float32')
        Transform['T_ab'] = translation_ab.astype('float32')
        Transform['euler_ab'] = euler_ab.astype('float32')
        Transform['R_ba'] = R_ba.astype('float32')
        Transform['T_ba'] = translation_ba.astype('float32')
        Transform['euler_ba'] = euler_ba.astype('float32')

        return pcd, Transform

    def __len__(self):
        return len(self.files)
    
if __name__ == '__main__':
    data_dir = r'./traindata/train1'
    dataset = publicDataset(root_dir=data_dir, num_points=4000,num_subsampled_points=2916,gaussian_noise=True,rot_factor=64,down=True)
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

        source_pcd.paint_uniform_color([0.9, 0, 0])  # Red for source
        target_pcd.paint_uniform_color([0, 0, 0.9])  # blue for target

        o3d.visualization.draw_geometries([source_pcd,target_pcd], window_name=title)
        #o3d.visualization.draw_geometries([target_pcd], window_name=title)


    tgt = torch.from_numpy(pcData['tgt'])
    src = torch.from_numpy(pcData['src'])
    visualize_point_cloud(src.unsqueeze(0),tgt.unsqueeze(0))