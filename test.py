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
    src = np.array(sampled_pcd.points)

    return src

def nearest_neighbor_distances(point_cloud):
    """
    计算点云中每个点到其最近邻点的距离。

    参数:
    point_cloud: np.array, shape = [3, N], 3维坐标的点云数据

    返回:
    distances: list, 每个点到最近邻点的距离
    """
    # 转置点云数据，使其形状为 [N, 3]
    points = point_cloud.T

    # 使用KDTree快速查找最近邻
    tree = KDTree(points)

    # 对每个点查找最近邻点的距离
    distances, _ = tree.query(points, k=2)  # k=2，因为第一个邻居是点自己，第二个是最近邻

    avg_distances = np.mean(distances[:, 1:])
    # 返回每个点到最近邻点的距离（不包括点自己）
    return avg_distances

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


def public_process(pc1,pc2,num_points,num_subsampled_points):
    pc1_points = np.asarray(pc1.points[:num_points]).astype('float32')
    pc2_points = np.asarray(pc2.points[:num_points]).astype('float32')
    pointcloud1, pointcloud2 = farthest_subsample_points(pc1_points.T, pc2_points.T,num_subsampled_points=num_subsampled_points)
    return pointcloud1,pointcloud2


def get_data_path(root_dir):
    data_pairs = []
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
                data_pairs.append((src_path, tgt_path, gt_path))
            else:
                print(f"警告：{src_file} 缺少对应的目标或ground truth文件")

    return data_pairs

def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])



def pointcloud_filter(pcd):
    points_np = np.asarray(pcd.points)
    #normals_np = np.asarray(pcd.normals)

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
    #filtered_normals = normals_np[xyz_filtered_indices]

    #print(filtered_points.shape[0])

    # 创建新的点云对象
    filtered_pc = o3d.geometry.PointCloud()
    filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
    #filtered_pc.normals = o3d.utility.Vector3dVector(filtered_normals)

    return filtered_pc


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
    #加载点云
    num_points = 729
    curvature_sample_ratio = 0
    save_dir = r'./results'
    pc_path = r'./filteredtestdata'
    data_pairs = get_data_path(pc_path)
    csv_file_path = os.path.join('./checkpoints/',f'{args.exp_name}/errors.csv') # 指定你的 CSV 文件路径
    with open(csv_file_path, mode='a+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csvfile.seek(0)
        first_char = csvfile.read(1)
        if not first_char:
            csv_writer.writerow(['Index', 'rotational_error_x', 'rotational_error_y','rotational_error_z','translational_error_x','translational_error_y','translational_error_z'])

        for i in range(len(data_pairs)):
            src_path, tgt_path, gt_path = data_pairs[i]
            matrix_ab = np.loadtxt(gt_path)
            R_gt = matrix_ab[:3, :3].T
            t_gt = matrix_ab[:3, 3]

            pc1 = o3d.io.read_point_cloud(src_path)
            pc2 = o3d.io.read_point_cloud(tgt_path)

            # filter_pc1 = pointcloud_filter(pc1)
            # filter_pc2 = pointcloud_filter(pc2)

            # pc1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            
            # pc2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            # 计算每个点的曲率（通过与邻居点的距离来衡量）
            pc1_distances = np.asarray(pc1.compute_nearest_neighbor_distance())
            pc1_curvatures = 1 / (pc1_distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

            pc2_distances = np.asarray(pc2.compute_nearest_neighbor_distance())
            pc2_curvatures = 1 / (pc2_distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

            # pc1_distances = np.asarray(filter_pc1.compute_nearest_neighbor_distance())
            # pc1_curvatures = 1 / (pc1_distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

            # pc2_distances = np.asarray(filter_pc2.compute_nearest_neighbor_distance())
            # pc2_curvatures = 1 / (pc2_distances + 1e-8)  # 距离的倒数作为曲率的近似值，避免除0

            # 按曲率从大到小排序，选取曲率较大的点
            num_curvature_points = int(num_points * curvature_sample_ratio)
            num_random_points = num_points - num_curvature_points

            pc1_total_points = len(pc1.points)
            pc2_total_points = len(pc2.points)

            # pc1_total_points = len(filter_pc1.points)
            # pc2_total_points = len(filter_pc2.points)

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

            # indices = np.random.choice(len(pc1.points), 7000, replace=False)
            # indices1 = np.random.choice(len(pc2.points), 7000, replace=False)

            # pc1_origin = pc1.select_by_index(indices)
            # pc2_origin = pc2.select_by_index(indices1)

            # pc1_origin = np.array(pc1_origin.points).astype(np.float32)
            # pc2_origin = np.array(pc2_origin.points).astype(np.float32)

            pc1_origin = copy.deepcopy(np.asarray(pc1.points))
            pc2_origin = copy.deepcopy(np.asarray(pc2.points))
            #src和tgt分别取点
            sampled_pcd = pc1.select_by_index(all_indices)
            sampled_pcd1 = pc2.select_by_index(all_indices1)

            # sampled_pcd = filter_pc1.select_by_index(all_indices)
            # sampled_pcd1 = filter_pc2.select_by_index(all_indices1)

            #sampled_pcd,sampled_pcd1 = public_process(pc1,pc2,num_points=1000,num_subsampled_points=729)
            #print(sampled_pcd.shape, sampled_pcd1.shape)
            #转化为numpy
            src = np.array(sampled_pcd.points).astype('float32')
            tgt = np.array(sampled_pcd1.points).astype('float32')
            # src = np.array(sampled_pcd).astype('float32')
            # tgt = np.array(sampled_pcd1).astype('float32')

            # 转换为 PyTorch 张量
            pc1 = torch.from_numpy(src.T)
            pc2 = torch.from_numpy(tgt.T)
            pc1_origin = torch.from_numpy(pc1_origin.T)
            pc2_origin = torch.from_numpy(pc2_origin.T)

            R,t = net.show3(pc1.unsqueeze(0).cuda(), pc2.unsqueeze(0).cuda(),pc1_origin.unsqueeze(0).cuda(),pc2_origin.unsqueeze(0).cuda())
            #print(R.shape, t.shape)

            pc = o3d.io.read_point_cloud(src_path)

            t = t.squeeze(0).detach().cpu().numpy()
            R = R.squeeze(0).detach().cpu().numpy()

            matric = np.eye(4)
            matric[:3, :3] = R
            matric[:3, 3] = t

            pc.transform(matric)

            save_dir = r'./results'
            output_file_path = os.path.join(save_dir, f"{i}_pre.ply")

            # 保存点云到指定文件
            o3d.io.write_point_cloud(output_file_path, pc)
            # 将 R 和 R_gt 转换为欧拉角
            R_euler = rotation_matrix_to_euler_angles(R)
            R_gt_euler = rotation_matrix_to_euler_angles(R_gt)

            # 计算旋转误差（弧度差）
            rotational_error = np.linalg.norm(R_euler - R_gt_euler)
            # 计算旋转误差，分别计算每个轴的误差
            rotational_error_x = np.abs(R_euler[0] - R_gt_euler[0])
            rotational_error_y = np.abs(R_euler[1] - R_gt_euler[1])
            rotational_error_z = np.abs(R_euler[2] - R_gt_euler[2])

            # 计算位移误差（欧几里得距离）
            translational_error = np.linalg.norm(t - t_gt)
            translational_error_x = np.abs(t[0] - t_gt[0])
            translational_error_y = np.abs(t[1] - t_gt[1])
            translational_error_z = np.abs(t[2] - t_gt[2])


             # 将误差写入 CSV 文件
            csv_writer.writerow([gt_path, rotational_error_x, rotational_error_y,rotational_error_z,translational_error_x,translational_error_y,translational_error_z])
            # 打印误差
            # print(f"旋转误差（弧度）：{rotational_error}")
            # print(f"位移误差（单位）：{translational_error}")

if __name__ == '__main__':
    main()

