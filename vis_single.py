import open3d as o3d
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# 读取 PLY 文件
def read_ply(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    return vertices, normals

# 计算基于目标网格点的距离差，并计算正负值
def compute_distance_diff(mesh_target, normals_target, mesh_from):
    tree = cKDTree(mesh_from)
    dist, idx = tree.query(mesh_target)
    
    # 计算正负距离
    direction_vectors = mesh_from[idx] - mesh_target  # 目标点到查询点的向量差
    dot_products = np.einsum('ij,ij->i', direction_vectors, normals_target)  # 点积计算
    
    # 根据点积确定距离的正负
    signed_distances = dist * np.sign(dot_products)
    
    return signed_distances

# 使用 PyVista 进行同步交互式 3D 可视化
def visualize_synchronized_views(mesh3, dist1, dist2, dist_diff):
    # 创建 PyVista 点云对象，使用 mesh3
    point_cloud = pv.PolyData(mesh3)

    # 创建三个副本，用于不同的距离值
    point_cloud1 = point_cloud.copy(deep=True)
    point_cloud1['Distance Difference'] = dist1

    point_cloud2 = point_cloud.copy(deep=True)
    point_cloud2['Distance Difference'] = dist2

    point_cloud_diff = point_cloud.copy(deep=True)
    point_cloud_diff['Distance Difference'] = dist_diff

    # 创建一个包含三个渲染器的绘图器
    plotter = pv.Plotter(shape=(1, 3), title='Synchronized Views')
    # plotter.window_position = [400, 200]  # 将窗口位置移到屏幕当中

    # 在第一个子图中添加第一个点云
    plotter.subplot(0, 0)
    plotter.add_mesh(point_cloud1, scalars='Distance Difference',
                     point_size=5, render_points_as_spheres=True, cmap='jet',
                     clim=[-1,1])
    plotter.add_text('File3 vs File1', position='upper_edge', font_size=14)
    plotter.show_grid()

    # 在第二个子图中添加第二个点云
    plotter.subplot(0, 1)
    plotter.add_mesh(point_cloud2, scalars='Distance Difference',
                     point_size=5, render_points_as_spheres=True, cmap='jet',
                     clim=[-1,1])
    plotter.add_text('File3 vs File2', position='upper_edge', font_size=14)
    plotter.show_grid()

    # 在第三个子图中添加距离差异
    plotter.subplot(0, 2)
    clim = [np.min(dist_diff), np.max(dist_diff)]  # 包含负值时的颜色映射范围
    plotter.add_mesh(point_cloud_diff, scalars='Distance Difference',
                     point_size=5, render_points_as_spheres=True, cmap='jet',
                     clim=[-1,1])
    plotter.add_text('Difference between Dist1 and Dist2', position='upper_edge', font_size=14)
    plotter.show_grid()

    # 同步三个渲染器的相机
    plotter.link_views()

    # 添加一个公共的标量条
    plotter.add_scalar_bar(title='Distance Difference', n_labels=10, vertical=True)

    # 设置背景为白色
    plotter.set_background('white', all_renderers=True)

    # 显示绘图器
    # plotter.show()
    plotter.show(full_screen=True)

# 主程序

file_path1 = r"C:\Users\78570\Desktop\Align\data\tooth_normal\1727675275572_source.ply"
file_path2 = r'./results/14_pre.ply'
file_path3 = r"C:\Users\78570\Desktop\Align\data\tooth_normal\1727675275572_sv.ply"

# file_path1 = r'./testdatanormal/1727675041665_source.ply'
# file_path2 = r'./results/1_pre.ply'
# file_path3 = r'./testdatanormal/1727675041665_sv.ply'

# file_path1 = r'./testdatanormal/1727675275572_source.ply'
# file_path2 = r'./results/2_pre.ply'
# file_path3 = r'./testdatanormal/1727675275572_sv.ply'

# file_path1 = r'E:\workspace\extract_inner_surface_and_register\0924_demo_gold\gold_1to5\our_method_recon\icp_aligned_source_mesh.ply'
# file_path2 = r'0924_demo_gold/gold_1to5/D0820_mesh_nas_2024_10_06__20_38_28/icp_aligned_source_mesh.ply'
# file_path3 = r'0924_demo_gold/2024-09-30_001_202-UpperJaw_segmented.ply'

# 读取文件
mesh1, normals1 = read_ply(file_path1)
mesh2, normals2 = read_ply(file_path2)
mesh3, normals3 = read_ply(file_path3)

# 计算从 mesh3 的点到 mesh1 和 mesh2 的距离差，并计算正负值
dist1 = compute_distance_diff(mesh3, normals3, mesh1)
dist2 = compute_distance_diff(mesh3, normals3, mesh2)

# 计算距离差异
dist_diff = dist1 - dist2

# 可视化同步的视图
visualize_synchronized_views(mesh3, dist1, dist2, dist_diff)

