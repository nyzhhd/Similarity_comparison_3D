import struct
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image ,ImageDraw,ImageFont
import time
import os
import cv2
from scipy.spatial import KDTree
import datetime 
import sys
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# base_path = '/home/lzy/BoltLoose'   # 部署时需要修改该路径
# sys.path.insert(0,base_path)  # 添加import时的搜索路径

class Sim_3D():
    def __init__(self):
        self.height = 1200           # 图片高度
        self.width = 1920            # 图片宽度
        self.length = 3 * self.height * self.width   # 图片数据长度
        self.p = 0.1                 #  图片存储精度 点云文件与真实坐标之间的比例系数
        self.Sampling_interval = 4   # 点云降采样系数
 
       
    def readxyz_old(self,Image_XYZ_Name, box, p, Sampling_interval):
        point_depth = []
        fd = open(Image_XYZ_Name, "rb")                # 打开文件
        file = fd.read()                                    # 文件尾为空字串
        data = []                                           # 创建列表存储.data文件
        for cell in struct.unpack('%dH' % self.length, file):  # 转换二进制数据
            data.append(cell)                               # 写入二进制数据
        depth_map = np.array(data, dtype=np.uint16).reshape((self.height, self.width, 3), order="C")
        fd.close()                                          # 关闭文件流   

        # 转换为3d点云，只读取box范围内的
        inited = False
        xmin, ymin, xmax, ymax, _, _ = box
        for y in range(int(ymin), int(ymax), Sampling_interval):
            for x in range(int(xmin), int(xmax), Sampling_interval):
                depth_x = depth_map[y][x][0]*p               # 获取深度坐标x
                depth_y = depth_map[y][x][1]*p               # 获取深度坐标y
                depth_z = depth_map[y][x][2]*p               # 获取深度坐标z
                if depth_z!=0:    # 判断无效点
                    if not inited:
                        point_cloud = np.array([[depth_x,depth_y,depth_z]])
                        inited = True
                    else:
                        point_cloud = np.row_stack((point_cloud, [depth_x,depth_y,depth_z]))
                
            try:
                point_cloud    # 判断点云是否存在，如果螺栓区域所有点云点都无效，则point_cloud不存在
            except NameError:
                point_cloud = []  # 定义一个空列表，防止变量不存在时，函数返回报错
                
        return point_cloud,point_depth
    
    def readxyz(self, Image_XYZ_Name, box, p=0.1, Sampling_interval=4):
        #改进的读取点云的函数，可以节省很多时间
        # 使用内存映射读取文件
        with open(Image_XYZ_Name, "rb") as fd:
            # 计算映射大小和偏移
            dtype = np.dtype(np.uint16)
            offset = dtype.itemsize * self.width * self.height * 3  #根据文件格式调整
            fd.seek(0, 2)  # 移动到文件末尾
            file_size = fd.tell()
            assert file_size >= offset, "文件大小不匹配"

            # 创建内存映射
            mm = np.memmap(fd, dtype=dtype, mode='r', shape=(self.height, self.width, 3), order="C")

        # 转换为3D点云，只读取box范围内的
        xmin, ymin, xmax, ymax, _, _ = box
        y_indices = np.arange(ymin, ymax, Sampling_interval, dtype=int)
        x_indices = np.arange(xmin, xmax, Sampling_interval, dtype=int)

        # 构建网格索引
        y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
        depth_points = mm[y_grid, x_grid].reshape(-1, 3) * p

        # 过滤无效点（假设深度值为0表示无效点）
        valid_points = depth_points[depth_points[:, 2] != 0]

        return valid_points
    

    def abnorm_detect_3D(self, img_path1, box1, img_path2, box2, thresh_distance,Sampling_interval = 4):
        # img_path1 应该是标准图的路径，是标准的
        # img_path2 应该是待检测图像的路径，可能是异常的，可能是正常的
        # 相似度检测过程中img_path1和img_path2不建议调换

        # 开始总时间记录
        total_start_time = time.time()
        self.Sampling_interval=Sampling_interval
        mean_distance=-1

        #---------------------获取第一个数据的点云-----------------------------
        start_time = time.time()
        xyz_path1 = os.path.splitext(img_path1)[0]+'.xyz'
        point_cloud1= self.readxyz(xyz_path1, box1, self.p, self.Sampling_interval)
        file_name1 = os.path.splitext(os.path.basename(img_path1))[0]  # 不含后缀的文件名
        read_time1 = time.time() - start_time
        if len(point_cloud1) == 0:
            print('该区域点云无效')
            return 0,-1

        #---------------------获取第二个数据的点云-----------------------------
        start_time = time.time()
        xyz_path2 = os.path.splitext(img_path2)[0]+'.xyz'
        point_cloud2= self.readxyz(xyz_path2, box2, self.p, self.Sampling_interval)
        file_name2 = os.path.splitext(os.path.basename(img_path2))[0]  # 不含后缀的文件名
        read_time2 = time.time() - start_time
        if len(point_cloud2) == 0:
            print('该区域点云无效')
            return 0,-1

        #---------------------预处理点云数据-----------------------------
        start_time = time.time()
        point_cloud1, point_cloud2 = self.center_point_clouds(point_cloud1, point_cloud2)#中心对齐
        point_cloud1=self.filter_pcd(point_cloud1)#滤除噪点
        point_cloud2=self.filter_pcd(point_cloud2)#滤除噪点  
        preprocess_time = time.time() - start_time


        #-----作为对比进行显示，真实使用过程中进行屏蔽-----
        # similarity,mean_distance = self.mean_nearest_distance(point_cloud1, point_cloud2)
        # print("-----配准前的平均距离------", mean_distance)
        # self.draw_pcd(point_cloud1, point_cloud2)

        
        #---------------------比较两个点云的相似度-----------------------------
        start_time = time.time()
        point_cloud1, point_cloud2, icp_mean_error = self.align_point_clouds(point_cloud1, point_cloud2)
        icp_mean_error= round(icp_mean_error, 2)
        similarity_time = time.time() - start_time
        print("-----配准后的平均距离------", icp_mean_error)

        #-------------------------记录时间----------------------------------------
        # 总耗时
        #total_time = time.time() - total_start_time
        total_time =read_time1+read_time2+preprocess_time+similarity_time
        # 写入文件
        with open("time_stats.txt", "a", encoding='utf-8') as file:  # 使用追加模式和UTF-8编码
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file.write(f"\n时间戳: {current_time}\n")
            file.write(f"文件1: {file_name1}\n")
            file.write(f"文件2: {file_name2}\n")
            file.write(f"读取点云1时间: {read_time1}s\n")
            file.write(f"读取点云2时间: {read_time2}s\n")
            file.write(f"预处理时间: {preprocess_time}s\n")
            file.write(f"相似度计算时间: {similarity_time}s\n")
            file.write(f"点云1点数: {len(point_cloud1)}\n")
            file.write(f"点云2点数: {len(point_cloud2)}\n")
            file.write(f"配准前的平均距离: {mean_distance}\n")
            file.write(f"配准后的平均距离: {icp_mean_error}\n")
            file.write(f"总耗时: {total_time}s\n")

        #-------------------------绘制点云----------------------------------------
        #self.draw_pcd(point_cloud1, point_cloud2) #实际过程中无需绘制

        if icp_mean_error < thresh_distance:
            return 1 ,icp_mean_error#1：点云平均距离相差较小，可认为大概率无缺陷
        else:
            return 2 ,icp_mean_error #2：点云平均距离相差较大，可认为大概率有缺陷


    def draw_pcd(self,point_cloud1,point_cloud2):
        # 绘制point_cloud1,point_cloud2点云
        # 创建 Open3D 的 PointCloud 对象
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
        pcd1.paint_uniform_color([1, 0, 0])  # 红色
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(point_cloud2)
        pcd2.paint_uniform_color([0, 1, 0])  # 绿色
        # 可视化设置
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])  # 背景颜色
        vis.get_render_option().point_size = 2  # 点的大小
        # 添加点云到可视化
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)
        # 运行可视化
        vis.run()
        vis.destroy_window()

    def filter_pcd(self, point_cloud, z_threshold=60, edge_threshold=0.1):
        # 将点云转换为Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # 滤除深度太远的点
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < z_threshold)[0])

        # 统计滤波去除噪声
        nb_neighbors = 30  # 邻域内的点数
        std_ratio = 2.0    # 标准差倍数
        statistical_filtered = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)[0]

        # 转换为numpy数组
        filtered_points = np.asarray(statistical_filtered.points)

        # 滤除边缘点
        min_bound = filtered_points.min(axis=0) + edge_threshold
        max_bound = filtered_points.max(axis=0) - edge_threshold
        filtered_points = filtered_points[
            np.all(filtered_points > min_bound, axis=1) &
            np.all(filtered_points < max_bound, axis=1)
        ]

        return filtered_points
    
    def center_point_clouds(self,pc1, pc2):
        # 将两个点云的中心对齐。
        center1 = np.mean(pc1, axis=0)
        center2 = np.mean(pc2, axis=0)

        # 将两个点云的中心移动到原点
        center_aligned_pc1 = pc1 - center1
        center_aligned_pc2 = pc2 - center2

        return center_aligned_pc1, center_aligned_pc2
    

    def align_point_clouds(self,A, B, max_iterations=20, tolerance=0.001):
        """
        使用迭代最近点（ICP）算法对齐两个点云。

        参数:
        A, B -- 输入的两个点云（Nx3和Mx3）
        max_iterations -- 最大迭代次数
        tolerance -- 迭代终止的容差

        返回:
        src_aligned, dst_aligned -- 对齐后的两个点云
        mean_error -- 平均距离误差
        """
        # 可以先用一个大的框求平移和旋转矩阵进行配准，然后再对感兴趣的区域进行旋转变换
        src = np.copy(A)
        dst = np.copy(B)

        prev_error = 0
        # 使用 KD-树优化最近点搜索
        kd_tree = cKDTree(dst)
        for i in range(max_iterations):
            # 使用 KD-树找到最近的点
            distances, indices = kd_tree.query(src, k=1)
            closest_points = dst[indices]

            # 计算平移和旋转
            mean_src = np.mean(src, axis=0)
            mean_dst = np.mean(closest_points, axis=0)
            src_centered = src - mean_src
            dst_centered = closest_points - mean_dst
            H = np.dot(src_centered.T, dst_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)

            t = mean_dst.T - np.dot(R, mean_src.T)

            # 更新点云A
            src = np.dot(src, R.T) + t.T

            # 检查是否收敛
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # 对齐B到A
        dst_aligned = np.dot(dst - mean_dst, R.T) + mean_src

        # 返回对齐后的两个点云和平均误差
        return src, dst_aligned, mean_error 
    

    def mean_nearest_distance(self,point_cloud1, point_cloud2):
        #这种方法计算的是一个点云中每个点到另一个点云中最近点的平均距离
        tree = KDTree(point_cloud2)
        distances, _ = tree.query(point_cloud1)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        # 检查是否所有距离都相等
        if min_distance == max_distance:
            return 1.0 if min_distance == 0 else 0.0
        # 标准化距离
        normalized_distance = (distances - min_distance) / (max_distance - min_distance)
        # 计算相似度得分
        similarity_score = 1 - normalized_distance.mean()
        return similarity_score,distances.mean()
    

    def detect_dissimilar_regions_2d(self, standard_cloud, test_cloud, grid_size=1.0, similarity_threshold=0.8):
        #没用这个方法
        # 对齐点云
        standard_cloud,aligned_test_cloud, _ = self.align_point_clouds(standard_cloud,test_cloud)
        
        # 计算点云覆盖的范围
        min_x, min_y = np.min(standard_cloud, axis=0)[:2] - grid_size
        max_x, max_y = np.max(standard_cloud, axis=0)[:2] + grid_size
        
        # 初始化差异区域列表
        dissimilar_regions = []
        
        # 迭代网格
        x = min_x
        while x < max_x:
            y = min_y
            while y < max_y:
                # 确定当前网格的边界
                grid_bounds = np.array([[x, y], [x + grid_size, y + grid_size]])
                
                # 选择在当前网格内的点
                standard_indices = np.where(
                    (standard_cloud[:, 0] >= grid_bounds[0, 0]) & (standard_cloud[:, 0] < grid_bounds[1, 0]) &
                    (standard_cloud[:, 1] >= grid_bounds[0, 1]) & (standard_cloud[:, 1] < grid_bounds[1, 1])
                )[0]
                
                test_indices = np.where(
                    (aligned_test_cloud[:, 0] >= grid_bounds[0, 0]) & (aligned_test_cloud[:, 0] < grid_bounds[1, 0]) &
                    (aligned_test_cloud[:, 1] >= grid_bounds[0, 1]) & (aligned_test_cloud[:, 1] < grid_bounds[1, 1])
                )[0]
                
                # 如果网格内有点，则计算平均距离
                if len(standard_indices) > 0 and len(test_indices) > 0:
                    standard_points = standard_cloud[standard_indices]
                    test_points = aligned_test_cloud[test_indices]
                    
                    # 使用KDTree查找最近点并计算平均距离
                    tree = KDTree(test_points)
                    distances, _ = tree.query(standard_points, k=1)
                    average_distance = np.mean(distances)
                    
                    # 如果平均距离大于阈值，则记录差异
                    if average_distance > similarity_threshold:
                        dissimilar_regions.append((x, y, x + grid_size, y + grid_size))
                
                y += grid_size
            x += grid_size
        
        return dissimilar_regions



if __name__ == '__main__':

    pass 

