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
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import DBSCAN


# base_path = '/home/lzy/BoltLoose'   # 部署时需要修改该路径
# sys.path.insert(0,base_path)  # 添加import时的搜索路径

class Sim_3D():
    def __init__(self):
        self.height = 1200           # 图片高度
        self.width = 1920            # 图片宽度
        self.length = 3 * self.height * self.width   # 图片数据长度
        self.p = 0.1                 #  图片存储精度 点云文件与真实坐标之间的比例系数
        self.Sampling_interval = 4   # 点云降采样系数
        self.XYZ_Name=''
        self.kdtree=[]
 
       
    def readxyz1(self,Image_XYZ_Name, box, p, Sampling_interval):
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
                
        return point_cloud
    
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
        # print('mm',mm) 
        # print(y_grid, x_grid) 

        # 过滤无效点（假设深度值为0表示无效点）
        valid_points = depth_points[depth_points[:, 2] != 0]
        return valid_points


    def get_kdtree_from_xyz(self):
        with open(self.XYZ_Name, "rb") as fd:
            # 计算映射大小和偏移
            dtype = np.dtype(np.uint16)
            offset = dtype.itemsize * self.width * self.height * 3  # 根据文件格式调整
            fd.seek(0, 2)  # 移动到文件末尾
            file_size = fd.tell()
            assert file_size >= offset, "文件大小不匹配"
            # 创建内存映射
            a = np.memmap(fd, dtype=dtype, mode='r', shape=(self.height, self.width, 3), order="C")
            
        # 将所有点的坐标转换为KD树需要的格式，并除以p
        points = np.reshape(a, (-1, 3)) 
        self.kdtree = KDTree(points)
           


    def calculate_box_from_points(self, top_left, bottom_right, p=0.1):
        kd_tree = self.kdtree
        
        # 查找最接近的点
        top_left_idx = kd_tree.query(top_left / p)[1]
        bottom_right_idx = kd_tree.query(bottom_right / p)[1]
        
        # 将一维索引转换回二维索引
        matching_indice1 = np.unravel_index(top_left_idx, (self.height, self.width))
        matching_indice2 = np.unravel_index(bottom_right_idx, (self.height, self.width))
        
        return matching_indice1, matching_indice2



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
        point_cloud1, point_cloud2, icp_mean_error,_ = self.align_point_clouds(point_cloud1, point_cloud2)
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
        self.draw_pcd(point_cloud1, point_cloud2) #实际过程中无需绘制

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
    def draw_pcd_single(self,point_cloud):
        # 绘制point_cloud1,point_cloud2点云
        # 创建 Open3D 的 PointCloud 对象
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_cloud)
        pcd1.paint_uniform_color([1, 0, 0])  # 红色
        # 可视化设置
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0.1, 0.1, 0.1])  # 背景颜色
        vis.get_render_option().point_size = 2  # 点的大小
        # 添加点云到可视化
        vis.add_geometry(pcd1)
        # 运行可视化
        vis.run()
        vis.destroy_window()

    
    
            
   

    def filter_pcd(self, point_cloud, z_threshold=30, edge_threshold=0.1, eps=1, min_samples=9):
        # 将点云转换为Open3D点云对象
        import colorsys
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # 滤除深度太远的点
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < z_threshold)[0])

        # 统计滤波去除噪声
        nb_neighbors = 50  # 邻域内的点数
        std_ratio = 2.0    # 标准差倍数
        statistical_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

        # 转换为numpy数组
        filtered_points = np.asarray(statistical_filtered.points)

        # 滤除边缘点
        min_bound = filtered_points.min(axis=0) + edge_threshold
        max_bound = filtered_points.max(axis=0) - edge_threshold
        filtered_points = filtered_points[
            np.all(filtered_points > min_bound, axis=1) &
            np.all(filtered_points < max_bound, axis=1)
        ]

        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
        labels = clustering.labels_

        # 找出所有聚类（不包括噪声点）
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # 如果没有找到聚类，返回所有点
        if len(unique_labels) == 0:
            print("Warning: No clusters found. Returning all points.")
            return filtered_points

        # 为每个聚类生成不同的颜色
        num_clusters = len(unique_labels)
        colors = [colorsys.hsv_to_rgb(i / num_clusters, 1, 1) for i in range(num_clusters)]
        color_map = dict(zip(unique_labels, colors))
        
        # 为噪声点指定颜色（灰色）
        color_map[-1] = (0.7, 0.7, 0.7)

        # 为每个点分配颜色
        point_colors = np.array([color_map[label] for label in labels])

        # 可视化
        # 原始点云
        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
        pcd_original.paint_uniform_color([0.5, 0.5, 0.5])  # 深灰色

        # 聚类后的点云
        pcd_clustered = o3d.geometry.PointCloud()
        pcd_clustered.points = o3d.utility.Vector3dVector(filtered_points)
        pcd_clustered.colors = o3d.utility.Vector3dVector(point_colors)

        # 可视化
        o3d.visualization.draw_geometries([pcd_original, pcd_clustered],
                                        window_name="Point Cloud Clustering",
                                        width=800, height=600)

        return filtered_points
    # def filter_pcd(self, point_cloud, z_threshold=30, edge_threshold=0.1):
    #     # 将点云转换为Open3D点云对象
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(point_cloud)

    #     # 滤除深度太远的点
    #     # print(np.asarray(pcd.points)[:, 2])
    #     pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < z_threshold)[0])

    #     # 统计滤波去除噪声
    #     nb_neighbors = 50  # 邻域内的点数
    #     std_ratio = 2.0    # 标准差倍数
    #     statistical_filtered = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)[0]

    #     # 转换为numpy数组
    #     filtered_points = np.asarray(statistical_filtered.points)

    #     # 滤除边缘点
    #     min_bound = filtered_points.min(axis=0) + edge_threshold
    #     max_bound = filtered_points.max(axis=0) - edge_threshold
    #     filtered_points = filtered_points[
    #         np.all(filtered_points > min_bound, axis=1) &
    #         np.all(filtered_points < max_bound, axis=1)
    #     ]

    #     return filtered_points
    
    def center_point_clouds2(self,pc1, pc2):
        # 将两个点云的中心对齐。
        center1 = np.mean(pc1, axis=0)
        center2 = np.mean(pc2, axis=0)

        # 将两个点云的中心移动到原点
        center_aligned_pc1 = pc1 - center1+center2
        center_aligned_pc2 = pc2 

        return center_aligned_pc1, center_aligned_pc2
    
    def center_point_clouds(self,pc1, pc2):
        # 将两个点云的中心对齐。
        center1 = np.mean(pc1, axis=0)
        center2 = np.mean(pc2, axis=0)

        # 将两个点云的中心移动到原点
        center_aligned_pc1 = pc1 - center1
        center_aligned_pc2 = pc2 - center2

        return center_aligned_pc1, center_aligned_pc2
    

    def align_point_clouds(self,A, B, max_iterations=20, tolerance=0.001):
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

            #更新点云A
            src = np.dot(src, R.T) + t.T

            # 检查是否收敛
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break

            prev_error = mean_error

        # 对齐B到A
        R_T=R.T
        dst_aligned = np.dot(dst - mean_dst, R_T) + mean_src

        # dst_2 = np.dot((dst_aligned - mean_src), R) + mean_dst
        # _ , mean_error = self.mean_nearest_distance(dst_aligned, dst_2)
        # print('-----------',mean_error)
        # self.draw_pcd(dst, B)

        RT = {
                    "mean_dst": mean_dst,  
                    "R_T":  R_T,  
                    "mean_src":mean_src
             }

        mean_error= round(mean_error, 2)
        return src, dst_aligned, mean_error ,RT
    

    def align_point_clouds2(self,A, B,src_true ,dst_true, max_iterations=20, tolerance=0.001):
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

            #更新点云A
            src = np.dot(src, R.T) + t.T
            src_true = np.dot(src_true, R.T) + t.T

            # 检查是否收敛
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break

            prev_error = mean_error

        # 对齐B到A
        R_T=R.T
        dst_aligned = np.dot(dst - mean_dst, R_T) + mean_src
        dst_true_aligned=np.dot(dst_true - mean_dst, R_T) + mean_src

        # dst_2 = np.dot((dst_aligned - mean_src), R) + mean_dst
        # _ , mean_error = self.mean_nearest_distance(dst_aligned, dst_2)
        # print('-----------',mean_error)
        # self.draw_pcd(dst, B)

        RT = {
                    "mean_dst": mean_dst,  
                    "R_T":  R_T,  
                    "mean_src":mean_src
             }

        mean_error= round(mean_error, 2)
        return src_true , dst_true_aligned,dst_aligned, mean_error ,RT
    

    def mean_nearest_distance(self,point_cloud1, point_cloud2):
        #这种方法计算的是一个点云中每个点到另一个点云中最近点的平均距离
        tree = KDTree(point_cloud2)
        distances, _ = tree.query(point_cloud1)
        distances_mean=distances.mean()
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        # 检查是否所有距离都相等
        if min_distance == max_distance:
            if min_distance == 0:
                return 1 , distances_mean  
            else  :
                return 0 , distances_mean
        # 标准化距离
        normalized_distance = (distances - min_distance) / (max_distance - min_distance)
        # 计算相似度得分
        similarity_score = 1 - normalized_distance.mean()

        distances_mean= round(distances_mean, 2)
        return similarity_score , distances_mean
    
    
    def deform_detect_3D(self, norm_file, abnorm_file, Sampling_interval=4,ratio=4):
        #3D变形检测
        start_time = time.time()
        abnorm_box = []
        self.Sampling_interval = Sampling_interval
        xyz_path1 = os.path.splitext(norm_file)[0] + '.xyz'
        standard_cloud = self.readxyz(xyz_path1, [0, 0, 1920-0, 1200-0, 0, 0], 0.1, self.Sampling_interval)
        xyz_path2 = os.path.splitext(abnorm_file)[0] + '.xyz'
        self.XYZ_Name=xyz_path2
        test_cloud = self.readxyz(xyz_path2, [0, 0, 1920-0, 1200-0, 0, 0], 0.1, self.Sampling_interval)
        if ratio!=1:
            standard_cloud2 = self.readxyz(xyz_path1, [0, 0, 1920-0, 1200-0, 0, 0], 0.1, self.Sampling_interval/ratio)
            test_cloud2 = self.readxyz(xyz_path2, [0, 0, 1920-0, 1200-0, 0, 0], 0.1, self.Sampling_interval/ratio)
            test_cloud2 = self.create_pit_in_cloud(test_cloud2, (900,900) , 5 , 10)
            test_cloud2 = self.create_pit_in_cloud(test_cloud2, (1000,900), 10, 10)
            test_cloud2 = self.create_pit_in_cloud(test_cloud2, (1100,900), 15, 10)
            test_cloud2 = self.create_pit_in_cloud(test_cloud2, (1100,1000),20, 10)
            test_cloud2 = self.create_pit_in_cloud(test_cloud2, (850,1000), 25, 10)
            test_cloud2 = self.create_pit_in_cloud(test_cloud2, (950,1000), 30, 10)
        else:
            pass
            test_cloud = self.create_pit_in_cloud(test_cloud, (900,900  ), 28, 25)
            test_cloud = self.create_pit_in_cloud(test_cloud, (1100,1000), 25, 25)
            test_cloud = self.create_pit_in_cloud(test_cloud, (1000,900 ), 20, 34)
            test_cloud = self.create_pit_in_cloud(test_cloud, (850,1000 ), 20, 34)
        read_time = time.time() - start_time 

        # np.savetxt("test_cloud.txt", test_cloud)
        

        self.draw_pcd(standard_cloud, test_cloud) #实际过程中无需绘制 
        self.draw_pcd_single(standard_cloud) #实际过程中无需绘制   
          
        # 对齐点云
        start_time = time.time() 
        if ratio==1:
            standard_cloud, aligned_test_cloud, mean_error ,RT= self.align_point_clouds(standard_cloud,test_cloud)
            test_cloud_sampling = aligned_test_cloud
        else:
            standard_cloud, aligned_test_cloud, test_cloud_sampling ,mean_error ,RT= self.align_point_clouds2(standard_cloud,test_cloud,standard_cloud2,test_cloud2)
        similarity_time = time.time() - start_time
        self.draw_pcd(standard_cloud, aligned_test_cloud) #实际过程中无需绘制
        # print(mean_error)
        # print(self.mean_nearest_distance(standard_cloud, aligned_test_cloud))
        
        

        #TODO: 使得对齐后的点云standard_cloud[[x1,y1,z1],[x2,y2,z2]...], aligned_test_cloud[[x1,y1,z1],[x2,y2,z2]...]在Z上相减。注意standard_cloud,test_cloud里面的点数量可能不一样，并且xy也不一定完全相等，只需以test_cloud为基准,找到最近的xy平面上的standard_cloud的z,以此求出z_diff,把test_cloud的z替换为z_diff。获得[[x1,y1,z_diff1],[x2,y2,z_diff2]...]，然后对于z_diff距离太小设置为0，然后在xy平面对有数值的z_diff进行聚类，获得聚类得到框，存储到abnorm_box
        start_time = time.time() 
        # 使用standard_cloud构建KDTree
        tree = KDTree(standard_cloud[:, :2])

        # 计算最近点的索引和距离
        distances, indices = tree.query(aligned_test_cloud[:, :2], k=1)

        # 计算Z轴上的差异
        z_diffs = aligned_test_cloud[:, 2] - standard_cloud[indices.flatten(), 2]

        # 应用阈值，大于等于4保留原值，否则设置为0
        z_diffs_abs = np.abs(z_diffs)
        condition = (z_diffs_abs >= 5) & (z_diffs_abs <= 18)
        z_diffs_filtered = np.where(condition, z_diffs, 0)

        # 构建结果数组
        z_diffs_array = np.hstack((aligned_test_cloud[:, :2], z_diffs_filtered.reshape(-1, 1)))

        # 移除z_diff为0的点
        nonzero_diffs = z_diffs_array[z_diffs_array[:, 2] != 0]

        print('---',nonzero_diffs)
        np.savetxt("nonzero_diffs.txt", nonzero_diffs)
        if len(nonzero_diffs)==0:
            return abnorm_box
        
        z_time = time.time() - start_time

        self.draw_pcd_single(z_diffs_array) #实际过程中无需绘制
        start_time = time.time()
        # 使用DBSCAN进行聚类，并强制使用KD树
        clustering = DBSCAN(eps=10, min_samples=16*ratio, algorithm='kd_tree').fit(nonzero_diffs[:, :2])
        labels = clustering.labels_

        # 根据聚类结果计算边界框
        for label in np.unique(labels):
            if label == -1:
                continue  # 忽略噪声点
            cluster = nonzero_diffs[labels == label, :2]
            xmin, ymin = np.min(cluster, axis=0)
            xmax, ymax = np.max(cluster, axis=0)
            abnorm_box.append([xmin, ymin, xmax, ymax,'abnorm','0'])
            
        print(abnorm_box)
        DBSCAN_time = time.time() - start_time 

        # 开始绘图
        # 根据聚类结果设置颜色
        import matplotlib.pyplot as plt
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # 将噪声点设为黑色
        colors = colors[:, :3]  # 仅保留RGB值
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nonzero_diffs[:, :3])  # 使用X, Y坐标和Z差异
        pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置颜色
        # 可视化点云
        o3d.visualization.draw_geometries([pcd])


        start_time = time.time()     
        self.get_kdtree_from_xyz()
        abnorm_box = self.RT_adjacent_boxes(abnorm_box,test_cloud_sampling ,test_cloud,RT) #变换到图像上的坐标        
        print(abnorm_box)
        

        abnorm_box = self.merge_adjacent_boxes(abnorm_box) #合并边或角相邻的框,滤除太大的变形
        print(abnorm_box)
        processbox_time = time.time() - start_time
         


        #-------------------------记录时间----------------------------------------
        total_time =read_time+similarity_time+z_time+DBSCAN_time+processbox_time# 总耗时
        # 写入文件
        with open("time_stats.txt", "a", encoding='utf-8') as file:  # 使用追加模式和UTF-8编码
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            file.write(f"\n时间戳: {current_time}\n")
            file.write(f"文件1: {norm_file}\n")
            file.write(f"文件2: {abnorm_file}\n")
            file.write(f"读取点云时间: {read_time}s\n")
            file.write(f"配准时间: {similarity_time}s\n")
            file.write(f"计算z距离差时间: {z_time}s\n")
            file.write(f"聚类时间: {DBSCAN_time}s\n")
            file.write(f"处理框的时间: {processbox_time}s\n")
            file.write(f"总耗时: {total_time}s\n")

        return abnorm_box
    
    def crop_point_cloud(self, point_cloud, box):
        cropped_cloud = []
        for point in point_cloud:
            x, y, z = point
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                cropped_cloud.append([x, y, z])
        return cropped_cloud

    
    def RT_adjacent_boxes(self,boxes,cloud,cloud2,RT):

        mean_dst = RT["mean_dst"]
        R_T = RT["R_T"]
        mean_src = RT["mean_src"]

        # 旋转矩阵R是R_T的转置
        R = R_T.T

        # 构建KD树
        kd_tree = KDTree(cloud[:, :2])  # 使用点云的x和y坐标

        # 初始化一个空列表来存储变换后的框
        transformed_boxes = []

        for box in boxes:
            # 提取框的坐标
            x1, y1, x2, y2,type, val = box

            # 查找最近的点
            _, idx1 = kd_tree.query([x1, y1], k=1)
            _, idx2 = kd_tree.query([x2, y2], k=1)

            # 获取z坐标
            z1 = cloud[idx1, 2]
            z2 = cloud[idx2, 2]

            print(z1,z2)

            # 将xy坐标扩展到3D空间
            top_left = np.array([x1, y1, z1])
            bottom_right = np.array([x2, y2, z2])
            #self.draw_pcd(cloud, [top_left, bottom_right]) #实际过程中无需绘制
            
            # 对框的两个点进行变换
            transformed_top_left =(np.dot((top_left - mean_src), R) + mean_dst)
            transformed_bottom_right = (np.dot((bottom_right - mean_src), R) + mean_dst)
            start_time = time.time()
            a,b=self.calculate_box_from_points(transformed_top_left,transformed_bottom_right, self.p)
            print(a,b)
            all_time = time.time() - start_time
            print(f'用时{all_time}s')


            #self.draw_pcd(cloud2, [transformed_top_left, transformed_bottom_right]) #实际过程中无需绘制

            # 由于我们不处理z坐标，只取x和y
            transformed_box = [
                a[1],a[0],  
                b[1], b[0], 
                type, val
            ]
            # 添加变换后的框到列表中
            transformed_boxes.append(transformed_box)

        return transformed_boxes

    def merge_adjacent_boxes(self, boxes):
        def boxes_intersect(box1, box2):
            return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

        def merge_boxes(box1, box2):
            x1 = min(box1[0], box2[0])
            y1 = min(box1[1], box2[1])
            x2 = max(box1[2], box2[2])
            y2 = max(box1[3], box2[3])
            return [x1, y1, x2, y2, 0, 0]  

        def is_box_valid(box):
            # Ensure x2 >= x1 and y2 >= y1
            return box[2] >= box[0] and box[3] >= box[1]
        
        def is_box_at_border(box):
            # Ensure box is not at border of image
            return box[0]>5 and box[1]>5 and box[2]<self.width-5 and box[3]<self.height-5

        def is_box_too_large(box):
            scale = 0.4
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            return box_width >= self.width * scale or box_height >= self.height * scale

        changed = True
        while changed:
            changed = False
            merged_boxes = []
            while boxes:
                box = boxes.pop(0)
                if not is_box_valid(box):
                    continue  # Skip invalid boxes
                for i, other in enumerate(boxes):
                    if boxes_intersect(box, other):
                        box = merge_boxes(box, other)
                        boxes.pop(i)
                        changed = True
                        break
                if not is_box_too_large(box):
                    merged_boxes.append(box)
            boxes = [box for box in merged_boxes if is_box_valid(box) and not is_box_too_large(box) and is_box_at_border(box)]
        return boxes



    def create_pit_in_cloud(self, test_cloud, pit_center, pit_radius, max_depth):
        # 计算每个点到凹坑中心的距离
        distances = np.sqrt((test_cloud[:, 0] - pit_center[0])**2 + (test_cloud[:, 1] - pit_center[1])**2)
        
        # 确定哪些点位于凹坑区域内
        in_pit = distances < pit_radius
        
        # 使用二次函数调整深度减少的比例，以创建更自然的弧度效果
        # 比例因子随着距离的增加而减少，距离中心越近，深度减少越多
        depth_factor = (1 - (distances[in_pit] / pit_radius)**2)
        depth_decrease = max_depth * depth_factor
        
        # 减少这些点的z值来创建带有弧度的凹坑效果
        test_cloud[in_pit, 2] -= depth_decrease
        
        return test_cloud


    
    def deform_detect_3D2(self, norm_file,  abnorm_file,  Sim_3D_dis_thresh, Sampling_interval=4):
        self.Sampling_interval = Sampling_interval
        xyz_path1 = os.path.splitext(norm_file)[0] + '.xyz'
        standard_cloud = self.readxyz(xyz_path1, [0, 0, 1920-0, 1200-0, 0, 0], 0.1, self.Sampling_interval)
        xyz_path2 = os.path.splitext(abnorm_file)[0] + '.xyz'
        test_cloud = self.readxyz(xyz_path2, [0, 0, 1920-0, 1200-0, 0, 0], 0.1, self.Sampling_interval)
        print(test_cloud)
        standard_cloud, aligned_test_cloud,  mean_error ,RT= self.align_point_clouds(standard_cloud,test_cloud)
        self.draw_pcd(standard_cloud, aligned_test_cloud) #实际过程中无需绘制
        image_width = 1920
        image_height = 1200
        box_width = 24*6
        box_height = 15*6
        abnorm_box=[]
        for y in range(0, image_height, box_height):
            for x in range(0, image_width, box_width):
                 box = [x, y, x + box_width, y + box_height, 0, 0]
                 print(box)
                 # 根据box切割standard_cloud, aligned_test_cloud
                 cloud1 = self.crop_point_cloud(standard_cloud, box)
                 print(cloud1)
                 cloud2 = self.crop_point_cloud(aligned_test_cloud, box)
                 print(cloud2)
                 if len(cloud1)<8 or len(cloud2)<8:
                     continue
                 # 获得cloud1, cloud2和mean_error
                 _ , mean_error = self.mean_nearest_distance(cloud1, cloud2)
                 print('-----------',mean_error)
                 self.draw_pcd(cloud1, cloud2) #实际过程中无需绘制
                 box[4]='abnorm'
                 box[5]=mean_error
                 if mean_error < Sim_3D_dis_thresh:
                    print('点云平均距离相差较小，可认为大概率是正常的部件')
                 else:
                    print('点云平均距离相差较大，可认为大概率有缺陷的部件')
                    box[0],box[1],box[2],box[3]=self.calculate_box_from_points(cloud2)
                    abnorm_box.append(box)        
        print(abnorm_box)      
        abnorm_box = self.merge_adjacent_boxes(abnorm_box)
        print(abnorm_box) 
        abnorm_box = self.RT_adjacent_boxes(abnorm_box,RT)
        print(abnorm_box) 
        return abnorm_box

if __name__ == '__main__':

    pass 

