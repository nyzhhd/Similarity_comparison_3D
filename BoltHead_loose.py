import os
import random

import cv2
import time
import torch
# import openpyxl
import itertools
import numpy as np
import onnxruntime as ort
import xml.etree.cElementTree as ET
import os
import struct
import numpy as np
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

from tqdm import tqdm
from collections import Counter
from sklearn.cluster import DBSCAN
from scipy.spatial import procrustes
from torchvision import transforms as T
# from src.loftr import LoFTR, default_cfg
# from config import *
#部件裁剪文件夹配置文件


#TODO:将配准获取的结果框与标准图上的框进行匹配，查看哪些框是带有_s的，返回_s的框即为螺栓松动测量的框

#------------------定义一个onnx目标检测器的类,得到目标检测框--------------------
class OnnxDetector(object):
    # 初始化
    def __init__(self, onnx_path, classes_path):
        self.sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.sess.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])
        with open(classes_path, 'r') as f:
            names = f.readlines()
        self.class_names = np.array([name.replace('\n', '').replace('\t', '') for name in names])
        self.format_transfer = T.Compose([T.ToTensor(), T.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])])
    # 运行
    def run(self, img):
        # 获取原始图像的高度和宽度
        h_orig, w_orig = img.shape[:2]
        # 进行图像缩放
        scale = 1280
        size_divisor = 64
        # 计算缩放因子，保持宽高比并限制在指定的尺寸范围内
        scale_factor = min(scale / h_orig, scale / w_orig)
        # 缩放图像
        img_resize = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor)
        h_resize, w_resize = img_resize.shape[:2]
        # 计算需要补齐的像素数，使宽高都是size_divisor的整数倍
        right_pad = int(np.ceil(w_resize / size_divisor) * size_divisor - w_resize)
        bottom_pad = int(np.ceil(h_resize / size_divisor) * size_divisor - h_resize)
        # 使用常数值填充边界
        img_padding = cv2.copyMakeBorder(img_resize, 0, bottom_pad, 0, right_pad, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))
        # 将填充后的图像转换为模型输入所需的格式
        img_tensor = self.format_transfer(img_padding).unsqueeze(0).numpy()
        # 运行模型获取预测结果
        num_dets, pred_boxes, pre_scores, pred_labels = self.sess.run(
            ["num_dets", "boxes", "scores", "labels"],
            {"images": img_tensor})
        # 处理模型输出结果
        num_dets = int(num_dets[0])
        pred_boxes = np.array(pred_boxes[0][:num_dets]) / scale_factor  # 缩放回原图大小
        pre_scores = np.array(pre_scores[0][:num_dets]).reshape(-1, 1)
        pred_labels = np.array(pred_labels[0][:num_dets])
        pred_label_names = self.class_names[pred_labels]
        # 构建目标列表
        target_list = []
        for i in range(pred_boxes.shape[0]):
            target = list(pred_boxes[i])
            target.append(pred_label_names[i])
            target.append(pre_scores[i][0])
            target_list.append(target)
        # # 过滤置信度低于阈值的目标
        # target_list = self.filter_confidence(target_list)
        return target_list

#------------------读取xyz文件--------------------
def fun_readXYZ(path: str,p = 0.1):
    height = 1200
    width = 1920
    try:
        fd = open(path, "rb")
    except:
        print('{} 点云文件不存在'.format(path))
        return -1
    file = fd.read() # 文件尾为空字串
    data = []        # 创建列表存储.xyz
    # 转换二进制数据，读取3*h*w个，每个类型为H(unsigned_short 2Bytes)
    for cell in struct.unpack('%dH'%(3*height*width), file):
        data.append(cell)
    # 关闭文件流
    fd.close()
    # 存储为np.array (uint16类型)
    depthmap = np.array(data, dtype=np.uint16).reshape((height, width, 3))

    # depthmap = np.transpose(depthmap, (0, 1, 2))
    # return depthmap[y][x][0]*p, depthmap[y][x][1]*p, depthmap[y][x][2]*p
    return depthmap

#------------------获取每个像素点的xyz坐标--------------------
def accept_xyz(depthmap,x,y,p):
    return depthmap[y][x][0]*p, depthmap[y][x][1]*p, depthmap[y][x][2]*p

#------------------定义一个onnx目标检测器的类,得到目标检测框--------------------
def Ransac3d(cloud, maxIterations, distanceTol):
    inliersResult = []
    point_data = []
    # point_data = []
    num_points = cloud.shape[0]
    a_best = 0
    b_best = 0
    c_best = 0
    d_best = 0
    while (maxIterations > 0):
        if num_points < 3:
            break
        inliers = []  # 局内点列表，存放cloud的行索引
        # 随机在点云中选取3个初始点作为局内点
        inliers.extend(random.sample(range(0, num_points), 3))
        # 3个初始点的坐标
        x1 = cloud[inliers[0], 0]
        y1 = cloud[inliers[0], 1]
        z1 = cloud[inliers[0], 2]

        x2 = cloud[inliers[1], 0]
        y2 = cloud[inliers[1], 1]
        z2 = cloud[inliers[1], 2]

        x3 = cloud[inliers[2], 0]
        y3 = cloud[inliers[2], 1]
        z3 = cloud[inliers[2], 2]

        # 求取拟合平面的系数
        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
        # sqrt_abc = (a * a + b * b + c * c) ** 0.5
        sqrt_abc = np.sqrt(a * a + b * b + c * c )
        # 求非局内点到拟合平面的距离
        for ind in range(0, num_points):
            if ind in inliers:
                # 跳过局内点
                continue

            x = cloud[ind, 0]
            y = cloud[ind, 1]
            z = cloud[ind, 2]

            # 计算点和平面的距离
            dist = np.fabs(a * x + b * y + c * z + d) / sqrt_abc

            if dist < distanceTol:
                # 小于阈值的则加入局内点
                inliers.append(ind)

            if len(inliers) > len(inliersResult):
                # 局内点的数量大于已存在的最好拟合数量，则更新最好的拟合局内点集
                inliersResult = inliers.copy()
                a_best = a
                b_best = b
                c_best = c
                d_best = d
        maxIterations -= 1
        point_data = cloud[inliersResult]
        # print("len(point_data):",len(inliersResult))
    return inliersResult, point_data, (a_best, b_best, c_best, d_best)

def get_dist_to_plane(plane_params, cloud):
    # 计算制定范围的点到指定平面的距离
    a,b,c,d = plane_params
    sqrt_abc = (a * a + b * b + c * c)**0.5

    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]

    # 计算点和平面的距离
    dist = np.fabs(a * x + b * y + c * z + d) / sqrt_abc
    return dist

import numpy as np
import random

def random_downsample(point_cloud, sample_ratio):
    # print("type(point_cloud)----",type(point_cloud))
    # print("(point_cloud)----",point_cloud)

    """
    随机下采样点云数据。

    参数:
    - point_cloud: 输入的点云数组，每行包含三个整数
    - sample_ratio: 下采样比例，范围在(0, 1)之间

    返回:
    - 下采样后的点云数组
    """
    if not 0 < sample_ratio < 1:
        raise ValueError("下采样比例应在0和1之间")

    num_points = point_cloud.shape[0]
    num_points_to_keep = int(num_points * sample_ratio)

    sampled_indices = random.sample(range(num_points), num_points_to_keep)

    sampled_point_cloud = point_cloud[sampled_indices, :]

    return sampled_point_cloud





from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
import numpy as np

from sklearn.linear_model import RANSACRegressor

if __name__ == "__main__":
    classes_path = "./data/classes/classes_380.txt"
    onnx_path = "./data/onnx_model/yolov5.onnx"
    object_detector = OnnxDetector(onnx_path, classes_path)
    file_path = "./measure_3D/dbzc20240110/"
    for test_im in os.listdir(file_path):
        if test_im.endswith(".jpg"):
            test_image = cv2.imread(file_path  + test_im)
            # print(test_image.shape)
            target_list = object_detector.run(test_image)
            box_list = [target[:4] for target in target_list if target[4] == 'BoltHead']
            print("box_list++++++",box_list)
            file_path_xyz = test_im[:-4] + ".xyz"
            xyz_path = file_path  + file_path_xyz
            if os.path.exists(xyz_path) is False:
                print(f"缺少点云文件：{xyz_path}")
            else:
                depthmap = fun_readXYZ(xyz_path)
                # print("depthmap",depthmap)
                for box_index,box in enumerate(box_list):
                    coordinates_list = []
                    box = [int(val) for val in box]
                    box_left = [box[0], box[1]]
                    box_right = [box[2], box[3]]
                    for i in range(box_left[0], box_right[0]) :
                         for j in range(box_left[1], box_right[1]):
                             bolt_x, bolt_y, bolt_z = accept_xyz(depthmap,i,j, p = 0.1)
                             coordinates_list.append([int(bolt_x), int(bolt_y), int(bolt_z)])
                    print("sorted_points_new------", len(coordinates_list))

                    coordinates_list = [po for po in coordinates_list if po[2]!=0]

                    sorted_points = coordinates_list
                    print(box_index,box)
                    print("sorted_points_old------",len(sorted_points))
                    if len(sorted_points) <= 100:
                        print(f"{test_im}三维点云真实数量太少，跳过处理！！！")
                        continue
                    # sorted_points = [point_0 for point_0 in coordinates_list if point_0[2] != 0]
                    # print("sorted_points_new------",sorted_points)
                    # 计算1%和99%的索引
                    percentile_1 = int(0.01 * len(sorted_points))
                    percentile_99 = int(0.99 * len(sorted_points))
                    # 选择5%到95%的点
                    point_cloud = sorted_points[percentile_1:percentile_99]
                    # 将数组列表转换为numpy列表
                    point_cloud = np.array(point_cloud)
                    point_cloud = random_downsample(point_cloud, 0.5)
                    print(f"数量：{len(point_cloud)}")

                    # print("selected_points:",len(selected_points))
                    # 拟合第一个平面
                    first_plane_points_ind, first_plane_params = Ransac3d(point_cloud, 150,
                                                                         0.5)
                    # print("first_plane_points_ind-------------",first_plane_points_ind)
                    # 找到所有不属于第一平面的点
                    mask = np.ones(point_cloud.shape[0], bool)
                    mask[first_plane_points_ind] = False
                    not_in_plane_points = point_cloud[mask]
                    # 非一点的在原点云的索引
                    temp_index = np.where(mask)[0]
                    second_plane_points_ind, second_plane_params = Ransac3d(not_in_plane_points,150,
                                                                                 0.5)
                    # 根据两个平面的参数计算平面夹角的余弦值
                    x0, y0, z0, _ = first_plane_params
                    sqrt_xyz = (x0 * x0 + y0 * y0 + z0 * z0) ** 0.5
                    a, b, c, _ = second_plane_params
                    sqrt_abc = (a * a + b * b + c * c) ** 0.5
                    cos = np.fabs(a * x0 + b * y0 + c * z0) / (sqrt_abc * sqrt_xyz)
                    # 计算两个平面之间的距离
                    dists = get_dist_to_plane(first_plane_params, point_cloud[temp_index[second_plane_points_ind]])
                    bolt_dist = np.mean(dists[int(0.3*dists.shape[0]):int(0.7*dists.shape[0])])
                    box_list[box_index].append(bolt_dist)


                for box in box_list:
                    distance_bolt  =round(box[4],2)
                # 绘制矩形框
                    cv2.rectangle(test_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                # 获取矩形框中心点位置
                    center_position = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

                    # 在矩形框上显示 max_dist_value
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(test_image, f"BoltHead: {distance_bolt}", center_position, font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

                # 显示图像
            # cv2.namedWindow("Distance", 0)
            # cv2.imshow('Distance', test_im)  # 调整窗口大小
            # output_path_filename =test_im
                output_folder = "./measure_result"
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, test_im)
                cv2.imwrite(output_path, test_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
