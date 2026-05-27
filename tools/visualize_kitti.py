#!/usr/bin/env python3
"""
增强版KITTI 3D可视化工具 - 支持基线方法对比
核心特性：
1. 支持基线方法和检测方法的对比可视化
2. 分别加载不同的配置文件和预训练模型
3. 顺序执行以避免内存溢出
4. 生成多种组合视图
5. 优化显示范围和背景设置
6. 新增纯净点云和真值框内点云可视化
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
import matplotlib
import logging
import pickle
import time
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加项目根目录
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu


def setup_plot_style():
    """设置统一的绘图样式 - 使用透明背景"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': False,
        'figure.facecolor': 'none',  # 透明背景
        'axes.facecolor': 'none',  # 透明背景
        'savefig.facecolor': 'none',  # 透明背景
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.labelcolor': 'black',
        'text.color': 'black'
    })


def boxes_to_corners_3d(boxes_3d):
    """
    将3D边界框转换为8个角点
    Args:
        boxes_3d: [N, 7] 格式的3D框，每行包含 [x, y, z, l, w, h, yaw]
    Returns:
        corners_3d: [N, 8, 3] 角点坐标
    """
    if len(boxes_3d) == 0:
        return np.zeros((0, 8, 3))

    corners_3d = []
    for box in boxes_3d:
        if len(box) > 7:
            box = box[:7]

        x, y, z, l, w, h, yaw = box
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # 激光雷达坐标系：X向前，Y向左，Z向上
        rot_mat = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # 半尺寸
        half_l, half_w, half_h = l / 2, w / 2, h / 2

        # 8个角点的局部坐标
        corners_local = np.array([
            [half_l, half_w, half_h],  # 前右上
            [half_l, half_w, -half_h],  # 前右下
            [half_l, -half_w, half_h],  # 前左上
            [half_l, -half_w, -half_h],  # 前左下
            [-half_l, half_w, half_h],  # 后右上
            [-half_l, half_w, -half_h],  # 后右下
            [-half_l, -half_w, half_h],  # 后左上
            [-half_l, -half_w, -half_h]  # 后左下
        ])

        # 旋转并平移到全局坐标系
        corners_global = np.dot(corners_local, rot_mat.T) + np.array([x, y, z])
        corners_3d.append(corners_global)

    return np.array(corners_3d)


def get_bottom_indices():
    """返回底面矩形的角点索引（形成闭合多边形）"""
    return [3, 7, 5, 1, 3]  # 前左下 -> 后左下 -> 后右下 -> 前右下 -> 前左下


def calculate_fixed_display_range(points, range_size=60.0):
    """
    计算固定的显示范围，X轴0-60米，Y轴对称缩放，保持正方形视图
    Args:
        points: 点云数据 [N, 3]
        range_size: X轴显示范围大小（米），默认60米
    Returns:
        x_lim, y_lim: X轴和Y轴的显示范围
    """
    # 固定X轴范围 [0, 60]
    x_min, x_max = 0.0, range_size

    # 计算Y轴范围，以保持正方形视图
    y_center = 0.0  # 假设场景中心在Y轴0点
    y_half_range = range_size / 2  # 使Y轴范围与X轴范围的一半相同，以保持正方形视图
    y_min, y_max = y_center - y_half_range, y_center + y_half_range

    return (x_min, x_max), (y_min, y_max)


def extract_points_in_boxes(points, boxes_3d):
    """
    提取真值框内的点云
    Args:
        points: 原始点云 [N, 3]
        boxes_3d: 3D边界框 [M, 7]
    Returns:
        inside_points: 框内的点云 [K, 3]
    """
    if len(points) == 0 or len(boxes_3d) == 0:
        return np.array([]).reshape(0, 3)

    inside_points = []
    points = np.asarray(points)

    for box in boxes_3d:
        if len(box) < 7:
            continue

        x, y, z, l, w, h, yaw = box[:7]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # 将点转换到框的局部坐标系
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        dz = points[:, 2] - z

        # 旋转到框的坐标系
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        local_z = dz

        # 检查点是否在框内
        in_x = np.abs(local_x) <= l / 2
        in_y = np.abs(local_y) <= w / 2
        in_z = np.abs(local_z) <= h / 2

        inside_mask = in_x & in_y & in_z
        inside_indices = np.where(inside_mask)[0]

        if len(inside_indices) > 0:
            inside_points.append(points[inside_indices])

    if len(inside_points) > 0:
        return np.vstack(inside_points)
    else:
        return np.array([]).reshape(0, 3)


def visualize_pure_points_bev(points, output_path, range_size=60.0):
    """
    可视化纯净点云（不包含任何框）
    Args:
        points: 点云数据 [N, 3]
        output_path: 输出文件路径
        range_size: 显示范围大小
    """
    try:
        # 数据验证和预处理
        points = np.asarray(points) if points is not None else np.array([]).reshape(0, 3)
        if points.ndim != 2 or points.shape[1] < 3:
            points = np.array([]).reshape(0, 3)

        setup_plot_style()

        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('none')  # 透明背景
        ax.set_facecolor('none')  # 透明背景

        # 计算固定显示范围
        x_lim, y_lim = calculate_fixed_display_range(points, range_size)

        # 绘制点云（使用高度信息着色）
        if len(points) > 0:
            # 使用更适合透明背景的颜色映射
            scatter = ax.scatter(points[:, 1], points[:, 0], c=points[:, 2],
                                 cmap='viridis', s=0.5, alpha=0.7, vmin=-2, vmax=2)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Height (Z) [m]', color='black', fontsize=10)
            cbar.ax.tick_params(colors='black')

        # 设置坐标轴
        ax.set_xlabel('Y (Left/Right) [m]', fontsize=12, color='black')
        ax.set_ylabel('X (Forward) [m]', fontsize=12, color='black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_aspect('equal')

        # 设置显示范围
        ax.set_xlim(y_lim)
        ax.set_ylim(x_lim)

        # 添加原点标记
        ax.plot(0, 0, 'ko', markersize=8, markeredgewidth=2, alpha=0.8, label='Origin')

        # 添加图例（透明背景格式）
        ax.legend(loc='upper right', facecolor='white', edgecolor='black',
                  labelcolor='black', fontsize=10, framealpha=0.8)

        plt.tight_layout()
        # 保存为透明背景
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='none', edgecolor='none', transparent=True)
        plt.close()

        print(f"纯净点云BEV可视化已保存: {output_path}")
        return True

    except Exception as e:
        print(f"纯净点云BEV可视化失败: {e}")
        return False


def visualize_gt_inside_points_bev(points, gt_boxes, output_path, range_size=60.0):
    """
    可视化真值框内点云（细线框+框内点云）
    Args:
        points: 原始点云数据 [N, 3]
        gt_boxes: 真值框 [M, 7]
        output_path: 输出文件路径
        range_size: 显示范围大小
    """
    try:
        # 数据验证和预处理
        points = np.asarray(points) if points is not None else np.array([]).reshape(0, 3)
        if points.ndim != 2 or points.shape[1] < 3:
            points = np.array([]).reshape(0, 3)

        gt_boxes = np.asarray(gt_boxes) if gt_boxes is not None else np.array([]).reshape(0, 7)
        if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 7:
            gt_boxes = np.array([]).reshape(0, 7)

        # 提取真值框内的点云
        inside_points = extract_points_in_boxes(points, gt_boxes)

        setup_plot_style()

        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('none')  # 透明背景
        ax.set_facecolor('none')  # 透明背景

        # 计算固定显示范围
        x_lim, y_lim = calculate_fixed_display_range(points, range_size)

        # 绘制框内点云（使用高度信息着色）
        if len(inside_points) > 0:
            scatter = ax.scatter(inside_points[:, 1], inside_points[:, 0], c=inside_points[:, 2],
                                 cmap='viridis', s=1.0, alpha=0.8, vmin=-2, vmax=2)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Height (Z) [m]', color='black', fontsize=10)
            cbar.ax.tick_params(colors='black')

        # 绘制真值框（细线）
        if len(gt_boxes) > 0:
            corners_3d = boxes_to_corners_3d(gt_boxes)
            bottom_indices = get_bottom_indices()

            for i, corners in enumerate(corners_3d):
                if i == 0:
                    # 第一个框添加标签
                    ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                            color='green', linewidth=1.0, alpha=0.9, label='Ground Truth')
                else:
                    # 后续框不添加标签
                    ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                            color='green', linewidth=1.0, alpha=0.9)

        # 设置坐标轴
        ax.set_xlabel('Y (Left/Right) [m]', fontsize=12, color='black')
        ax.set_ylabel('X (Forward) [m]', fontsize=12, color='black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_aspect('equal')

        # 设置显示范围
        ax.set_xlim(y_lim)
        ax.set_ylim(x_lim)

        # 添加原点标记
        ax.plot(0, 0, 'ko', markersize=8, markeredgewidth=2, alpha=0.8, label='Origin')

        # 添加图例（透明背景格式）
        if len(gt_boxes) > 0:
            ax.legend(loc='upper right', facecolor='white', edgecolor='black',
                      labelcolor='black', fontsize=10, framealpha=0.8)

        plt.tight_layout()
        # 保存为透明背景
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='none', edgecolor='none', transparent=True)
        plt.close()

        print(f"真值框内点云BEV可视化已保存: {output_path}")
        print(f"真值框内点云数量: {len(inside_points)}")
        return True

    except Exception as e:
        print(f"真值框内点云BEV可视化失败: {e}")
        return False


def visualize_3d_bev_standard(points, boxes_3d, output_path, box_color='green',
                              range_size=60.0, box_label=None):
    """
    标准化的3D BEV可视化，使用透明背景
    Args:
        points: 点云数据 [N, 3]
        boxes_3d: 3D边界框 [M, 7]
        output_path: 输出文件路径
        box_color: 框的颜色 ('green' for GT, 'blue' for detection, 'orange' for baseline)
        range_size: 显示范围大小
        box_label: 框的标签说明
    """
    try:
        # 数据验证和预处理
        points = np.asarray(points) if points is not None else np.array([]).reshape(0, 3)
        if points.ndim != 2 or points.shape[1] < 3:
            points = np.array([]).reshape(0, 3)

        boxes_3d = np.asarray(boxes_3d) if boxes_3d is not None else np.array([]).reshape(0, 7)
        if boxes_3d.ndim != 2 or boxes_3d.shape[1] < 7:
            boxes_3d = np.array([]).reshape(0, 7)

        setup_plot_style()

        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('none')  # 透明背景
        ax.set_facecolor('none')  # 透明背景

        # 计算固定显示范围
        x_lim, y_lim = calculate_fixed_display_range(points, range_size)

        # 绘制点云（使用高度信息着色）
        if len(points) > 0:
            # 注意坐标轴映射：X向前（纵轴），Y向左（横轴）
            # 使用更适合透明背景的颜色映射
            scatter = ax.scatter(points[:, 1], points[:, 0], c=points[:, 2],
                                 cmap='viridis', s=0.5, alpha=0.7, vmin=-2, vmax=2)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Height (Z) [m]', color='black', fontsize=10)
            cbar.ax.tick_params(colors='black')

        # 绘制3D边界框
        if len(boxes_3d) > 0:
            corners_3d = boxes_to_corners_3d(boxes_3d)
            bottom_indices = get_bottom_indices()

            # 为第一个框添加标签，避免重复
            for i, corners in enumerate(corners_3d):
                if i == 0 and box_label:
                    # 第一个框添加标签
                    ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                            color=box_color, linewidth=2.0, alpha=0.9, label=box_label)
                else:
                    # 后续框不添加标签
                    ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                            color=box_color, linewidth=2.0, alpha=0.9)

        # 设置坐标轴
        ax.set_xlabel('Y (Left/Right) [m]', fontsize=12, color='black')
        ax.set_ylabel('X (Forward) [m]', fontsize=12, color='black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_aspect('equal')

        # 设置显示范围
        ax.set_xlim(y_lim)
        ax.set_ylim(x_lim)

        # 添加原点标记
        ax.plot(0, 0, 'ko', markersize=8, markeredgewidth=2, alpha=0.8, label='Origin')

        # 添加图例（透明背景格式）
        if box_label:
            ax.legend(loc='upper right', facecolor='white', edgecolor='black',
                      labelcolor='black', fontsize=10, framealpha=0.8)

        plt.tight_layout()
        # 保存为透明背景
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='none', edgecolor='none', transparent=True)
        plt.close()

        print(f"3D BEV可视化已保存: {output_path}")
        return True

    except Exception as e:
        print(f"3D BEV可视化失败: {e}")
        return False


def visualize_gt_bev(dataset, sample_idx, output_dir, range_size=60.0):
    """
    可视化真值数据的3D BEV视图
    """
    try:
        # 从数据集中获取数据
        data_dict = dataset[sample_idx]

        # 提取点云数据
        points = data_dict['points'][:, :3] if 'points' in data_dict else np.array([]).reshape(0, 3)

        # 提取真值框
        gt_boxes = []
        if 'gt_boxes' in data_dict and len(data_dict['gt_boxes']) > 0:
            for box in data_dict['gt_boxes']:
                if len(box) >= 7:  # 确保有完整的7个参数
                    gt_boxes.append(box[:7])

        # 创建输出路径
        output_path = output_dir / f"{sample_idx:06d}_gt_bev.png"

        # 可视化
        success = visualize_3d_bev_standard(
            points, np.array(gt_boxes), output_path,
            box_color='green', range_size=range_size,
            box_label='Ground Truth'
        )

        if success:
            print(f"样本 {sample_idx:06d} 真值框数量: {len(gt_boxes)}")

        return success

    except Exception as e:
        print(f"真值BEV可视化失败: {e}")
        return False


def visualize_pure_points(dataset, sample_idx, output_dir, range_size=60.0):
    """
    可视化纯净点云（不包含任何框）
    """
    try:
        # 从数据集中获取数据
        data_dict = dataset[sample_idx]

        # 提取点云数据
        points = data_dict['points'][:, :3] if 'points' in data_dict else np.array([]).reshape(0, 3)

        # 创建输出路径
        output_path = output_dir / f"{sample_idx:06d}_pure_points_bev.png"

        # 可视化
        success = visualize_pure_points_bev(points, output_path, range_size)

        if success:
            print(f"样本 {sample_idx:06d} 纯净点云可视化完成")

        return success

    except Exception as e:
        print(f"纯净点云可视化失败: {e}")
        return False


def visualize_gt_inside_points(dataset, sample_idx, output_dir, range_size=60.0):
    """
    可视化真值框内点云（细线框+框内点云）
    """
    try:
        # 从数据集中获取数据
        data_dict = dataset[sample_idx]

        # 提取点云数据
        points = data_dict['points'][:, :3] if 'points' in data_dict else np.array([]).reshape(0, 3)

        # 提取真值框
        gt_boxes = []
        if 'gt_boxes' in data_dict and len(data_dict['gt_boxes']) > 0:
            for box in data_dict['gt_boxes']:
                if len(box) >= 7:  # 确保有完整的7个参数
                    gt_boxes.append(box[:7])

        # 创建输出路径
        output_path = output_dir / f"{sample_idx:06d}_gt_inside_points_bev.png"

        # 可视化
        success = visualize_gt_inside_points_bev(
            points, np.array(gt_boxes), output_path, range_size
        )

        if success:
            print(f"样本 {sample_idx:06d} 真值框内点云可视化完成")

        return success

    except Exception as e:
        print(f"真值框内点云可视化失败: {e}")
        return False


def get_detection_results(model, dataloader, sample_idx, device, score_threshold=0.3):
    """
    获取检测结果并缓存，确保一致性
    Returns:
        det_boxes: 检测框列表
    """
    try:
        dataset = dataloader.dataset
        data_dict = dataset[sample_idx]

        # 创建批次数据
        batch_dict = dataset.collate_batch([data_dict])

        # 数据转移到GPU
        load_data_to_gpu(batch_dict)

        # 模型推理
        model.eval()
        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)

        # 提取检测框
        det_boxes = []
        if pred_dicts and len(pred_dicts) > 0:
            for pred_dict in pred_dicts:
                if 'pred_boxes' in pred_dict and 'pred_scores' in pred_dict:
                    boxes = pred_dict['pred_boxes'].cpu().numpy()
                    scores = pred_dict['pred_scores'].cpu().numpy()

                    for i, box in enumerate(boxes):
                        if scores[i] > score_threshold and len(box) >= 7:
                            det_boxes.append(box[:7])

        return det_boxes
    except Exception as e:
        print(f"获取检测结果失败: {e}")
        return []


def visualize_det_bev_enhanced(model, dataloader, sample_idx, output_dir, device,
                               range_size=60.0, score_threshold=0.3, method_name='detection',
                               cached_results=None):
    """
    增强版检测结果可视化 - 支持缓存结果确保一致性
    Args:
        cached_results: 可选的缓存结果，避免重复计算
    """
    try:
        dataset = dataloader.dataset
        data_dict = dataset[sample_idx]

        # 提取点云数据
        points = data_dict['points'][:, :3] if 'points' in data_dict else np.array([]).reshape(0, 3)

        # 使用缓存结果或重新计算
        if cached_results is not None and sample_idx in cached_results:
            det_boxes = cached_results[sample_idx]
            print(f"使用缓存结果: 样本 {sample_idx:06d}")
        else:
            det_boxes = get_detection_results(model, dataloader, sample_idx, device, score_threshold)
            # 缓存结果
            if cached_results is not None:
                cached_results[sample_idx] = det_boxes

        # 根据方法名称设置颜色和标签
        if method_name == 'baseline':
            box_color = 'orange'
            box_label = 'Baseline'
            file_suffix = 'baseline'
        else:  # detection
            box_color = 'blue'
            box_label = 'Detection'
            file_suffix = 'det'

        # 创建输出路径
        output_path = output_dir / f"{sample_idx:06d}_{file_suffix}_bev.png"

        # 可视化
        success = visualize_3d_bev_standard(
            points, np.array(det_boxes), output_path,
            box_color=box_color, range_size=range_size,
            box_label=box_label
        )

        if success:
            print(f"样本 {sample_idx:06d} {box_label}框数量: {len(det_boxes)}")

        return success, det_boxes  # 返回检测结果用于缓存

    except Exception as e:
        print(f"{method_name.capitalize()} BEV可视化失败: {e}")
        return False, []


def visualize_comparison_bev(det_boxes, baseline_boxes, dataset, sample_idx, output_dir,
                             range_size=60.0, comparison_type='gt_det'):
    """
    对比可视化 - 使用缓存结果确保一致性
    Args:
        det_boxes: 检测方法的检测框（已缓存）
        baseline_boxes: 基线方法的检测框（已缓存）
        comparison_type: 对比类型
            'gt_det' - 真值与检测方法对比
            'baseline_det' - 基线方法与检测方法对比
    """
    try:
        data_dict = dataset[sample_idx]

        # 提取点云数据
        points = data_dict['points'][:, :3] if 'points' in data_dict else np.array([]).reshape(0, 3)

        # 提取真值框
        gt_boxes = []
        if 'gt_boxes' in data_dict and len(data_dict['gt_boxes']) > 0:
            for box in data_dict['gt_boxes']:
                if len(box) >= 7:
                    gt_boxes.append(box[:7])

        # 根据对比类型创建输出路径
        if comparison_type == 'gt_det':
            output_path = output_dir / f"{sample_idx:06d}_gt_det_comparison_bev.png"
        else:  # baseline_det
            output_path = output_dir / f"{sample_idx:06d}_baseline_det_comparison_bev.png"

        # 设置绘图样式
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        # 计算固定显示范围
        x_lim, y_lim = calculate_fixed_display_range(points, range_size)

        # 绘制点云
        if len(points) > 0:
            scatter = ax.scatter(points[:, 1], points[:, 0], c=points[:, 2],
                                 cmap='viridis', s=0.5, alpha=0.7, vmin=-2, vmax=2)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Height (Z) [m]', color='black', fontsize=10)
            cbar.ax.tick_params(colors='black')

        # 根据对比类型绘制不同的框
        if comparison_type == 'gt_det':
            # 绘制真值框（绿色）
            if len(gt_boxes) > 0:
                gt_corners = boxes_to_corners_3d(gt_boxes)
                bottom_indices = get_bottom_indices()
                for i, corners in enumerate(gt_corners):
                    if i == 0:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='green', linewidth=2.5, alpha=0.9, label='Ground Truth')
                    else:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='green', linewidth=2.5, alpha=0.9)

            # 绘制检测框（蓝色）
            if len(det_boxes) > 0:
                det_corners = boxes_to_corners_3d(det_boxes)
                bottom_indices = get_bottom_indices()
                for i, corners in enumerate(det_corners):
                    if i == 0:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='blue', linewidth=2.0, alpha=0.9, label='Detection')
                    else:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='blue', linewidth=2.0, alpha=0.9)

        else:  # baseline_det
            # 绘制基线方法框（橙色）
            if len(baseline_boxes) > 0:
                baseline_corners = boxes_to_corners_3d(baseline_boxes)
                bottom_indices = get_bottom_indices()
                for i, corners in enumerate(baseline_corners):
                    if i == 0:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='orange', linewidth=2.5, alpha=0.9, label='Baseline')
                    else:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='orange', linewidth=2.5, alpha=0.9)

            # 绘制检测框（蓝色）
            if len(det_boxes) > 0:
                det_corners = boxes_to_corners_3d(det_boxes)
                bottom_indices = get_bottom_indices()
                for i, corners in enumerate(det_corners):
                    if i == 0:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='blue', linewidth=2.0, alpha=0.9, label='Detection')
                    else:
                        ax.plot(corners[bottom_indices, 1], corners[bottom_indices, 0],
                                color='blue', linewidth=2.0, alpha=0.9)

        # 设置坐标轴
        ax.set_xlabel('Y (Left/Right) [m]', fontsize=12, color='black')
        ax.set_ylabel('X (Forward) [m]', fontsize=12, color='black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_aspect('equal')
        ax.set_xlim(y_lim)
        ax.set_ylim(x_lim)

        # 添加原点标记
        ax.plot(0, 0, 'ko', markersize=8, markeredgewidth=2, alpha=0.8, label='Origin')

        # 添加图例（透明背景格式）
        ax.legend(loc='upper right', facecolor='white', edgecolor='black',
                  labelcolor='black', fontsize=10, framealpha=0.8)

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                    facecolor='none', edgecolor='none', transparent=True)
        plt.close()

        # 打印统计信息
        if comparison_type == 'gt_det':
            print(f"样本 {sample_idx:06d} 真值-检测对比完成 - 真值框: {len(gt_boxes)}, 检测框: {len(det_boxes)}")
        else:
            print(f"样本 {sample_idx:06d} 基线-检测对比完成 - 基线框: {len(baseline_boxes)}, 检测框: {len(det_boxes)}")
        return True

    except Exception as e:
        print(f"对比BEV可视化失败: {e}")
        return False


def load_model_enhanced(cfg, ckpt_path, dataset, device, logger=None):
    """增强版模型加载 - 与评估工具保持一致"""
    if logger is None:
        logger = common_utils.create_logger()

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset
    )

    model.load_params_from_file(
        filename=ckpt_path,
        logger=logger,
        to_cpu=device.type == 'cpu'
    )

    model.eval()
    model = model.to(device)

    return model


def parse_sample_ids(sample_ids_str, dataset_size):
    """解析样本ID字符串"""
    if sample_ids_str.isdigit():
        num_samples = int(sample_ids_str)
        return list(range(min(num_samples, dataset_size)))
    elif ':' in sample_ids_str:
        parts = sample_ids_str.split(':')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, min(end, dataset_size)))
    else:
        try:
            sample_ids = [int(idx.strip()) for idx in sample_ids_str.split(',')]
            return [idx for idx in sample_ids if idx < dataset_size]
        except ValueError:
            pass

    print(f"无法解析样本ID格式: {sample_ids_str}，使用默认前5个样本")
    return list(range(min(5, dataset_size)))


def main():
    """主函数 - 增强版，支持基线方法对比"""
    parser = argparse.ArgumentParser(description='增强版KITTI 3D可视化工具 - 支持基线方法对比')

    # 必需参数
    parser.add_argument('--data_path', type=str,
                        default='../data/kitti',
                        help='数据路径')

    # 检测方法参数
    parser.add_argument('--det_cfg_file', type=str,
                        default='../output/visualization/detection/20251217-1000/FEDM_second_with_lion_mamba_64dim.yaml',
                        help='检测方法配置文件路径')
    parser.add_argument('--det_model_ckpt', type=str,
                        default='../output/visualization/detection/20251217-1000/checkpoint_epoch_80.pth',
                        help='检测方法模型权重路径')

    # 基线方法参数
    parser.add_argument('--baseline_cfg_file', type=str,
                        default='../output/visualization/baseline/20251222-1100/second_with_lion_mamba_64dim.yaml',
                        help='基线方法配置文件路径')
    parser.add_argument('--baseline_model_ckpt', type=str,
                        default='../output/visualization/baseline/20251222-1100/checkpoint_epoch_80.pth',
                        help='基线方法模型权重路径')

    # 可选参数
    parser.add_argument('--output_dir', type=str,
                        default='../output/visualization',
                        help='输出目录')

    parser.add_argument('--mode', choices=['all', 'single', 'comparison', 'dataset_only'], default='all',
                        help='可视化模式: all=全部视图, single=单独视图, comparison=仅对比视图, dataset_only=仅数据集相关视图')
    parser.add_argument('--sample_ids', type=str, default='5',
                        help='样本ID列表，支持格式: 1) "0,1,2" 2) "0:5" 3) "10" (前10个样本)')
    parser.add_argument('--range_size', type=float, default=60.0,  # 修改默认范围为60米
                        help='可视化范围大小（米），默认60米')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='检测框置信度阈值，默认0.3')
    parser.add_argument('--extra_tag', type=str, default='enhanced_bev_visualization', help='输出目录标签')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小（用于数据加载器）')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "output" / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print(f"可视化范围: {args.range_size} 米")
    print(f"检测置信度阈值: {args.score_threshold}")
    print(f"可视化模式: {args.mode}")

    # 创建日志器
    logger = common_utils.create_logger()

    # 首先加载检测方法配置（用于数据加载器）
    cfg_from_yaml_file(args.det_cfg_file, cfg)
    cfg.DATA_PATH = str(args.data_path)
    cfg.DATA_CONFIG.DATA_PATH = str(args.data_path)

    # 构建数据加载器
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        logger=logger,
        training=False,
        total_epochs=1
    )

    dataset_size = len(dataset)
    print(f"数据集大小: {dataset_size} 个样本")

    # 解析样本ID
    sample_ids = parse_sample_ids(args.sample_ids, dataset_size)
    print(f"将要可视化的样本: {sample_ids}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 顺序加载模型以避免内存溢出（仅在需要时加载）
    baseline_model = None
    detection_model = None

    # 仅当需要模型推理时才加载模型
    if args.mode in ['all', 'single', 'comparison']:
        # 加载基线模型
        print("加载基线模型...")
        baseline_cfg = cfg.__class__()
        cfg_from_yaml_file(args.baseline_cfg_file, baseline_cfg)
        baseline_cfg.DATA_PATH = str(args.data_path)
        baseline_cfg.DATA_CONFIG.DATA_PATH = str(args.data_path)

        baseline_model = load_model_enhanced(baseline_cfg, args.baseline_model_ckpt, dataset, device, logger)
        print("基线模型加载完成")

        # 加载检测模型
        print("加载检测模型...")
        detection_cfg = cfg.__class__()
        cfg_from_yaml_file(args.det_cfg_file, detection_cfg)
        detection_cfg.DATA_PATH = str(args.data_path)
        detection_cfg.DATA_CONFIG.DATA_PATH = str(args.data_path)

        detection_model = load_model_enhanced(detection_cfg, args.det_model_ckpt, dataset, device, logger)
        print("检测模型加载完成")

    # 创建缓存字典，确保检测结果一致性
    baseline_results_cache = {}
    detection_results_cache = {}

    # 执行可视化
    success_count = 0
    for i, sample_idx in enumerate(sample_ids):
        print(f"\n处理样本 {sample_idx:06d}...")

        success = True

        try:
            # 数据集相关视图（总是执行，除非明确指定不执行）
            if args.mode in ['all', 'single', 'dataset_only']:
                # 纯净点云可视化
                if not visualize_pure_points(dataset, sample_idx, output_dir, args.range_size):
                    success = False
                    print(f"样本 {sample_idx:06d} 纯净点云可视化失败")

                # 真值框内点云可视化
                if not visualize_gt_inside_points(dataset, sample_idx, output_dir, args.range_size):
                    success = False
                    print(f"样本 {sample_idx:06d} 真值框内点云可视化失败")

                # 真值可视化
                if not visualize_gt_bev(dataset, sample_idx, output_dir, args.range_size):
                    success = False
                    print(f"样本 {sample_idx:06d} 真值可视化失败")

            # 单独视图（需要模型推理）
            if args.mode in ['all', 'single']:
                # 基线方法可视化（使用缓存）
                baseline_success, baseline_boxes = visualize_det_bev_enhanced(
                    baseline_model, dataloader, sample_idx, output_dir,
                    device, args.range_size, args.score_threshold, 'baseline',
                    baseline_results_cache
                )
                if not baseline_success:
                    success = False
                    print(f"样本 {sample_idx:06d} 基线方法可视化失败")

                # 检测方法可视化（使用缓存）
                detection_success, detection_boxes = visualize_det_bev_enhanced(
                    detection_model, dataloader, sample_idx, output_dir,
                    device, args.range_size, args.score_threshold, 'detection',
                    detection_results_cache
                )
                if not detection_success:
                    success = False
                    print(f"样本 {sample_idx:06d} 检测方法可视化失败")

            # 对比视图（使用缓存结果确保一致性）
            if args.mode in ['all', 'comparison']:
                # 确保检测结果已缓存
                if sample_idx not in detection_results_cache:
                    detection_boxes = get_detection_results(
                        detection_model, dataloader, sample_idx, device, args.score_threshold
                    )
                    detection_results_cache[sample_idx] = detection_boxes
                else:
                    detection_boxes = detection_results_cache[sample_idx]

                # 真值与检测方法对比
                if not visualize_comparison_bev(
                        detection_boxes, None, dataset, sample_idx, output_dir,
                        args.range_size, 'gt_det'
                ):
                    success = False
                    print(f"样本 {sample_idx:06d} 真值-检测对比可视化失败")

                # 确保基线结果已缓存
                if sample_idx not in baseline_results_cache:
                    baseline_boxes = get_detection_results(
                        baseline_model, dataloader, sample_idx, device, args.score_threshold
                    )
                    baseline_results_cache[sample_idx] = baseline_boxes
                else:
                    baseline_boxes = baseline_results_cache[sample_idx]

                # 基线方法与检测方法对比
                if not visualize_comparison_bev(
                        detection_boxes, baseline_boxes, dataset, sample_idx, output_dir,
                        args.range_size, 'baseline_det'
                ):
                    success = False
                    print(f"样本 {sample_idx:06d} 基线-检测对比可视化失败")

        except Exception as e:
            print(f"样本 {sample_idx:06d} 处理失败: {e}")
            success = False

        if success:
            success_count += 1

        print(f"进度: {i + 1}/{len(sample_ids)}")

    print(f"\n✅ 增强版3D可视化任务完成!")
    print(f"成功处理: {success_count}/{len(sample_ids)} 个样本")
    print(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()