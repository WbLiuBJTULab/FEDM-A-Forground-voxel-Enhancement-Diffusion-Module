import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2
from ....utils import box_utils


class DynamicVoxelVFE_ForGtMap(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        # 基本配置
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        # PFN层配置
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # 体素化参数
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # 预计算缩放因子
        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        # 转换为tensor并注册为buffer
        self.register_buffer('grid_size', torch.tensor(grid_size))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range))

        # 预计算网格坐标
        self._precompute_grid_coords(grid_size)

        # === 新增：类别筛选配置 ===
        # 从配置中获取类别名称列表，默认为空（使用所有类别）
        self.target_class_names = getattr(self.model_cfg, 'TARGET_CLASS_NAMES', [])
        if self.target_class_names:
            print(f"[VFE] 将只为以下类别生成前景地图: {self.target_class_names}")

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def _precompute_grid_coords(self, grid_size):
        """优化版的网格坐标预计算"""
        device = self.grid_size.device

        z_coords = torch.arange(grid_size[0], device=device)
        y_coords = torch.arange(grid_size[1], device=device)
        x_coords = torch.arange(grid_size[2], device=device)

        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        all_voxel_coords = torch.stack([zz.flatten(), yy.flatten(), xx.flatten()], dim=1)

        voxel_centers = (all_voxel_coords.float() + 0.5) * self.voxel_size + self.point_cloud_range[:3]

        self.register_buffer('_all_voxel_coords', all_voxel_coords)
        self.register_buffer('_voxel_centers_precomputed', voxel_centers)

    def _points_in_boxes_3d_gpu(self, points, boxes):
        """进一步优化的点框判断函数"""
        if boxes.numel() == 0:
            return torch.zeros(points.size(0), 0, dtype=torch.bool, device=points.device)

        points_expanded = points.unsqueeze(1)
        boxes_centers = boxes[:, :3].unsqueeze(0)

        rel_coords = points_expanded - boxes_centers

        angles = boxes[:, 6]
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)

        x_rot = rel_coords[..., 0] * cos_a.view(1, -1) + rel_coords[..., 1] * sin_a.view(1, -1)
        y_rot = -rel_coords[..., 0] * sin_a.view(1, -1) + rel_coords[..., 1] * cos_a.view(1, -1)
        z_rot = rel_coords[..., 2]

        half_sizes = boxes[:, 3:6] / 2.0
        in_box_mask = (x_rot.abs() <= half_sizes[:, 0].view(1, -1)) & \
                      (y_rot.abs() <= half_sizes[:, 1].view(1, -1)) & \
                      (z_rot.abs() <= half_sizes[:, 2].view(1, -1))

        return in_box_mask

    def _filter_boxes_by_class(self, gt_boxes, class_names, target_class_names):
        """
        根据目标类别名称筛选真值框

        参数:
            gt_boxes: 真值框张量 [N, 8+] (最后一个是类别标签)
            class_names: 数据集中所有类别的名称列表
            target_class_names: 要保留的目标类别名称列表

        返回:
            筛选后的真值框
        """
        if not target_class_names or len(target_class_names) == 0:
            return gt_boxes

        # 构建类别名称到索引的映射
        class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # 将目标类别名称转换为索引
        target_class_indices = []
        for name in target_class_names:
            if name in class_name_to_idx:
                target_class_indices.append(class_name_to_idx[name])
            else:
                print(f"[Warning] 类别 '{name}' 不在数据集的类别列表中: {class_names}")

        if not target_class_indices:
            return gt_boxes.new_empty((0, gt_boxes.shape[1]))

        # 获取真值框的类别标签
        if gt_boxes.shape[1] >= 10:  # nuScenes格式
            class_labels = gt_boxes[:, 9].long()
        else:  # KITTI格式
            class_labels = gt_boxes[:, 7].long()

        # 创建类别掩码
        class_mask = torch.zeros_like(class_labels, dtype=torch.bool)
        for cls_idx in target_class_indices:
            class_mask |= (class_labels == cls_idx)

        return gt_boxes[class_mask]

    def _generate_gt_voxel_maps(self, batch_dict, voxel_coords):
        """优化版的真值地图生成，支持按类别筛选"""
        if not self.training or 'gt_boxes' not in batch_dict:
            device = batch_dict['points'].device
            batch_dict['gt_fill_coords'] = torch.zeros((0, 4), dtype=torch.int32, device=device)
            batch_dict['gt_reference_coords'] = torch.zeros((0, 4), dtype=torch.int32, device=device)
            return batch_dict

        gt_boxes = batch_dict['gt_boxes']
        batch_size = batch_dict['batch_size']
        device = gt_boxes.device

        # 获取数据集类别名称
        class_names = batch_dict.get('class_names',
                                     ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                      'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'])

        gt_fill_coords_list = []
        gt_reference_coords_list = []

        all_voxel_coords = self._all_voxel_coords
        voxel_centers = self._voxel_centers_precomputed

        for batch_idx in range(batch_size):
            batch_gt_boxes = gt_boxes[batch_idx]

            # 过滤无效真值框
            valid_mask = (batch_gt_boxes[:, 3:6].sum(dim=1) > 0)
            valid_gt_boxes = batch_gt_boxes[valid_mask]

            if len(valid_gt_boxes) == 0:
                continue

            # 按类别筛选
            if self.target_class_names:
                valid_gt_boxes = self._filter_boxes_by_class(
                    valid_gt_boxes, class_names, self.target_class_names
                )

            if len(valid_gt_boxes) == 0:
                continue

            # 生成填充地图
            in_box_mask = self._points_in_boxes_3d_gpu(voxel_centers, valid_gt_boxes[:, 0:7])
            combined_mask = in_box_mask.any(dim=1)

            inside_voxel_coords = all_voxel_coords[combined_mask]
            if len(inside_voxel_coords) > 0:
                batch_indices = torch.full((len(inside_voxel_coords), 1), batch_idx,
                                           dtype=torch.int32, device=device)
                coords_with_batch = torch.cat([batch_indices, inside_voxel_coords], dim=1)
                gt_fill_coords_list.append(coords_with_batch)

            # 生成参考地图
            if voxel_coords is not None and len(voxel_coords) > 0:
                batch_mask = voxel_coords[:, 0] == batch_idx
                batch_voxel_coords = voxel_coords[batch_mask]

                if len(batch_voxel_coords) > 0:
                    voxel_centers_ref = (batch_voxel_coords[:,
                                         1:4].float() + 0.5) * self.voxel_size + self.point_cloud_range[:3]

                    in_box_mask_ref = self._points_in_boxes_3d_gpu(voxel_centers_ref, valid_gt_boxes[:, 0:7])
                    combined_mask_ref = in_box_mask_ref.any(dim=1)

                    inside_voxel_coords_ref = batch_voxel_coords[combined_mask_ref]
                    if len(inside_voxel_coords_ref) > 0:
                        gt_reference_coords_list.append(inside_voxel_coords_ref)

        # 合并结果
        batch_dict['gt_fill_coords'] = torch.cat(gt_fill_coords_list, dim=0) if gt_fill_coords_list else \
            torch.zeros((0, 4), dtype=torch.int32, device=device)
        batch_dict['gt_reference_coords'] = torch.cat(gt_reference_coords_list, dim=0) if gt_reference_coords_list else \
            torch.zeros((0, 4), dtype=torch.int32, device=device)

        return batch_dict

    def forward(self, batch_dict, **kwargs):
        """
        前向传播函数，保持原有逻辑，集成类别筛选功能
        """
        points = batch_dict['points']

        # 体素坐标计算
        points_coords = torch.floor(
            (points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        # 生成唯一体素坐标
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        # 特征计算
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        # 特征拼接
        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # PFN层处理
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # 生成体素坐标
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        # 生成真值地图（包含类别筛选）
        batch_dict = self._generate_gt_voxel_maps(batch_dict, voxel_coords)

        # 存储结果
        batch_dict['pillar_features'] = features
        batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict