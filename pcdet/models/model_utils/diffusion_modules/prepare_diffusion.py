"""
扩散模型工具函数库
包含与扩散模型相关的坐标处理、特征提取和条件生成功能
"""

import torch
import torch.nn as nn
import torch_scatter
import math
import numpy as np

from .cond_diffusion_denoise import Cond_Diff_Denoise
from .inference_network import ResidualMLPBlock


class NoiseLevelAdapter(nn.Module):
    """
    噪声水平适配器
    功能：将不同样本的噪声水平调整到相似范围，便于扩散模型处理

    设计原则：
    1. 复用ResidualMLPBlock结构
    2. 输出权重向量（每个特征维度独立调整）
    3. 实现可学习的噪声混合
    """

    def __init__(self, latent_dim=16, config=None):
        super().__init__()

        if config is None:
            config = {}

        # 从配置读取参数
        self.use_residual = getattr(config, 'use_residual', True)
        self.residual_skip = getattr(config, 'residual_skip', True)
        self.hidden_dims = getattr(config, 'hidden_dims', [32, 16])
        self.dropout = getattr(config, 'dropout', 0.1)
        self.min_weight = getattr(config, 'min_weight', 0.3)
        self.max_weight = getattr(config, 'max_weight', 0.7)
        self.temperature = getattr(config, 'temperature', 1.0)

        # 构建小型残差MLP网络
        layers = []
        input_dim = latent_dim

        # 复用ResidualMLPBlock结构
        for hidden_dim in self.hidden_dims:
            if self.use_residual:
                layers.append(ResidualMLPBlock(
                    input_dim, hidden_dim,
                    config={
                        'dropout': self.dropout,
                        'use_residual': self.use_residual,
                        'residual_skip': self.residual_skip
                    }
                ))
            else:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
            input_dim = hidden_dim

        # 输出层：通过sigmoid输出0-1的权重
        layers.append(nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid()  # 确保权重在0-1之间
        ))

        self.mlp = nn.Sequential(*layers)

        # 调试开关
        self.debug_prefix = False

    def forward(self, latent_features, coords=None):
        """
        前向传播：计算噪声混合权重并应用

        参数:
            latent_features: 潜变量特征 [N, D]
            coords: 坐标信息 [N, 4]（可选，用于位置引导）

        返回:
            mixed_latent: 混合后的特征 [N, D]
            adapter_weight: 适配器权重 [N, D]
        """
        # 计算权重向量
        raw_weight = self.mlp(latent_features)  # [N, D]

        # 应用温度参数
        if self.temperature != 1.0:
            raw_weight = torch.sigmoid(torch.logit(raw_weight) / self.temperature)

        # 权重范围约束
        adapter_weight = raw_weight * (self.max_weight - self.min_weight) + self.min_weight

        # 生成标准高斯噪声
        standard_noise = torch.randn_like(latent_features)

        # 噪声混合：weight * original + (1-weight) * noise
        mixed_latent = (adapter_weight * latent_features +
                       (1 - adapter_weight) * standard_noise)

        if self.debug_prefix:
            print(f"[NoiseLevelAdapter] 输入形状: {latent_features.shape}")
            print(f"[NoiseLevelAdapter] 权重均值: {adapter_weight.mean().item():.3f}")
            print(f"[NoiseLevelAdapter] 权重范围: [{adapter_weight.min().item():.3f}, {adapter_weight.max().item():.3f}]")

        return mixed_latent, adapter_weight


# 在 prepare_diffusion.py 中添加新的匹配算法
class CoordinateMatcher:
    """高效坐标匹配器"""

    @staticmethod
    def hash_based_match(voxel_coords, reference_coords, spatial_shape, device):
        """
        基于哈希的快速坐标匹配
        """
        d, h, w = spatial_shape

        # 创建哈希函数
        def create_hash(coords):
            # 使用更高效的哈希函数
            return (coords[:, 0] * (d * h * w) +
                    coords[:, 1] * (h * w) +
                    coords[:, 2] * w +
                    coords[:, 3])

        # 创建参考坐标的哈希集合
        ref_hashes = create_hash(reference_coords)
        ref_hash_set = set(ref_hashes.cpu().numpy())

        # 分批处理体素坐标（避免内存爆炸）
        batch_size = 100000  # 根据GPU内存调整
        matched_mask = torch.zeros(len(voxel_coords), dtype=torch.bool, device=device)

        for i in range(0, len(voxel_coords), batch_size):
            end_idx = min(i + batch_size, len(voxel_coords))
            batch_coords = voxel_coords[i:end_idx]

            # 计算批次哈希
            batch_hashes = create_hash(batch_coords)
            batch_hashes_cpu = batch_hashes.cpu().numpy()

            # 使用集合操作进行快速匹配
            batch_mask = np.array([hash_val in ref_hash_set for hash_val in batch_hashes_cpu])
            matched_mask[i:end_idx] = torch.tensor(batch_mask, device=device)

        return matched_mask

    @staticmethod
    def gpu_accelerated_match(voxel_coords, reference_coords, spatial_shape, device):
        """
        GPU加速的坐标匹配（最推荐）
        """
        d, h, w = spatial_shape
        hw = h * w
        dhw = d * hw

        # 线性化坐标
        def linearize(coords):
            return (coords[:, 0] * dhw +
                    coords[:, 1] * hw +
                    coords[:, 2] * w +
                    coords[:, 3])

        ref_linear = linearize(reference_coords)
        voxel_linear = linearize(voxel_coords)

        # 使用torch.unique和torch.isin进行GPU加速匹配
        unique_ref = torch.unique(ref_linear)
        matched_mask = torch.isin(voxel_linear, unique_ref)

        return matched_mask

class DiffusionCoordinateProcessor:
    """
    处理扩散模型相关的坐标操作
    """

    @staticmethod
    def downsample_coords(coords, down_scale, spatial_shape):
        """
        下采样坐标张量
        """
        if coords is None or len(coords) == 0:
            return None

        # 使用向量化操作
        downsampled_coords = coords.clone().float()

        # 批量除法运算
        downsampled_coords[:, 3] = torch.floor(coords[:, 3] / down_scale[0])  # X
        downsampled_coords[:, 2] = torch.floor(coords[:, 2] / down_scale[1])  # Y
        downsampled_coords[:, 1] = torch.floor(coords[:, 1] / down_scale[2])  # Z

        # 使用clamp确保坐标在有效范围内
        d, h, w = spatial_shape
        downsampled_coords[:, 3].clamp_(0, w - 1)
        downsampled_coords[:, 2].clamp_(0, h - 1)
        downsampled_coords[:, 1].clamp_(0, d - 1)

        return downsampled_coords.long()

    @staticmethod
    def generate_diffusion_coords(original_coords, num_points, coords_shift, spatial_shape):
        """
        基于扩散模型学习到的分布生成新坐标
        参数:
            original_coords: 原始坐标 [N, 4]
            num_points: 每个坐标生成的点数
            coords_shift: 坐标偏移量
            spatial_shape: 空间形状 (D, H, W)
        返回:
            生成的坐标张量
        """
        d, h, w = spatial_shape

        # 扩展原始坐标
        expanded_coords = original_coords.repeat(num_points, 1)

        # 生成随机偏移（基于学习到的分布）
        batch_size = expanded_coords[:, 0].max() + 1
        generated_coords_list = []

        for i in range(batch_size):
            batch_mask = expanded_coords[:, 0] == i
            if not batch_mask.any():
                continue

            batch_coords = expanded_coords[batch_mask]
            N_batch = len(batch_coords)

            # 生成随机偏移（正态分布）
            shift_z = torch.randn(N_batch, 1, device=original_coords.device) * coords_shift
            shift_y = torch.randn(N_batch, 1, device=original_coords.device) * coords_shift
            shift_x = torch.randn(N_batch, 1, device=original_coords.device) * coords_shift

            new_coords = batch_coords.clone()
            new_coords[:, 1:2] = (new_coords[:, 1:2] + shift_z).clamp(0, d - 1)  # Z
            new_coords[:, 2:3] = (new_coords[:, 2:3] + shift_y).clamp(0, h - 1)  # Y
            new_coords[:, 3:4] = (new_coords[:, 3:4] + shift_x).clamp(0, w - 1)  # X

            generated_coords_list.append(new_coords)

        return torch.cat(generated_coords_list) if generated_coords_list else torch.tensor([],
                                                                                           device=original_coords.device)

    @staticmethod
    def validate_coordinate_extraction(matched_coords, reference_coords, spatial_shape):
        """
        验证坐标提取结果的准确性
        参数:
            matched_coords: 匹配的坐标
            reference_coords: 参考坐标
            spatial_shape: 空间形状
        """
        d, h, w = spatial_shape

        # 验证所有匹配坐标确实在参考坐标中
        matched_linear = (matched_coords[:, 0] * (d * h * w) +
                          matched_coords[:, 1] * (h * w) +
                          matched_coords[:, 2] * w +
                          matched_coords[:, 3])

        ref_linear = (reference_coords[:, 0] * (d * h * w) +
                      reference_coords[:, 1] * (h * w) +
                      reference_coords[:, 2] * w +
                      reference_coords[:, 3])

        ref_set = set(ref_linear.cpu().numpy())
        for key in matched_linear:
            assert key.item() in ref_set, f" 扩散条件准备: 提取的坐标{key.item()}不在参考坐标中"

        print(f"[DEBUG] 扩散条件准备: 验证通过 - 所有匹配坐标均在参考坐标范围内")

class DiffusionFeatureExtractor:
    """
    处理扩散模型相关的特征提取操作
    包括真值框特征提取、特征融合、潜变量编码等
    """

    @staticmethod
    def extract_gt_reference_features(bs_voxel_coords, bs_voxel_features, reference_coords,
                                            spatial_shape, device, debug_prefix=False):
        """
        批量化优化版：使用向量化操作处理整个batch
        """
        if reference_coords is None or len(reference_coords) == 0:
            return None, None, None, None, None

        # 设备一致性处理
        bs_voxel_coords = bs_voxel_coords.to(device)
        bs_voxel_features = bs_voxel_features.to(device)
        reference_coords = reference_coords.to(device)

        d, h, w = spatial_shape
        hw = h * w
        dhw = d * hw

        # 批量化线性化函数
        def batch_linearize(coords):
            return (coords[:, 0] * dhw +
                    coords[:, 1] * hw +
                    coords[:, 2] * w +
                    coords[:, 3])

        # 线性化所有坐标
        ref_linear = batch_linearize(reference_coords)
        voxel_linear = batch_linearize(bs_voxel_coords)

        # 使用批量化匹配（优化关键点）
        # 方法：使用torch.bucketize进行快速批匹配
        sorted_ref, ref_indices = torch.sort(ref_linear)

        # 使用bucketize进行快速匹配（比searchsorted更快）
        bucket_indices = torch.bucketize(voxel_linear, sorted_ref)

        # 防止索引越界
        bucket_indices = torch.clamp(bucket_indices, 0, len(sorted_ref) - 1)

        # 检查匹配
        matched_mask = sorted_ref[bucket_indices] == voxel_linear

        # 提取匹配结果
        matched_features = bs_voxel_features[matched_mask]
        matched_coords = bs_voxel_coords[matched_mask]

        # 创建蒙版地图（使用稀疏张量优化内存）
        mask_features = torch.zeros_like(bs_voxel_features)
        mask_coords = torch.zeros_like(bs_voxel_coords)

        mask_features[matched_mask] = bs_voxel_features[matched_mask]
        mask_coords[matched_mask] = bs_voxel_coords[matched_mask]

        if debug_prefix:
            print(f"[DEBUG] 批量化蒙版地图创建完成")
            print(f"        - 匹配体素数量: {matched_mask.sum().item()}")
            print(f"        - 匹配率: {matched_mask.sum().item() / len(bs_voxel_features):.3f}")

        return matched_features, matched_coords, mask_features, mask_coords, matched_mask

    @staticmethod
    def create_sparse_mask(features, coords, matched_mask, device):
        """
        创建稀疏蒙版地图，避免全零张量的内存浪费
        """
        # 只存储非零元素
        non_zero_indices = torch.where(matched_mask)[0]

        if len(non_zero_indices) == 0:
            # 返回空的稀疏表示
            return torch.zeros(0, features.shape[1], device=device), \
                torch.zeros(0, 4, device=device, dtype=torch.long), \
                matched_mask

        sparse_features = features[non_zero_indices]
        sparse_coords = coords[non_zero_indices]

        return sparse_features, sparse_coords, matched_mask

    @staticmethod
    def prepare_diffusion_input(bs_voxel_features, bs_voxel_coords, reference_coords,
                                spatial_shape, device, debug_prefix=False):
        """
        修复版：准备扩散模型的输入条件
        保持蒙版地图全0模式，确保形状一致性
        """
        # 处理空reference_coords的情况
        if reference_coords is None or len(reference_coords) == 0:
            feature_dim = bs_voxel_features.shape[1] if len(bs_voxel_features) > 0 else 64

            # 创建全0蒙版地图，保持形状一致性
            gt_mask_features = torch.zeros_like(bs_voxel_features)
            gt_mask_coords = torch.zeros_like(bs_voxel_coords)
            mask_indicator = torch.zeros(len(bs_voxel_features), dtype=torch.bool, device=device)

            # 返回空的匹配特征
            empty_features = torch.empty(0, feature_dim, device=device)
            empty_coords = torch.empty(0, 4, device=device, dtype=torch.long)

            return empty_features, empty_coords, gt_mask_features, gt_mask_coords, mask_indicator

        # 使用GPU加速匹配
        matched_mask = CoordinateMatcher.gpu_accelerated_match(
            bs_voxel_coords, reference_coords, spatial_shape, device)

        # 提取匹配特征
        matched_features = bs_voxel_features[matched_mask]
        matched_coords = bs_voxel_coords[matched_mask]

        # 修复：创建全0蒙版地图，确保形状一致性
        gt_mask_features = torch.zeros_like(bs_voxel_features)
        gt_mask_coords = torch.zeros_like(bs_voxel_coords)

        # 只在匹配位置填充特征
        gt_mask_features[matched_mask] = bs_voxel_features[matched_mask]
        gt_mask_coords[matched_mask] = bs_voxel_coords[matched_mask]

        if debug_prefix:
            print(f"[DEBUG] 扩散条件准备: 蒙版地图创建完成（全0模式）")
            print(f"        - 匹配体素数量: {matched_mask.sum().item()}/{len(bs_voxel_features)}")
            print(f"        - 蒙版地图形状: {gt_mask_features.shape}")

        return matched_features, matched_coords, gt_mask_features, gt_mask_coords, matched_mask


class SimpleVoxelMerging(nn.Module):
    """
    简化版潜变量降采样模块
    """

    def __init__(self):
        super().__init__()

    def forward(self, voxel_features, voxel_coords, spatial_shape,
                down_scale=[2, 2, 2],
                training=False):
        """
        参数:
            voxel_features: 体素特征 [N, 64]
            voxel_coords: 对应体素坐标 [N, 4]
            spatial_shape: 原始空间形状 (D, H, W)
        返回:
            merged_voxel: 降采样后的体素 [M, 64]
            merged_coords: 降采样后的坐标 [M, 4]
            unq_inv: 索引映射 [N]
            new_sparse_shape: 新的空间形状
        """
        # 坐标缩放
        coords = voxel_coords.clone().float()
        coords[:, 3] = torch.floor(coords[:, 3] / down_scale[0])  # X
        coords[:, 2] = torch.floor(coords[:, 2] / down_scale[1])  # Y
        coords[:, 1] = torch.floor(coords[:, 1] / down_scale[2])  # Z
        coords = coords.long()

        # 坐标线性化
        d, h, w = spatial_shape
        scale_xyz = (d // down_scale[2]) * (h // down_scale[1]) * (w // down_scale[0])
        scale_yz = (d // down_scale[2]) * (h // down_scale[1])
        scale_z = (d // down_scale[2])
        merge_coords = coords[:, 0] * scale_xyz + coords[:, 3] * scale_yz + coords[:, 2] * scale_z + coords[:, 1]

        # 计算新的空间形状
        new_d = math.ceil(d / down_scale[2])
        new_h = math.ceil(h / down_scale[1])
        new_w = math.ceil(w / down_scale[0])
        new_sparse_shape = [new_d, new_h, new_w]

        # 特征聚合
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True)
        merged_voxel_mean = torch_scatter.scatter_mean(voxel_features, unq_inv, dim=0)
        if training:
            merged_voxel_max, _ = torch_scatter.scatter_max(voxel_features, unq_inv, dim=0)
        else:
            merged_voxel_max = merged_voxel_mean

        # 生成降采样后的坐标
        unq_coords = unq_coords.int()
        merged_coords = torch.stack((
            unq_coords // scale_xyz,
            (unq_coords % scale_xyz) // scale_yz,
            (unq_coords % scale_yz) // scale_z,
            unq_coords % scale_z
        ), dim=1)
        merged_coords = merged_coords[:, [0, 3, 2, 1]]  # 调整顺序为 (batch_idx, z, y, x)

        return merged_voxel_mean, merged_voxel_max, merged_coords, unq_inv, new_sparse_shape

class VoxelLatentEncoder(nn.Module):
    """
    增强版编码器：使用深层MLP将体素特征和坐标映射到潜空间。
    输入: features [N, C],
    输出: latent [N, D], D=8-16
    """

    def __init__(self, input_dim=64, latent_dim=16):
        super().__init__()
        temp_dim = max(input_dim // 2, latent_dim * 2) # 32
        self.voxel_to_latent_encoder = nn.Sequential(
            nn.Linear(input_dim, temp_dim),
            nn.ReLU(),
            nn.Linear(temp_dim, latent_dim)  # 坐标编码为16维
        )

    def forward(self, features):
        latent = self.voxel_to_latent_encoder(features)  # [N, latent_dim]
        return latent

class VoxelLatentDecoder(nn.Module):
    """
    潜变量解码器：将扩散模型输出的潜变量解码回原始体素特征维度
    输入: latent [N, 16]
    输出: features [N, 64]
    """

    def __init__(self, latent_dim=16, output_dim=64):
        super().__init__()
        temp_dim = max(latent_dim * 2, output_dim // 2)  # 32
        self.latent_to_voxel_decoder = nn.Sequential(
            nn.Linear(latent_dim, temp_dim),
            nn.ReLU(),
            nn.Linear(temp_dim, output_dim)  # 解码回64维体素特征
        )

    def forward(self, latent):
        features = self.latent_to_voxel_decoder(latent)
        return features

class SimpleVoxelExpanding(nn.Module):
    """
    简化版潜变量上采样模块
    功能：利用索引映射将低分辨率潜变量复制到高分辨率位置
    输入: lowres_latent [M, D], lowres_coords [M, 4], unq_inv [N] (来自降采样)
    输出: 上采样后的潜变量 [N, D]
    """
    def __init__(self):
        super().__init__()

    def forward(self, lower_data, unq_inv):
        """
        参数:
            lowres_latent: 降采样后的潜变量 [M, D]
            lowres_coords: 降采样后的坐标 [M, 4]（未直接使用，但保留以兼容）
            unq_inv: 索引映射 [N]，来自降采样步骤
        返回:
            upsampled_latent: 上采样后的潜变量 [N, D]
        """
        # 直接通过索引映射复制特征：每个高分辨率位置获取对应低分辨率特征
        upsampled_data = torch.gather(lower_data, 0, unq_inv.unsqueeze(1).repeat(1, lower_data.shape[1]))
        return upsampled_data

class DiffusionModelManager:
    """
    管理扩散模型的训练流程
    包括训练/推断模式切换、梯度控制、损失计算等
    """

    def __init__(self, diff_model):

        self.adapter_enable = getattr(diff_model.DIFF_MODEL_CFG, 'adapter_enable', True)
        self.use_mixed_as_start = getattr(diff_model.DIFF_MODEL_CFG, 'use_mixed_as_start', True)
        self.adapter_weighted_loss = getattr(diff_model.DIFF_MODEL_CFG, 'adapter_weighted_loss', False)
        self.all_down_scale = getattr(diff_model.LATENT, 'latent_down_scale', [[2, 2, 8], [2, 2, 4], [2, 2, 2], [2, 2, 1]])

        self.voxel_feature_dim_cfg = getattr(diff_model.LATENT, 'voxel_feature_dim', 64)
        self.voxel_latent_dim_cfg = getattr(diff_model.LATENT, 'voxel_latent_dim', 16)

        self.voxel_downsampler = SimpleVoxelMerging()

        self.latent_encoder = VoxelLatentEncoder(
            input_dim=self.voxel_feature_dim_cfg,
            latent_dim=self.voxel_latent_dim_cfg
        )

        # 初始化噪声水平适配器（如果启用）
        if self.adapter_enable:
            # 从配置中读取噪声适配器参数
            noise_adapter_cfg = getattr(diff_model.DIFF_MODEL_CFG, 'latent_noise_adapter', {})
            self.noise_adapter = NoiseLevelAdapter(
                latent_dim=self.voxel_latent_dim_cfg,
                config=noise_adapter_cfg
            )

        # 新增解码器
        self.latent_decoder = VoxelLatentDecoder(
            latent_dim=self.voxel_latent_dim_cfg,
            output_dim=self.voxel_feature_dim_cfg
        )

        self.voxel_upsampler = SimpleVoxelExpanding()

        diffusion_model_instance_cfg = getattr(diff_model, 'DIFF_MODEL_CFG', {})
        self.diffusion_model_instance = Cond_Diff_Denoise(
            model_cfg=diffusion_model_instance_cfg,
            embed_dim=self.voxel_latent_dim_cfg
        )

        # self.diff_noise_scale = diff_model.DIFF_PROCESS.diff_noise_scale

        self._device_synced = False

        self.debug_prefix = False

    def _sync_device(self, target_device):
        """同步模型设备到输入张量所在的设备"""
        if not self._device_synced:
            # 同步所有子模块到目标设备
            self.latent_encoder = self.latent_encoder.to(target_device)
            self.latent_decoder = self.latent_decoder.to(target_device)  # 新增解码器同步
            # 关键修复：同步噪声适配器
            if self.adapter_enable and hasattr(self, 'noise_adapter'):
                self.noise_adapter = self.noise_adapter.to(target_device)
            # 同步扩散模型
            self.diffusion_model_instance = self.diffusion_model_instance.to(target_device)
            # 同步下采样器和上采样器
            self.voxel_downsampler = self.voxel_downsampler.to(target_device)
            self.voxel_upsampler = self.voxel_upsampler.to(target_device)
            self._device_synced = True

    def apply_diffusion(self, bs_voxel_features, bs_voxel_coords,
                        reference_coords, spatial_shape, device, lionblock_idx, training=True):
        """
        应用扩散模型
        参数:
            bs_voxel_features: 体素特征
            bs_voxel_coords: 体素坐标
            reference_coords: 参考坐标
            spatial_shape: 空间形状
            device: 计算设备
            lionblock_idx: LION块索引
            training: 是否训练模式
        返回:
            增强后的特征和扩散损失
        """
        self._sync_device(device)

        if training:
            if self.debug_prefix:
                print(f"[DEBUG] 已进入扩散训练流程")

            # 下采样处理
            bs_voxel_mean_down, _, bs_coords_down, unq_inv, spatial_shape_down = self.voxel_downsampler(
                voxel_features=bs_voxel_features,
                voxel_coords=bs_voxel_coords,
                spatial_shape=spatial_shape,
                down_scale=self.all_down_scale[lionblock_idx],
                training=training
            )

            # 下采样参考坐标
            reference_coords_down = DiffusionCoordinateProcessor.downsample_coords(
                reference_coords, self.all_down_scale[lionblock_idx], spatial_shape_down
            ) if reference_coords is not None else None

            # 准备扩散输入
            _, _, gt_mask_features_down, gt_mask_coords_down, _ = (
                DiffusionFeatureExtractor.prepare_diffusion_input(
                    bs_voxel_mean_down,
                    bs_coords_down,
                    reference_coords_down,
                    spatial_shape_down,
                    device,
                    debug_prefix=self.debug_prefix
                ))

            # 编码为潜变量
            bs_voxel_latent = self.latent_encoder(bs_voxel_mean_down)
            gt_mask_latent = self.latent_encoder(gt_mask_features_down)

            # 应用噪声适配器（如果启用）
            if self.adapter_enable:
                bs_voxel_latent, adapter_weight = self.noise_adapter(
                    bs_voxel_latent,
                    coords=bs_coords_down
                )
                if self.debug_prefix:
                    print(f"[DEBUG] 噪声适配器权重形状: {adapter_weight.shape}")
            else:
                adapter_weight = None

            # 准备扩散模型输入
            diffusion_input = {
                'bs_voxel_latent': bs_voxel_latent,
                'gt_mask_latent': gt_mask_latent,
                'bs_voxel_coords': bs_coords_down,
                'gt_mask_coords': gt_mask_coords_down
            }

            # 如果需要加权损失，传递适配器权重
            if self.adapter_weighted_loss and adapter_weight is not None:
                diffusion_input['adapter_weight'] = adapter_weight

            # 扩散模型处理
            enhanced_latent, diffusion_loss = self.diffusion_model_instance(diffusion_input)

            # 上采样和解码
            enhanced_latent_up = self.voxel_upsampler(lower_data=enhanced_latent, unq_inv=unq_inv)
            enhanced_voxel = self.latent_decoder(enhanced_latent_up)

            # 残差连接
            enhanced_voxel = enhanced_voxel + bs_voxel_features

            return enhanced_voxel, diffusion_loss

        else:
            # 推断模式
            with torch.no_grad():
                if self.debug_prefix:
                    print(f"[DEBUG] 推断流程")

                # 下采样处理
                bs_voxel_mean_down, _, bs_coords_down, unq_inv, _ = self.voxel_downsampler(
                    voxel_features=bs_voxel_features,
                    voxel_coords=bs_voxel_coords,
                    spatial_shape=spatial_shape,
                    down_scale=self.all_down_scale[lionblock_idx],
                    training=False
                )

                # 推断时没有真值框，创建零值蒙版
                gt_mask_features_down = torch.zeros_like(bs_voxel_mean_down)
                gt_mask_coords_down = bs_coords_down.clone()

                # 编码为潜变量
                bs_voxel_latent = self.latent_encoder(bs_voxel_mean_down)
                gt_mask_latent = self.latent_encoder(gt_mask_features_down)

                # 应用噪声适配器（如果启用）
                if self.adapter_enable:
                    bs_voxel_latent, _ = self.noise_adapter(
                        bs_voxel_latent,
                        coords=bs_coords_down
                    )

                # 准备扩散模型输入
                diffusion_input = {
                    'bs_voxel_latent': bs_voxel_latent,
                    'gt_mask_latent': gt_mask_latent,
                    'bs_voxel_coords': bs_coords_down,
                    'gt_mask_coords': gt_mask_coords_down
                }

                # 扩散模型处理
                enhanced_latent, _ = self.diffusion_model_instance(diffusion_input)

                # 上采样和解码
                enhanced_latent_up = self.voxel_upsampler(lower_data=enhanced_latent, unq_inv=unq_inv)
                enhanced_voxel = self.latent_decoder(enhanced_latent_up)

                # 残差连接
                enhanced_voxel = enhanced_voxel + bs_voxel_features

            return enhanced_voxel, 0.0
