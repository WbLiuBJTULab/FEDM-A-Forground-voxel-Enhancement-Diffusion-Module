import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualMLPBlock(nn.Module):
    """
    残差MLP块
    支持可配置的残差连接和跳跃连接
    修改：BatchNorm1d -> LayerNorm，适配稀疏体素变长输入
    """

    def __init__(self, in_dim, out_dim, config=None):
        super().__init__()
        if config is None:
            config = {}

        # 从配置读取参数，提供默认值
        self.dropout = config.get('dropout', 0.1)
        self.use_residual = config.get('use_residual', True)
        self.residual_skip = config.get('residual_skip', True)

        # 构建主干网络
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),          # BatchNorm1d -> LayerNorm
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, out_dim),
        )

        # 构建跳跃连接
        if self.use_residual and self.residual_skip and in_dim != out_dim:
            self.skip_connect = nn.Linear(in_dim, out_dim)
        elif self.use_residual and self.residual_skip:
            self.skip_connect = nn.Identity()
        else:
            self.skip_connect = None
            self.use_residual = False  # 禁用残差连接

        # Post-Norm结构
        self.norm = nn.LayerNorm(out_dim) if self.use_residual else nn.Identity()  # BatchNorm1d -> LayerNorm
        self.act = nn.SiLU() if self.use_residual else nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 [N, in_dim]

        返回:
            out: 输出张量 [N, out_dim]
        """
        if self.use_residual and self.skip_connect is not None:
            identity = self.skip_connect(x)
        else:
            identity = x

        out = self.net(x)

        if self.use_residual:
            out = out + identity

        out = self.norm(out)
        out = self.act(out)
        return out


class GatedMLPBlock(nn.Module):
    """
    轻量门控残差MLP (SiLU-Gate)
    设计: gate分支(SiLU) 与 value分支(线性) 逐元素乘，再投影。
    优势: 门控机制天然适合扩散噪声预测，对前景/背景分离更敏感。
    参数量: 仅比标准残差块多一个Linear投影，但表达能力更强。
    """

    def __init__(self, in_dim, out_dim, config=None):
        super().__init__()
        if config is None:
            config = {}

        self.dropout = config.get('dropout', 0.1)
        self.use_residual = config.get('use_residual', True)
        self.residual_skip = config.get('residual_skip', True)

        # 门控分支: SiLU激活
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )
        # 价值分支: 线性
        self.value = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Identity()
        )
        # 融合投影
        self.fuse = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, out_dim)
        )

        if self.use_residual and self.residual_skip and in_dim != out_dim:
            self.skip_connect = nn.Linear(in_dim, out_dim)
        elif self.use_residual and self.residual_skip:
            self.skip_connect = nn.Identity()
        else:
            self.skip_connect = None
            self.use_residual = False

        self.norm = nn.LayerNorm(out_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = self.skip_connect(x) if self.use_residual and self.skip_connect is not None else x
        g = self.gate(x)
        v = self.value(x)
        h = self.fuse(g * v)
        if self.use_residual:
            h = h + identity
        h = self.norm(h)
        return h

class Diffusion_MLPs(nn.Module):
    """
    简化版DiffusionUNet网络
    输入: [N, 32] (16维特征 + 16维噪声掩码)
    输出: [N, 16] (去噪后的潜在特征)
    """

    def __init__(self, config):
        super().__init__()
        # 从配置读取核心参数
        self._parse_config(config)
        # 计算时间步嵌入维度
        self.temb_channels = self.base_channels * self.temb_coefficient  # 8 * 2 = 16
        # 计算总输入维度：特征+时间步嵌入
        self.total_input_dim = self.in_channels + self.temb_channels  # 16 + 16 = 32
        # 构建时间步嵌入网络
        self._build_timestep_network()
        # 构建位置引导网络（如果启用）
        self._build_position_guide()
        # 构建MLP主干网络
        self._build_mlp_layers()
        # 初始化权重
        self._initialize_weights()

    def _parse_config(self, config):
        """解析配置文件参数"""
        self.in_channels = getattr(config, 'in_channels', 16)
        self.out_channels = getattr(config, 'out_channels', 16)
        # === MODIFIED: base_channels 8->16，提高时间步嵌入表达能力 ===
        self.base_channels = getattr(config, 'base_channels', 16)
        self.temb_coefficient = getattr(config, 'temb_coefficient', 2)
        self.dropout = getattr(config, 'dropout', 0.1)

        self.use_residual = getattr(config, 'use_residual', True)
        self.residual_skip = getattr(config, 'residual_skip', True)
        self.mlp_dims = getattr(config, 'mlp_dims', [64, 32])

        # 新增：block_type 支持三档切换 ('gated' | 'residual' | 'sequential')
        # 向后兼容：未配置时根据 use_residual 推断原有行为
        self.block_type = getattr(config, 'block_type', None)
        if self.block_type is None:
            self.block_type = 'gated' if self.use_residual else 'sequential'
        assert self.block_type in ('gated', 'residual', 'sequential'), \
            f"block_type must be 'gated', 'residual', or 'sequential', got {self.block_type}"

        self.use_pos_guide = getattr(config, 'use_pos_guide', True)
        self.pos_embed_dim = getattr(config, 'pos_embed_dim', 8)
        # === MODIFIED: pos_guide_strength 0.5->1.0，充分利用3D坐标条件 ===
        self.pos_guide_strength = getattr(config, 'pos_guide_strength', 1.0)

    def _build_timestep_network(self):
        """构建时间步嵌入网络"""
        self.temb_net = nn.Sequential(
            nn.Linear(self.base_channels, self.temb_channels),
            nn.SiLU(),
            nn.Linear(self.temb_channels, self.temb_channels)
        )

    def _build_position_guide(self):
        """构建位置引导网络（如果启用）"""
        if self.use_pos_guide:
            self.pos_encoder = nn.Sequential(
                nn.Linear(3, self.pos_embed_dim),  # 输入坐标(z, y, x)
                nn.ReLU(),
                nn.Linear(self.pos_embed_dim, self.pos_embed_dim)
            )
            # 投影层：将坐标编码映射到特征维度
            self.pos_proj = nn.Linear(self.pos_embed_dim, self.total_input_dim)
        else:
            self.pos_encoder = None
            self.pos_proj = None

    def _build_mlp_layers(self):
        """构建MLP主干网络层
        支持三档切换：gated(门控残差) / residual(常规残差) / sequential(顺序MLP)
        """
        layers = []
        current_dim = self.total_input_dim

        # 根据 block_type 选择 block 类
        if self.block_type == 'gated':
            block_cls = GatedMLPBlock
        elif self.block_type == 'residual':
            block_cls = ResidualMLPBlock
        else:
            block_cls = None  # sequential

        if block_cls is not None:
            for dim in self.mlp_dims:
                layers.append(block_cls(
                    current_dim, dim,
                    config={
                        'dropout': self.dropout,
                        'use_residual': self.use_residual,
                        'residual_skip': self.residual_skip
                    }
                ))
                current_dim = dim
        else:
            # sequential：顺序MLP，无残差连接
            for dim in self.mlp_dims:
                layers.extend([
                    nn.Linear(current_dim, dim),
                    nn.LayerNorm(dim),
                    nn.SiLU(),
                    nn.Dropout(self.dropout)
                ])
                current_dim = dim

        # 输出层（不使用残差）
        layers.append(nn.Linear(current_dim, self.out_channels))

        # 封装：gated/residual 用 ModuleList（逐层调用），sequential 用 Sequential
        if self.block_type in ('gated', 'residual'):
            self.mlp_layers = nn.ModuleList(layers)
        else:
            self.mlp_layers = nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, t, coords=None):
        """
        前向传播

        参数:
            x: 输入特征 [N, 32]
            t: 时间步 [1] 或 [B]
            coords: 坐标信息 [N, 3] (可选)

        返回:
            h: 输出特征 [N, 16]
        """
        # 1. 时间步嵌入处理
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.size(0) == 1 and x.size(0) > 1:
            t = t.repeat(x.size(0))

        temb = get_timestep_embedding(t, self.base_channels)
        temb = self.temb_net(temb)

        # 确保时间步嵌入与输入特征批次大小匹配
        if x.size(0) != temb.size(0):
            if x.size(0) % temb.size(0) == 0:
                repeat_factor = x.size(0) // temb.size(0)
                temb = temb.repeat(repeat_factor, 1)
            else:
                temb = temb.expand(x.size(0), -1)

        # 2. 特征拼接
        h = torch.cat([x, temb], dim=1)  # [N, 64]

        # 3. 位置引导（如果启用）
        if self.use_pos_guide and coords is not None:
            coords_float = coords.float()
            pos_embed = self.pos_encoder(coords_float)  # [N, pos_embed_dim]
            pos_embed_proj = self.pos_proj(pos_embed)  # [N, 64]

            # 应用引导强度
            pos_embed_proj = pos_embed_proj * self.pos_guide_strength

            # 加法融合
            h = h + pos_embed_proj

        # 4. 通过MLP层
        if isinstance(self.mlp_layers, nn.ModuleList):
            for layer in self.mlp_layers:
                h = layer(h)
        else:
            h = self.mlp_layers(h)

        # === 新增：输出归一化 + 硬裁剪，防止异常大值溢出 RWKV 门控 ===
        h = nn.functional.layer_norm(h, h.shape[-1:])
        h = torch.clamp(h, min=-100.0, max=100.0)
        # ===

        return h


def get_timestep_embedding(timesteps, embedding_dim):
    """
    生成时间步嵌入

    参数:
        timesteps: 时间步张量 [B]
        embedding_dim: 嵌入维度

    返回:
        emb: 时间步嵌入 [B, embedding_dim]
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)

    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))

    return emb
