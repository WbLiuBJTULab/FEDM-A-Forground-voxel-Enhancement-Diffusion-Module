import torch
import torch.nn as nn

from .detector3d_template import Detector3DTemplate


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        # 扩散模型损失配置
        self.diff_cfg = self.model_cfg.BACKBONE_3D.DIFF_MODEL.DIFF_LOSS_CFG
        self.weight_mode = getattr(self.diff_cfg, 'weight_mode', 'uncertainty')

        # 同方差不确定性加权参数（仅 uncertainty 模式且扩散启用时创建）
        if self.diff_cfg.enable and self.weight_mode == 'uncertainty':
            # 初始 log_sigma=0.0 (sigma=1.0)，初始有效权重均为 1.0
            self.log_sigma_rpn = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.log_sigma_diff = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.debug_prefix = False

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        loss = loss_rpn

        if self.diff_cfg.enable:
            diffusion_loss = batch_dict.get('diffusion_loss', 0.0)

            # === MODIFIED: 确保 diffusion_loss 始终是 tensor 且 requires_grad=True ===
            if not isinstance(diffusion_loss, torch.Tensor):
                diffusion_loss = torch.tensor(0.0, device=loss_rpn.device, dtype=loss_rpn.dtype)
            
            # 关键：即使数值为0，也保持计算图连接，防止DDP误判为unused
            # 通过 (diffusion_loss * 0.0) 保持梯度路径，但不影响loss数值
            if self.weight_mode == 'uncertainty':
                precision_rpn = torch.exp(-2 * self.log_sigma_rpn)
                precision_diff = torch.exp(-2 * self.log_sigma_diff)
                loss = (precision_rpn * loss_rpn + self.log_sigma_rpn +
                        precision_diff * diffusion_loss + self.log_sigma_diff)
            else:
                loss = (self.diff_cfg.origin_weight * loss_rpn +
                        self.diff_cfg.fuse_weight * diffusion_loss)

            # 只有真正非零时才打印调试信息
            if self.debug_prefix and diffusion_loss.item() != 0.0:
                print(f"  - loss计算最终统计: {loss.item()}")
                print(f"  - loss计算扩散模型损失: {diffusion_loss.item()}")
                print(f"  - 当前融合模式: {self.weight_mode}")

        return loss, tb_dict, disp_dict

