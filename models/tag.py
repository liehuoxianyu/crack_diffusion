import torch
import torch.nn as nn
from diffusers.models.controlnet import ControlNetOutput


class TimeAdaptiveGating(nn.Module):
    """
    时间步自适应门控融合模块 (TAG)。
    输入：时间步的 embedding
    输出：num_blocks 个介于 0~1 之间的缩放权重。
    """

    def __init__(self, time_embed_dim=1280, num_blocks=13):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_blocks),
            nn.Sigmoid(),
        )

    def forward(self, t_emb):
        # t_emb shape: (B, D)
        # 返回 scales shape: (B, num_blocks)
        return self.mlp(t_emb)


def _resolve_time_embed_dim(controlnet: nn.Module) -> int:
    time_embedding = getattr(controlnet, "time_embedding", None)
    if time_embedding is not None and hasattr(time_embedding, "linear_2"):
        return int(time_embedding.linear_2.out_features)
    if time_embedding is not None and hasattr(time_embedding, "linear_1"):
        return int(time_embedding.linear_1.out_features)
    return 1280


def inject_tag_into_controlnet(controlnet: nn.Module):
    """
    动态修改 ControlNetModel 的 forward 函数，将 TAG 模块无缝注入。
    """
    if getattr(controlnet, "_tag_injected", False):
        return controlnet

    controlnet_device = getattr(controlnet, "device", next(controlnet.parameters()).device)
    controlnet_dtype = getattr(controlnet, "dtype", next(controlnet.parameters()).dtype)
    time_embed_dim = _resolve_time_embed_dim(controlnet)

    # 1. 初始化并挂载 TAG 模块到 controlnet 上
    controlnet.tag_module = TimeAdaptiveGating(time_embed_dim=time_embed_dim).to(
        device=controlnet_device, dtype=controlnet_dtype
    )

    # 2. 保存原生的 forward 方法
    original_forward = controlnet.forward
    controlnet._tag_original_forward = original_forward

    # 3. 定义新的 forward 方法
    def new_forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        conditioning_scale=1.0,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        added_cond_kwargs=None,
        cross_attention_kwargs=None,
        guess_mode=False,
        return_dict=True,
    ):
        # 首先调用原生 forward 拿到原本的 13 个特征图输出
        out = original_forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            cross_attention_kwargs=cross_attention_kwargs,
            guess_mode=guess_mode,
            return_dict=True,
        )

        down_samples = list(out.down_block_res_samples)
        mid_sample = out.mid_block_res_sample

        # 获取时间步 embedding (与本地 ControlNet forward 逻辑对齐)
        if not torch.is_tensor(timestep):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timesteps = timestep[None].to(sample.device)
        else:
            timesteps = timestep.to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        t_emb = self.time_embedding(t_emb, timestep_cond)

        # 通过 TAG 模块计算缩放权重
        scales = self.tag_module(t_emb)
        expected = len(down_samples) + 1
        if scales.shape[1] != expected:
            raise ValueError(f"TAG scales mismatch: got {scales.shape[1]}, expected {expected}")

        # 将权重应用到特征图上
        for i in range(len(down_samples)):
            scale_factor = scales[:, i].to(dtype=down_samples[i].dtype).view(-1, 1, 1, 1)
            down_samples[i] = down_samples[i] * scale_factor

        mid_scale = scales[:, -1].to(dtype=mid_sample.dtype).view(-1, 1, 1, 1)
        mid_sample = mid_sample * mid_scale

        if not return_dict:
            return (tuple(down_samples), mid_sample)

        return ControlNetOutput(
            down_block_res_samples=tuple(down_samples),
            mid_block_res_sample=mid_sample,
        )

    # 4. 替换 forward 方法
    controlnet.forward = new_forward.__get__(controlnet, type(controlnet))
    controlnet._tag_injected = True
    return controlnet
