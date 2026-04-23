import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    空间注意力机制：通过沿着通道维度进行最大池化和平均池化，
    再通过卷积层生成注意力掩码，让网络自适应聚焦于裂缝骨架区域。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        return x * self.sigmoid(scale)


class CAFEEmbedding(nn.Module):
    """
    Condition-Adaptive Feature Extractor (CAFE)
    替换原生 ControlNetConditioningEmbedding。
    输入大小: (B, 3, H, W)
    输出大小: (B, 320, H/8, W/8)
    """

    def __init__(self, conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)):
        super().__init__()
        self.conv_in = nn.Conv2d(3, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])
        prev_channel = block_out_channels[0]

        for i, channel in enumerate(block_out_channels):
            # 基础特征提取
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channel, channel, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
            # 插入空间注意力机制 (CAFE的核心创新)
            self.blocks.append(SpatialAttention())

            # 降采样层 (除了最后一层外)
            if i != len(block_out_channels) - 1:
                self.blocks.append(nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1))

            prev_channel = channel

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        x = self.conv_in(conditioning)
        for module in self.blocks:
            x = module(x)
        x = self.conv_out(x)
        return x
