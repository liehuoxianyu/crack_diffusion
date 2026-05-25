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
        # NOTE:
        # The shipped checkpoints in `outputs/exp_dt_patch512` and `outputs/exp_binary_patch512`
        # expect `controlnet_cond_embedding.blocks.{0..5}.{weight,bias}` to be *direct Conv2d modules*
        # (no nested Sequential/attention blocks under `blocks`).
        #
        # Their shapes imply the following conv channel path (for default block_out_channels=(16,32,96,256)):
        #   blocks: 16->16, 16->32(stride2), 32->32, 32->96(stride2), 96->96, 96->256(stride2)
        # which yields /8 spatial downsampling.
        self.act = nn.SiLU()
        self.conv_in = nn.Conv2d(3, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        # Layout: keep channels[i], then downsample+expand to channels[i+1], repeated.
        # For N stages, blocks length becomes 2*(N-1) (default N=4 => blocks length 6).
        for i in range(len(block_out_channels) - 1):
            c = block_out_channels[i]
            cn = block_out_channels[i + 1]
            if i == 0:
                # blocks[0] = c -> c
                self.blocks.append(nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1))
            else:
                # blocks[2*i] is the "keep" conv for the previous stage channels (added at previous iter)
                # so nothing to do here.
                pass

            # blocks[2*i+1] = c -> cn with stride2
            self.blocks.append(nn.Conv2d(c, cn, kernel_size=3, padding=1, stride=2))

            # blocks[2*i+2] (except after the last stage) = cn -> cn with stride1
            # This matches that the checkpoint has no extra 256->256 conv before conv_out.
            if i + 1 < len(block_out_channels) - 1:
                self.blocks.append(nn.Conv2d(cn, cn, kernel_size=3, padding=1, stride=1))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        x = self.conv_in(conditioning)
        for module in self.blocks:
            x = self.act(module(x))
        x = self.conv_out(x)
        return x
