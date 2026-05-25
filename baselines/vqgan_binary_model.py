import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, self.embedding_dim)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view_as(z_perm).permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, loss


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x):
        skip = self.block(x)
        return self.down(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


class ConditionalVQGAN(nn.Module):
    """Conditional VQ generator for binary-condition -> real-crack translation."""

    def __init__(self, in_channels=3, out_channels=3, hidden=64, z_channels=256, codebook_size=512):
        super().__init__()
        self.down1 = DownBlock(in_channels, hidden)
        self.down2 = DownBlock(hidden, hidden * 2)
        self.down3 = DownBlock(hidden * 2, hidden * 4)
        self.bottleneck = ConvBlock(hidden * 4, z_channels)
        self.quantizer = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=z_channels)
        self.up3 = UpBlock(z_channels, hidden * 4, hidden * 4)
        self.up2 = UpBlock(hidden * 4, hidden * 2, hidden * 2)
        self.up1 = UpBlock(hidden * 2, hidden, hidden)
        self.out = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, cond):
        x, skip1 = self.down1(cond)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        z = self.bottleneck(x)
        z_q, vq_loss = self.quantizer(z)
        x = self.up3(z_q, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        out = self.out(x)
        return out, vq_loss


class ConditionalPatchDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on the binary mask."""

    def __init__(self, cond_channels=3, image_channels=3, hidden=64, n_layers=3):
        super().__init__()
        in_ch = cond_channels + image_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, hidden, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        mult = 1
        for i in range(1, n_layers + 1):
            prev = mult
            mult = min(2**i, 8)
            layers.extend(
                [
                    nn.Conv2d(hidden * prev, hidden * mult, 4, stride=2 if i < n_layers else 1, padding=1),
                    nn.InstanceNorm2d(hidden * mult),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden * mult, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, cond, image):
        return self.net(torch.cat([cond, image], dim=1))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        device_obj = torch.device(device)
        features = vgg16(weights=weights).features[:16].eval().to(device_obj)
        for param in features.parameters():
            param.requires_grad = False
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device_obj).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device_obj).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred + 1.0) * 0.5
        target = (target + 1.0) * 0.5
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.features(pred), self.features(target))


def hinge_d_loss(real_logits, fake_logits):
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()
