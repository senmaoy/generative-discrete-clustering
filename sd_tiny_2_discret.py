from __future__ import print_function
import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from sinkhornloss import SinkhornLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sinkhorn_loss_fn = SinkhornLoss(eps=0.1, max_iter=30)


# ============================================================
# 工具函数
# ============================================================
def sinusoidal_embedding(n, d):
    """
    生成正弦余弦位置编码
    Args:
        n: 序列长度
        d: 嵌入维度
    Returns:
        形状为 (n, d) 的位置编码张量
    """
    embedding = torch.zeros(n, d)
    for i in range(n):
        for j in range(0, d, 2):
            # 正弦部分
            embedding[i, j] = torch.sin(torch.tensor(i / (10000 ** (2 * j / d))))
            # 余弦部分（确保不越界）
            if j + 1 < d:
                embedding[i, j+1] = torch.cos(torch.tensor(i / (10000 ** (2 * j / d))))
    return embedding

def linear_spatial_embedding(shape_bchw=(1, 256, 64, 64)):
    """
    构建初始特征图，结合正弦嵌入和线性映射
    Args:
        shape_bchw: 目标形状 (batch, channels, height, width)
    Returns:
        初始特征图张量
    """
    b, c, h, w = shape_bchw
    
    # 生成正弦嵌入
    emb_sin = sinusoidal_embedding(h * w, c)  # (h*w, c)
    emb_sin = emb_sin.reshape(h, w, c).permute(2, 0, 1)  # (c, h, w)
    
    # 生成线性映射并添加到最后4个通道
    if h > 1 and w > 1:
        emb_linear = torch.linspace(-1, 1, h * w).reshape(h, w)
        # 确保有至少4个通道来添加线性映射
        if c >= 4:
            linear_channels = min(4, c)
            for k in range(linear_channels):
                emb_sin[-(k+1)] += torch.rot90(emb_linear, k)
    
    # 扩展到batch大小
    feature_init = emb_sin.unsqueeze(0).repeat(b, 1, 1, 1)  # (b, c, h, w)
    return feature_init
# def sinusoidal_embedding(n, d):
#     embedding = torch.zeros(n, d)
#     for i in range(n):
#         for j in range(0, d, 2):
#             embedding[i, j] = torch.sin(torch.tensor(i / (10000 ** (2 * j / d))))
#             if j + 1 < d:
#                 embedding[i, j + 1] = torch.cos(torch.tensor(i / (10000 ** (2 * j / d))))
#     return embedding


# def linear_spatial_embedding(shape, device="cpu"):
#     size = 1
#     for s in shape:
#         size *= s
#     spatial_embedding = torch.linspace(-1, 1, size, device=device).reshape(*shape)
#     if len(shape) == 3:
#         if shape[-1] == shape[-2]:
#             spatial_embedding = torch.cat(
#                 [torch.rot90(e, i)[None] for i, e in enumerate(spatial_embedding)])
#         else:
#             spatial_embedding = torch.cat([
#                 torch.flip(e, (([0] if i % 4 in (1, 3) else []) +
#                                ([1] if i % 4 in (2, 3) else [])))[None]
#                 for i, e in enumerate(spatial_embedding)])
#     return spatial_embedding


def build_init_feature(shape_bchw=(1, 256, 64, 64)):
    b, c, h, w = shape_bchw
    emb_sin = sinusoidal_embedding(h * w, c).reshape(h, w, c).permute(2, 0, 1)
    if h > 1 and w > 1 and c >= 4:
        emb_linear = torch.linspace(-1, 1, h * w).reshape(h, w)
        for k in range(min(4, c)):
            emb_sin[-(k + 1)] += torch.rot90(emb_linear, k)
    return emb_sin.unsqueeze(0).repeat(b, 1, 1, 1)


# ============================================================
# U-Net 基础模块
# ============================================================

class MyConv(nn.Module):
    def __init__(self, in_c, out_c, size, kernel_size=3, stride=1, padding=1,
                 activation=None, normalize=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_c, affine=True) if normalize else nn.Identity()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(out)
        out = self.activation(out)
        return out


def MyTinyBlock(size, in_c, out_c):
    return nn.Sequential(
        MyConv(in_c, out_c, size),
        MyConv(out_c, out_c, size),
        MyConv(out_c, out_c, size),
    )


def MyTinyUp(size, in_c):
    return nn.Sequential(
        MyConv(in_c, in_c // 2, size),
        MyConv(in_c // 2, in_c // 4, size),
        MyConv(in_c // 4, in_c // 4, size),
    )


# ============================================================
# ✅ 新 DDN（修复通道不匹配）
# ============================================================

class DDN(nn.Module):
    def __init__(self, nc=3, ngf=64, K=8, size=32, class_n=100):
        super().__init__()
        self.K = K
        self.nc = nc
        self.size = size

        basec = ngf // 4
        self.last_c = nc * K          # 输出通道 = nc * K

        # 类别嵌入
        self.class_embed = nn.Embedding(class_n, ngf * 2)
        self.class_embed.weight.data = sinusoidal_embedding(class_n, ngf * 2)
        self.class_embed.requires_grad_(False)

        # TE 映射
        self.te1 = self._make_te(ngf * 2, 1 * basec)
        self.te2 = self._make_te(ngf * 2, 1 * basec)
        self.te3 = self._make_te(ngf * 2, 2 * basec)
        self.te_mid = self._make_te(ngf * 2, 4 * basec)
        self.te4 = self._make_te(ngf * 2, 8 * basec)
        self.te5 = self._make_te(ngf * 2, 4 * basec)
        self.te_out = self._make_te(ngf * 2, 2 * basec)

        # ✅ 输入投影：nc → 1*basec
        self.input_proj = nn.Conv2d(nc, 1 * basec, kernel_size=1)

        # ✅ feat_last 投影：last_c(nc*K) → 1*basec（和 input_proj 对称）
        self.feat_last_proj = nn.Conv2d(self.last_c, 1 * basec, kernel_size=1)

        # Encoder
        self.b1 = MyTinyBlock(size, 1 * basec, 1 * basec)
        self.down1 = nn.Conv2d(1 * basec, 1 * basec, 4, 2, 1)
        self.b2 = MyTinyBlock(size // 2, 1 * basec, 2 * basec)
        self.down2 = nn.Conv2d(2 * basec, 2 * basec, 4, 2, 1)
        self.b3 = MyTinyBlock(size // 4, 2 * basec, 4 * basec)
        self.down3 = nn.Conv2d(4 * basec, 4 * basec, 4, 2, 1)

        # Bottleneck
        self.b_mid = nn.Sequential(
            MyConv(4 * basec, 2 * basec, size // 8),
            MyConv(2 * basec, 2 * basec, size // 8),
            MyConv(2 * basec, 4 * basec, size // 8),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(4 * basec, 4 * basec, 4, 2, 1)
        self.b4 = MyTinyUp(size // 4, 8 * basec)
        self.up2 = nn.ConvTranspose2d(2 * basec, 2 * basec, 4, 2, 1)
        self.b5 = MyTinyUp(size // 2, 4 * basec)
        self.up3 = nn.ConvTranspose2d(1 * basec, 1 * basec, 4, 2, 1)

        # 输出头 → nc*K 通道
        self.b_out = MyTinyBlock(size, 2 * basec, self.last_c)
        self.out_last = nn.Tanh()
    def forward(self, x, t, feat_last=None):
        device = x.device
        n = x.shape[0]

        if t.dim() == 1:
            t = self.class_embed(t.to(device))

        # 初始化 feat_last：(B, nc*K, H, W)
        if feat_last is None:
            feat_last = linear_spatial_embedding(
                (self.last_c, self.size, self.size), device=device
            ).unsqueeze(0).repeat(n, 1, 1, 1)

        # ✅ 两个投影，完全对称
        x_proj = self.input_proj(x)                              # (B, 1*basec, H, W)
        fl_proj = self.feat_last_proj(feat_last)                  # (B, 1*basec, H, W)

        # Encoder
        te1 = self.te1(t).reshape(n, -1, 1, 1)
        out1 = self.b1(x_proj + te1)
        out1 = out1 + self.b1(fl_proj)   # ✅ 复用 b1 结构处理投影后的 feat_last

        te2 = self.te2(t).reshape(n, -1, 1, 1)
        out2 = self.b2(self.down1(out1) + te2)

        te3 = self.te3(t).reshape(n, -1, 1, 1)
        out3 = self.b3(self.down2(out2) + te3)

        # Bottleneck
        te_mid = self.te_mid(t).reshape(n, -1, 1, 1)
        out_mid = self.b_mid(self.down3(out3) + te_mid)

        # Decoder
        te4 = self.te4(t).reshape(n, -1, 1, 1)
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
        out4 = self.b4(out4 + te4)

        te5 = self.te5(t).reshape(n, -1, 1, 1)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)
        out5 = self.b5(out5 + te5)

        te_out = self.te_out(t).reshape(n, -1, 1, 1)
        out = torch.cat((out1, self.up3(out5)), dim=1)
        feat_last_new = self.b_out(out + te_out)                  # (B, nc*K, H, W)
        feat_last_new = self.out_last(feat_last_new)
        return feat_last_new

    @staticmethod
    def _make_te(dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )


# ============================================================
# ✅ Generator（无 Refiner，Sinkhorn 3D 安全）
# ============================================================

class Generator(nn.Module):
    def __init__(self, nc=3, ngf=64, K=8, num_ddns=3, image_size=32):
        super().__init__()
        self.K = K
        self.nc = nc
        self.num_ddns = num_ddns

        self.ddns = nn.ModuleList([
            DDN(nc=nc, ngf=ngf, K=K, size=image_size, class_n=num_ddns)
            for _ in range(num_ddns)
        ])

    def forward(self, input, prev_branch, target=None):
        batch_size = input.shape[0]
        device = input.device

        all_outputs = []
        all_components = []
        all_weights = []

        feat_last = prev_branch  # (B, nc*K, H, W)

        for i, ddn in enumerate(self.ddns):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            feat_last = ddn(input, t, feat_last=feat_last)

            # reshape 成 K 个分量
            components = feat_last.reshape(batch_size, self.K, self.nc, *input.shape[2:])

            # 混合权重
            if target is not None:
                target_resized = F.interpolate(
                    target, size=input.shape[2:], mode='bilinear', align_corners=False)
                dist = torch.sum(
                    (components - target_resized.unsqueeze(1)) ** 2, dim=[2, 3, 4]
                ) + 1e-8  
                mixing_weights = F.gumbel_softmax(-dist, dim=1)
                mixing_w = mixing_weights.view(batch_size, self.K, 1, 1, 1)
                x = torch.sum(mixing_w * components, dim=1)
            else:
                k = torch.randint(0, self.K, (batch_size,))  # 形状: (batch_size,)
                
                # 创建批次索引 [6,9](@ref)
                batch_indices = torch.arange(batch_size)  # 形状: (batch_size,)
                
                # 使用高级索引提取指定分量
                x = components[batch_indices, k]  # 形状: (batch_size, nc, H, W)
                mixing_weights = k#torch.ones(batch_size, self.K, device=device) / self.K

            

            all_outputs.append(x)
            all_weights.append(mixing_weights)

            # ✅ Sinkhorn：始终保持 (B, K, D) 3D
            if target is not None:
                comp_3d = components.reshape(1,batch_size* self.K, -1)
                tgt_3d = target_resized.reshape(1,batch_size, -1)
                loss = sinkhorn_loss_fn(comp_3d, tgt_3d)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=False)

            all_components.append(loss)

        if target is not None:
            return all_outputs, all_components, all_weights
        else:
            return x, all_outputs, all_weights


# ============================================================
# 训练
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--dataroot', default='./dataset')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batchSize', type=int, default=1024)
    parser.add_argument('--imageSize', type=int, default=32)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--niter', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--cuda', default='1111')
    parser.add_argument('--outf', default='.')
    parser.add_argument('--manualSeed', type=int, default=None)
    opt = parser.parse_args()

    print("配置参数:", opt)
    K = 10
    num_ddns = 2

    os.makedirs(f'{opt.outf}/images', exist_ok=True)
    os.makedirs(f'{opt.outf}/checkpoint', exist_ok=True)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    device = torch.device("cuda:0" if opt.cuda and torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    nc = 1 if opt.dataset == 'mnist' else 3
    transform = transforms.Compose([
        transforms.Resize(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * nc, (0.5,) * nc),
    ])
    if opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True, transform=transform)
    else:
        dataset = dset.CIFAR10(root=opt.dataroot, download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.workers, drop_last=True)

    netG = Generator(nc=nc, ngf=opt.ngf, K=K,
                     num_ddns=num_ddns, image_size=opt.imageSize).to(device)
    optimizer = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    print("参数量:", sum(p.numel() for p in netG.parameters()))

    # ✅ noise0 通道改为 nc*K，和 DDN 的 feat_last 输出完全一致
    noise0 = build_init_feature((opt.batchSize, nc * K, opt.imageSize, opt.imageSize)).to(device)
    noise2 = build_init_feature((opt.batchSize, nc, opt.imageSize, opt.imageSize)).to(device)

    print("训练开始...")
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader):
            real_images = data[0].to(device)
            bs = real_images.shape[0]

            netG.zero_grad()
            all_outputs, all_components, all_weights = netG(
                noise2[:bs], noise0[:bs], target=real_images
            )

            total_loss = sum(all_components)
            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'[{epoch}/{opt.niter}][{i}] Loss: {total_loss.item():.4f}')

        with torch.no_grad():
            fake, _, _ = netG(noise2[:32], noise0[:32])

            vutils.save_image( fake, 
                '%s/images/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                nrow=4, normalize=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': netG.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{opt.outf}/checkpoint/netG_epoch_{epoch:03d}.pth')

    print("训练完成!")


if __name__ == '__main__':
    main()