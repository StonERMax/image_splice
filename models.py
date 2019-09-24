import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import skimage
import cv2


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class Corr(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk

        self.alpha = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))

    def forward(self, xp, xq):
        b, c, h1, w1 = xp.shape
        _, _, h2, w2 = xq.shape

        xpn = F.normalize(xp, p=2, dim=-3)
        xqn = F.normalize(xq, p=2, dim=-3)

        x_aff = torch.matmul(
            xpn.permute(0, 2, 3, 1).view(b, -1, c), xqn.view(b, c, -1)
        )  # h1 * w1, h2 * w2

        x_c = F.softmax(x_aff * self.alpha, dim=-1) * F.softmax(
            x_aff * self.alpha, dim=-2
        )
        x_c = x_c.reshape(b, h1, w1, h2, w2)
        xc_o_q = x_c.view(b, h1 * w1, h2, w2)
        valq = get_topk(xc_o_q, k=self.topk, dim=-3)

        xc_o_p = xc_o_q.permute(0, 2, 3, 1).view(b, h2 * w2, h1, w1)
        valp = get_topk(xc_o_p, k=self.topk, dim=-3)

        x_soft_p = xc_o_p  # / (xc_o_p.sum(dim=-3, keepdim=True) + 1e-8)
        x_soft_q = xc_o_q  # / (xc_o_q.sum(dim=-3, keepdim=True) + 1e-8)

        return valp, valq, x_soft_p, x_soft_q


def std_mean(x):
    return (x - x.mean(dim=-3, keepdim=True)) / (
        1e-8 + x.std(dim=-3, keepdim=True)
    )


def plot_mat(x, name, cmap="jet", size=(120, 120)):
    x = x.data.cpu().numpy()
    xr = cv2.resize(x, size, interpolation=cv2.INTER_LINEAR)

    plt.matshow(xr, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        f"{name}.png",
        bbox_inches="tight",
        pad_inches=0,
        frameon=False,
        transparent=True,
    )


def plot_plt(x, name="1", size=(240, 240), cmap="jet"):
    if x.shape[0] == 2:
        x = torch.cat((x, x[[0]]), dim=-3)
    x = F.interpolate(x.unsqueeze(0), size=size, mode="bilinear").squeeze(0)

    def fn(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    xn = fn(x)
    xn = x.permute(1, 2, 0).data.cpu().numpy()[..., 0]
    plt.imsave(f"{name}_m.png", xn, cmap=cmap)


def plot(x, name="1", size=(240, 240)):
    if x.shape[0] == 2:
        x = torch.cat((x, x[[0]]), dim=-3)
    x = F.interpolate(x.unsqueeze(0), size=size, mode="bilinear").squeeze(0)

    def fn(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    torchvision.utils.save_image(fn(x), f"{name}.png")


class DetectionBranch(nn.Module):
    def __init__(self, in_cat=896):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_cat, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.lin = nn.Sequential(
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 1)
        )

        self.apply(weights_init_normal)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)

        x_cat = torch.cat(
            (
                F.adaptive_avg_pool2d(x, (1, 1)),
                F.adaptive_max_pool2d(x, (1, 1)),
            ),
            dim=-3,
        )

        x_cat = x_cat.reshape(b, 128)
        y = self.lin(x_cat)
        return y


class GCN(nn.Module):
    def __init__(self, in_feat=256):
        super().__init__()
        self.in_feat = in_feat

        self.conv = nn.Sequential(
            nn.Conv2d(in_feat, in_feat, 3, padding=1),
            nn.BatchNorm2d(in_feat),
            nn.ReLU(),
        )
        self.conv.apply(weights_init_normal)

    def forward(self, x, ind):
        b, c, h2, w2 = x.shape
        _, _, h1, w1 = ind.shape

        x = x.reshape(b, c, -1).permute(0, 2, 1)  # b, h2w2, c
        ind = ind.reshape(b, h2 * w2, h1 * w1).permute(0, 2, 1)
        # b, h1w1, h2w2

        D_mhalf = torch.diag_embed(
            torch.sqrt(1.0 / (torch.sum(ind, dim=-1) + 1e-8))
        )
        # D_mhalf = torch.sqrt(torch.inverse(D))  # b, h1w1, h1w1

        eye = torch.eye(ind.shape[-1], dtype=ind.dtype)
        ind_with_e = ind + eye.view(1, *eye.shape).to(ind.device)
        A = torch.bmm(D_mhalf, torch.bmm(ind_with_e, D_mhalf))  # b, h1w1, h2w2

        out_inter = torch.bmm(A, x).permute(0, 2, 1).reshape(b, c, h1, w1)
        out = self.conv(out_inter)
        return out

        # b, c, h2, w2 = x.shape
        # _, _, h1, w1 = ind.shape

        # x = x.reshape(b, c, -1)
        # ind = ind.reshape(b, h2 * w2, h1 * w1)
        # out_inter = torch.bmm(x, ind).reshape(b, c, h1, w1)
        # out = self.conv(out_inter)
        # return out


class DOAModel(nn.Module):
    def __init__(self, out_channel=1, topk=20, hw=(40, 40)):
        super().__init__()
        self.hw = hw
        self.topk = topk

        self.encoder = Extractor_VGG19()

        self.corrLayer = Corr(topk=topk)

        in_cat = 896

        self.val_conv = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self.aspp_mask = models.segmentation.deeplabv3.ASPP(
            in_channels=in_cat, atrous_rates=[12, 24, 36]
        )

        self.aspp_forge = models.segmentation.deeplabv3.ASPP(
            in_channels=in_cat, atrous_rates=[6, 12, 24]
        )

        self.head_mask_p = nn.Sequential(
            nn.Conv2d(4 * 256, 2 * 256, 1),
            nn.BatchNorm2d(2 * 256),
            nn.ReLU(),
            nn.Conv2d(2 * 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channel, 1),
        )

        self.head_mask_q = nn.Sequential(
            nn.Conv2d(4 * 256, 2 * 256, 1),
            nn.BatchNorm2d(2 * 256),
            nn.ReLU(),
            nn.Conv2d(2 * 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channel, 1),
        )

        self.gcn_mask = GCN()
        self.gcn_forge = GCN()

        # detection branch
        self.detection = DetectionBranch(4 * 256)
        self.head_mask_p.apply(weights_init_normal)
        self.head_mask_q.apply(weights_init_normal)

        self.val_conv.apply(weights_init_normal)

    def forward(self, xq, xp):
        input1, input2 = xq, xp
        b, c, h, w = xp.shape
        xp_feat = self.encoder(xp, out_size=self.hw)
        xq_feat = self.encoder(xq, out_size=self.hw)

        valp, valq, indp, indq = self.corrLayer(xp_feat, xq_feat)

        # attention weight
        val_conv_p = self.val_conv(valp)
        val_conv_q = self.val_conv(valq)

        #### Mask part : M  ####
        xp_as1 = self.aspp_mask(xp_feat) * val_conv_p
        xq_as1 = self.aspp_mask(xq_feat) * val_conv_q

        xp_as2 = self.aspp_forge(xp_feat) * val_conv_p
        xq_as2 = self.aspp_forge(xq_feat) * val_conv_q

        # Corrensponding mask and forge
        xp_as1_nl = self.gcn_mask(xq_as1, indp)
        xq_as1_nl = self.gcn_mask(xp_as1, indq)

        xp_as2_nl = self.gcn_forge(xq_as2, indp)
        xq_as2_nl = self.gcn_forge(xp_as2, indq)

        # Final Mask
        x_cat_p = torch.cat((xp_as1, xp_as1_nl, xp_as2, xp_as2_nl), dim=-3)
        outp = self.head_mask_p(x_cat_p)

        x_cat_q = torch.cat((xq_as1, xq_as1_nl, xq_as2, xq_as2_nl), dim=-3)
        outq = self.head_mask_q(x_cat_q)

        # final detection
        out_det = self.detection(
            torch.cat((xp_as2, xq_as1, xp_as1_nl, xq_as2_nl), dim=-3)
        )

        outp = F.interpolate(
            outp, size=(h, w), mode="bilinear", align_corners=True
        )
        outq = F.interpolate(
            outq, size=(h, w), mode="bilinear", align_corners=True
        )

        return outq, outp, out_det

    def non_local(self, x, ind):
        b, c, h2, w2 = x.shape
        _, _, h1, w1 = ind.shape

        x = x.reshape(b, c, -1)
        ind = ind.reshape(b, h2 * w2, h1 * w1)
        out = torch.bmm(x, ind).reshape(b, c, h1, w1)
        return out

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        self.apply(fn)

    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(
                    m[1], nn.BatchNorm2d
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [
            self.aspp_mask,
            self.aspp_forge,
            self.head_mask_p,
            self.head_mask_q,
            self.val_conv,
            self.corrLayer,
            # self.detection,
        ]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(
                    m[1], nn.BatchNorm2d
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Extractor_VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_temp = torchvision.models.vgg19(pretrained=True).features

        self.layer1 = nn.Sequential(cnn_temp[:10])  # stride 4
        self.layer2 = nn.Sequential(cnn_temp[10:19])  # s 8
        self.layer3 = nn.Sequential(cnn_temp[19:28])  # s 16

    def forward(self, x, out_size=(40, 40)):
        x1 = self.layer1(x)
        x1_u = F.interpolate(
            x1, size=out_size, mode="bilinear", align_corners=True
        )

        x2 = self.layer2(x1)
        x2_u = F.interpolate(
            x2, size=out_size, mode="bilinear", align_corners=True
        )

        x3 = self.layer3(x2)
        x3_u = F.interpolate(
            x3, size=out_size, mode="bilinear", align_corners=True
        )

        x_all = torch.cat([x1_u, x2_u, x3_u], dim=-3)

        return x_all


class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)
            ]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
            # nn.LeakyReLU(0.2, inplace=True)
        )

        # self.final = nn.Linear(10*10*1, 1)

        self.apply(weights_init_normal)

    def forward(self, img_A, Y):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, Y), 1)

        x = self.model(img_input)
        return x
