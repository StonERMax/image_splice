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
import math


# def get_topk(x, k=10, dim=-3):
#     b, c, h1, w1 = x.shape
#     val, _ = torch.topk(x, k=k, dim=dim)
#     return val
#     # xr = x.permute(0, 3, 4, 1, 2).view(b, -1, h2, w2)  # --> b, h1*w1, h2, w2
#     # val = F.adaptive_max_pool2d(xr, int(np.sqrt(k)))  # --> b, h1*w1, k, k
#     # val = val.view(b, h1, w1, -1).permute(0, 3, 1, 2)  # --> b, k*k, h1, w1
#     # return val


def _zero_window(x_in, h, w, rat_s=0.05):
    sigma = h * rat_s, w * rat_s
    # c = h * w
    b, c, h2, w2 = x_in.shape
    ind_r = torch.arange(h2).float()
    ind_c = torch.arange(w2).float()
    ind_r = ind_r.view(1, 1, -1, 1).expand_as(x_in)
    ind_c = ind_c.view(1, 1, 1, -1).expand_as(x_in)

    # center
    c_indices = torch.from_numpy(np.indices((h, w))).float()
    c_ind_r = c_indices[0].reshape(-1)
    c_ind_c = c_indices[1].reshape(-1)

    cent_r = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in)
    cent_c = c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)

    def fn_gauss(x, u, s):
        return torch.exp(-(x - u) ** 2 / (2 * s ** 2))

    gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
    gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
    out_g = 1 - gaus_r * gaus_c

    out = out_g.to(x_in.device) * x_in
    return out


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class Conv4d(nn.Module):
    def __init__(self, out_channel=10):
        super().__init__()
        self.conv_patch = nn.Conv3d(3, out_channel, 3, padding=1)
        self.conv_patch.apply(weights_init_normal)

    def forward(self, x):
        b, h1, w1, h2, w2 = x.shape
        x_c_pad = F.pad(x, [0, 0, 0, 0, 0, 0, 1, 1])
        x_c_cat = torch.cat((x_c_pad[:, 0:-2], x_c_pad[:, 1:-1], x_c_pad[:, 2:]), 1)
        x_c_cat = (
            x_c_cat.view(b, 3, h1, w1, h2, w2)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(b * h1, 3, w1, h2, w2)
        )

        # (b*h1, 3, w1, h2, w2) --> (b*h1, co, w1, h2, w2)
        x_conv = self.conv_patch(x_c_cat)
        x_max = torch.mean(x_conv, 1)  # --> b*h1, w1, h2, w2
        x_out = x_max.reshape(b, h1, w1, h2, w2)
        return x_out


class Corr(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk
        self.alpha = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        # self.patch_conv = Conv4d(out_channel=10)

    def forward(self, xp, xq):
        b, c, h1, w1 = xp.shape
        _, _, h2, w2 = xq.shape

        xpn = F.normalize(xp, p=2, dim=-3)
        xqn = F.normalize(xq, p=2, dim=-3)

        x_aff_o = torch.matmul(
            xpn.permute(0, 2, 3, 1).view(b, -1, c), xqn.view(b, c, -1)
        )  # h1 * w1, h2 * w2

        x_aff = _zero_window(x_aff_o.view(b, h1 * w1, h2, w2), h1, w1).reshape(
            b, h1 * w1, h2 * w2
        )
        x_c = F.softmax(x_aff * self.alpha, dim=-1) * F.softmax(
            x_aff * self.alpha, dim=-2
        )

        x_c = x_c.view(b, h1, w1, h2, w2)
        # apply patch-conv
        # x_c = self.patch_conv(x_c)

        x_c = x_c.reshape(b, h1, w1, h2, w2)
        xc_o_q = x_c.view(b, h1 * w1, h2, w2)
        valq = get_topk(xc_o_q, k=self.topk, dim=-3)

        xc_o_p = xc_o_q.permute(0, 2, 3, 1).view(b, h2 * w2, h1, w1)
        valp = get_topk(xc_o_p, k=self.topk, dim=-3)

        x_soft_p = xc_o_p  # / (xc_o_p.sum(dim=-3, keepdim=True) + 1e-8)
        x_soft_q = xc_o_q  # / (xc_o_q.sum(dim=-3, keepdim=True) + 1e-8)

        return valp, valq, x_soft_p, x_soft_q


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


def plot(x, name="1", size=(240, 240)):
    if x.shape[0] == 2:
        x = torch.cat((x, x[[0]]), dim=-3)
    if size is not None:
        x = F.interpolate(x.unsqueeze(0), size=size, mode="bilinear").squeeze(0)

    def fn(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    torchvision.utils.save_image(fn(x), f"{name}.png")


class GCN(nn.Module):
    def __init__(self, in_feat=256, out_feat=256):
        super().__init__()
        self.in_feat = in_feat

        self.conv = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, 1), nn.BatchNorm2d(out_feat), nn.ReLU()
        )
        self.conv.apply(weights_init_normal)

    def forward(self, x, ind):
        b, c, h2, w2 = x.shape
        _, _, h1, w1 = ind.shape

        x = x.reshape(b, c, -1).permute(0, 2, 1)  # b, h2w2, c
        ind = ind.reshape(b, h2 * w2, h1 * w1).permute(0, 2, 1)
        # b, h1w1, h2w2
        D_mhalf = torch.diag_embed(torch.sqrt(1.0 / (torch.sum(ind, dim=-1) + 1e-8)))

        eye = torch.eye(ind.shape[-1], dtype=ind.dtype)
        ind_with_e = ind + eye.view(1, *eye.shape).to(ind.device)
        A = torch.bmm(D_mhalf, torch.bmm(ind_with_e, D_mhalf))  # b, h1w1, h2w2

        out_inter = torch.bmm(A, x).permute(0, 2, 1).reshape(b, c, h1, w1)
        out = self.conv(out_inter)
        return out


class DOAModel(nn.Module):
    def __init__(self, out_channel=1, topk=20, hw=(40, 40)):
        super().__init__()
        self.hw = hw
        self.topk = topk

        # self.encoder_p = Extractor_VGG19()
        self.encoder_q = Extractor_VGG19()
        # self.encoder_p = self.encoder_q

        self.corrLayer = Corr(topk=topk)

        in_cat = 896

        # self.in_conv_p = nn.Sequential(
        #     nn.Conv2d(in_cat, 256, 3, padding=1),
        #     nn.BatchNorm2d(256), nn.ReLU()
        # )
        self.in_conv_q = nn.Sequential(
            nn.Conv2d(in_cat, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        # self.val_conv_p = nn.Sequential(
        #     nn.Conv2d(topk, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, 3, padding=1),
        #     nn.Conv2d(16, 1, 1),
        # )

        self.val_conv_q = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 1, 1),
        )

        self.conv_pyr = nn.Sequential(
            nn.Conv2d(3*256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # self.val_conv_p.apply(weights_init_normal)
        self.val_conv_q.apply(weights_init_normal)
        self.conv_pyr.apply(weights_init_normal)
        # self.in_conv_p.apply(weights_init_normal)
        self.in_conv_q.apply(weights_init_normal)

    def forward(self, xq, xp):
        input1, input2 = xq, xp
        b, c, h, w = xp.shape
        # xp_feat = self.in_conv_q(self.encoder_q(xp, out_size=self.hw))
        xq_feat = self.in_conv_q(self.encoder_q(xq, out_size=self.hw))

        # xp_spp = self.conv_pyr(self.spatial_pyramid_pool(xp_feat, [4, 2, 1]))
        xq_spp = self.conv_pyr(self.spatial_pyramid_pool(xq_feat, [4, 2, 1]))

        valp, valq, _, _ = self.corrLayer(xq_spp, xq_spp)
        # outp = self.val_conv_p(valp)
        outq = self.val_conv_q(valq)

        # outp = F.interpolate(outp, size=(h, w), mode="bilinear", align_corners=True)
        outq = F.interpolate(outq, size=(h, w), mode="bilinear", align_corners=True)

        return outq  #, outp

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm") != -1:
                m.eval()

        self.apply(fn)

    def spatial_pyramid_pool(self, previous_conv, out_pool_size):
        """
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features 
        size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        previous_conv_size = previous_conv.shape[-2:]
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            # h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            # w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            x = F.adaptive_max_pool2d(previous_conv, (h_wid, w_wid))
            if i == 0:
                spp = F.interpolate(x, previous_conv_size)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, F.interpolate(x, previous_conv_size)), -3)
        return spp


def weights_init_normal(m):
    try:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1 or classname.find("Linear") != -1:
            torch.nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    except AttributeError:
        return


class Extractor_VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_temp = torchvision.models.vgg19(pretrained=True).features

        self.layer1 = nn.Sequential(cnn_temp[:10])  # stride 4
        self.layer2 = nn.Sequential(cnn_temp[10:19])  # s 8
        self.layer3 = nn.Sequential(cnn_temp[19:28])  # s 16

    def forward(self, x, out_size=(40, 40)):
        x1 = self.layer1(x)
        x1_u = F.interpolate(x1, size=out_size, mode="bilinear", align_corners=True)

        x2 = self.layer2(x1)
        x2_u = F.interpolate(x2, size=out_size, mode="bilinear", align_corners=True)

        x3 = self.layer3(x2)
        x3_u = F.interpolate(x3, size=out_size, mode="bilinear", align_corners=True)

        x_all = torch.cat([x1_u, x2_u, x3_u], dim=-3)

        return x_all
