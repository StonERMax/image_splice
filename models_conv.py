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
from models import plot, get_topk
from models_custom import Extractor_custom



class ZeroWindow:
    def __init__(self):
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1):
        sigma = h * rat_s, w * rat_s
        # c = h * w
        b, c, h2, w2 = x_in.shape
        key = str(x_in.shape) + str(rat_s)
        if key not in self.store:
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
            out_g = out_g.to(x_in.device)
            out_g.requires_grad = False
            self.store[key] = out_g
        else:
            out_g = self.store[key]
        out = out_g * x_in
        return out


class CorrCMFD(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk
        self.patch_4d = nn.Sequential(
            Conv4d(1, 16, (3, 3, 3, 3), padding=1),
            Conv4d(16, 1, (3, 3, 3, 3), padding=1)
        )

        self.zero_window = ZeroWindow()

        self.apply(weights_init_normal)

    def forward(self, x):
        b, c, h1, w1 = x.shape
        h2, w2 = h1, w1

        xn = F.normalize(x, p=2, dim=-3)

        x_aff_o = torch.matmul(
            xn.permute(0, 2, 3, 1).view(b, -1, c), xn.view(b, c, -1)
        )  # h1 * w1, h2 * w2

        x_aff = self.zero_window(x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05).reshape(
            b, h1 * w1, h2 * w2
        )

        x_fc = x_aff.reshape(b, 1, h1, w1, h2, w2)
        x_c = self.patch_4d(x_fc).reshape(b, h1, w1, h2, w2)

        xc_o_q = x_c.reshape(b, h1 * w1, h2, w2)
        valq = get_topk(xc_o_q, k=self.topk, dim=-3)

        return valq




class Corr(nn.Module):
    def __init__(self, topk=3):
        super().__init__()
        self.topk = topk
        self.patch_4d = nn.Sequential(
            Conv4d(1, 16, (3, 3, 3, 3), padding=1),
            Conv4d(16, 1, (3, 3, 3, 3), padding=1)
        )

        self.apply(weights_init_normal)

    def forward(self, xp, xq):
        b, c, h1, w1 = xp.shape
        _, _, h2, w2 = xq.shape

        xpn = F.normalize(xp, p=2, dim=-3)
        xqn = F.normalize(xq, p=2, dim=-3)

        x_aff = torch.matmul(
            xpn.permute(0, 2, 3, 1).view(b, -1, c), xqn.view(b, c, -1)
        )  # h1 * w1, h2 * w2

        x_fc = x_aff.reshape(b, 1, h1, w1, h2, w2)
        x_c = self.patch_4d(x_fc).reshape(b, h1, w1, h2, w2)

        xc_o_q = x_c.reshape(b, h1 * w1, h2, w2)
        valq = get_topk(xc_o_q, k=self.topk, dim=-3)

        xc_o_p = xc_o_q.permute(0, 2, 3, 1).reshape(b, h2 * w2, h1, w1)
        valp = get_topk(xc_o_p, k=self.topk, dim=-3)

        return valp, valq


class CustomModel(nn.Module):
    def __init__(self, out_channel=1, topk=20, hw=(40, 40)):
        super().__init__()
        self.hw = hw
        self.topk = topk

        self.encoder = Extractor_custom()
        self.corrLayer = Corr(topk=topk)

        self.val_conv = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channel, 3, padding=1)
        )

        self.val_conv.apply(weights_init_normal)


    def forward(self, xq, xp):
        input1, input2 = xq, xp
        b, c, h, w = xp.shape
        xpq_feat = self.encoder(torch.cat((xp, xq), 0))
        xp_feat, xq_feat = xpq_feat[:b], xpq_feat[b:]

        valp, valq = self.corrLayer(xp_feat, xq_feat)  # b x h1 x w1 x h2 x w2
  
        # attention weight
        val_conv_pq = self.val_conv(torch.cat((valp, valq), 0))
        outpq = F.interpolate(
            val_conv_pq, size=(h, w), mode="bilinear"
        )
        outp, outq = outpq[:b], outpq[b:]

        return outq, outp


class CustomModelCMFDSim(nn.Module):
    def __init__(self, out_channel=1, topk=100, hw=(40, 40)):
        super().__init__()
        self.hw = hw
        self.topk = topk

        self.encoder = Extractor_custom()
        self.corrLayer = CorrCMFD(topk=topk)

        self.val_conv = nn.Sequential(
            nn.Conv2d(topk, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channel, 1)
        )
        self.val_conv.apply(weights_init_normal)


    def forward(self, xq, xp):
        input1, input2 = xq, xp
        b, c, h, w = xp.shape
        xp_feat = self.encoder(xp)
        val = self.corrLayer(xp_feat)  # b x h1 x w1 x h2 x w2
        # attention weight
        val_conv_pq = self.val_conv(val)
        outpq = F.interpolate(
            val_conv_pq, size=(h, w), mode="bilinear"
        )
        return outpq



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

from typing import Tuple, Callable

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int, int],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):

        super().__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------

        assert len(kernel_size) == 4, \
            '4D kernel size expected!'
        assert stride == 1, \
            'Strides other than 1 not yet implemented!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.bias = bias

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------

        # Shortcut for kernel dimensions
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(l_k):

            # Initialize a Conv3D layer
            conv3d_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=(d_k, h_k, w_k),
                                           padding=self.padding)

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Define shortcut names for dimensions of input and kernel
        (b, c_i, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        (l_o, d_o, h_o, w_o) = (l_i + 2 * self.padding - l_k + 1,
                                d_i + 2 * self.padding - d_k + 1,
                                h_i + 2 * self.padding - h_k + 1,
                                w_i + 2 * self.padding - w_k + 1)

        # Output tensors for each 3D frame
        frame_results = l_o * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):

            for j in range(l_i):

                # Add results to this output frame
                out_frame = j - (i - l_k // 2) - (l_i - l_o) // 2
                if out_frame < 0 or out_frame >= l_o:
                    continue

                frame_conv3d = \
                    self.conv3d_layers[i](input[:, :, j, :]
                                          .view(b, c_i, d_i, h_i, w_i))

                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d

        return torch.stack(frame_results, dim=2)