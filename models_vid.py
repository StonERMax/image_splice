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


def create_temp_affinity(t=5):
    col = torch.from_numpy(np.arange(t)).float().repeat(t, 1)
    row = col.permute(1, 0)
    At = torch.exp(torch.abs(col - row))
    return At


class TGCN(nn.Module):
    """ temporal GCN """

    def __init__(self, in_feat=4 * 256, out_feat=256):
        super().__init__()
        self.in_feat = in_feat

        self.conv = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, 1), nn.BatchNorm2d(out_feat), nn.ReLU()
        )
        self.conv.apply(weights_init_normal)

    def forward(self, X):
        b, t, cin, h, w = X.shape
        At = create_temp_affinity(t).to(X.device)
        D_mhalf = torch.diag_embed(
            torch.sqrt(1.0 / (torch.sum(At, dim=-1) + 1e-8))
        )  # --> t, t
        A = torch.mm(D_mhalf, torch.mm(At, D_mhalf))  # b, t, t
        A = A.unsqueeze(0).expand(b, t, t)  # (t, t) --> (1, t, t) --> (b, t, t)

        # (b, t, c, h, w) --> (b, t, -1)
        Xr = X.reshape(b, t, -1)

        # (b, t, t) x (b, t, -1) --> (b, t, -1)
        out_inter = torch.bmm(A, Xr).reshape(b, t, cin, h, w)
        out_inter = out_inter.view(b * t, cin, h, w)
        out = self.conv(out_inter).reshape(b * t, -1, h, w)  # -1 = cout
        return out


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
    return (x - x.mean(dim=-3, keepdim=True)) / (1e-8 + x.std(dim=-3, keepdim=True))


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
    if size is not None:
        x = F.interpolate(x.unsqueeze(0), size=size, mode="bilinear").squeeze(0)

    def fn(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    torchvision.utils.save_image(fn(x), f"{name}.png")


class TemporalDetectionBranch(nn.Module):
    def __init__(self, in_cat=3 * 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_cat, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_1d = nn.Sequential(
            nn.ReplicationPad1d(1),
            nn.Conv1d(128, 64, 3),
            nn.ReLU(),
            nn.ReplicationPad1d(1),
            nn.Conv1d(64, 32, 3),
            nn.ReLU()
        )
        self.final_conv = nn.Conv1d(32, 1, 1)
        self.pad = nn.ReplicationPad1d(1)

        self.apply(weights_init_normal)

    def forward(self, xp, xq):
        b, t, c, h, w = xp.shape
        xp_conv = self.conv(xp.view(b * t, c, h, w))
        xq_conv = self.conv(xq.view(b * t, c, h, w))

        # (b*t, c, h, w) --> (b*t, 128, 1, 1)
        x_cat_p = torch.cat(
            (F.adaptive_avg_pool2d(xp_conv, (1, 1)), F.adaptive_max_pool2d(xp_conv, (1, 1))), dim=-3
        )
        x_cat_q = torch.cat(
            (F.adaptive_avg_pool2d(xq_conv, (1, 1)), F.adaptive_max_pool2d(xq_conv, (1, 1))), dim=-3
        )

        x_cat = torch.abs(x_cat_p - x_cat_q)
        x_cat = x_cat.reshape(b, t, -1).permute(0, 2, 1)  # --> (b, 128, t)
        y = self.conv_1d(x_cat)  # --> (b, c, t')
        y = F.adaptive_avg_pool1d(y, 1)  # --> (b, 128, 1)
        y = self.final_conv(y).squeeze()  # --> (b, 1, 1)
        return y


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
        # D_mhalf = torch.sqrt(torch.inverse(D))  # b, h1w1, h1w1

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

        self.encoder_p = Extractor_VGG19()
        self.encoder_q = Extractor_VGG19()

        self.corrLayer = Corr(topk=topk)

        in_cat = 896

        self.val_conv_p = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self.val_conv_q = nn.Sequential(
            nn.Conv2d(topk, 16, 3, padding=1),
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
            in_channels=in_cat, atrous_rates=[12, 24, 36]
        )

        self.head_mask_p = nn.Sequential(
            nn.Conv2d(2 * 256, 2 * 256, 1),
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
            nn.Conv2d(2 * 256, 2 * 256, 1),
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

        self.gcn_p = GCN(in_feat=256, out_feat=256)
        self.gcn_q = GCN(in_feat=256, out_feat=256)

        self.t_gcn_p = TGCN(in_feat=2 * 256, out_feat=2 * 256)
        self.t_gcn_q = TGCN(in_feat=2 * 256, out_feat=2 * 256)

        self.temporal_detection = TemporalDetectionBranch(2*256)

        # detection branch
        self.head_mask_p.apply(weights_init_normal)
        self.head_mask_q.apply(weights_init_normal)

        self.val_conv_p.apply(weights_init_normal)
        self.val_conv_q.apply(weights_init_normal)

        self.head_mask_p.apply(weights_init_normal)
        self.head_mask_q.apply(weights_init_normal)

    def forward(self, xq, xp):
        input1, input2 = xq, xp
        b, t, c, h, w = xp.shape

        xpr = xp.reshape(-1, c, h, w)
        xqr = xq.reshape(-1, c, h, w)

        xp_feat = self.encoder_p(xpr, out_size=self.hw)
        xq_feat = self.encoder_q(xqr, out_size=self.hw)

        valp, valq, indp, indq = self.corrLayer(xp_feat, xq_feat)

        # attention weight
        val_conv_p = self.val_conv_p(valp)
        val_conv_q = self.val_conv_q(valq)

        #### Mask part : M  ####
        xp_as = self.aspp_forge(xp_feat) * val_conv_p
        xq_as = self.aspp_mask(xq_feat) * val_conv_q

        # Corrensponding mask and forge
        xp_as_nl = self.gcn_p(xq_as, indp)
        xq_as_nl = self.gcn_q(xp_as, indq)

        # Final Mask
        x_cat_p = torch.cat((xp_as, xp_as_nl), dim=-3)
        x_cat_q = torch.cat((xq_as, xq_as_nl), dim=-3)

        # b*t, c, h, w --> b*t, c', h, w
        x_tgcn_p = self.t_gcn_p(x_cat_p.reshape(b, t, *x_cat_p.shape[1:]))
        x_tgcn_q = self.t_gcn_q(x_cat_q.reshape(b, t, *x_cat_q.shape[1:]))

        x_final_p = (x_cat_p + x_tgcn_p) / 2
        x_final_q = (x_cat_q + x_tgcn_q) / 2

        outp = self.head_mask_p(x_final_p)
        outq = self.head_mask_q(x_final_q)

        # final detection
        out_det = self.temporal_detection(
           x_final_p.view(b, t, *x_final_p.shape[1:]),
           x_final_q.view(b, t, *x_final_q.shape[1:])
        )  # (b, )

        outp = F.interpolate(outp, size=(h, w), mode="bilinear", align_corners=True)
        outq = F.interpolate(outq, size=(h, w), mode="bilinear", align_corners=True)

        outp = outp.reshape(b, t, *outp.shape[1:])
        outq = outq.reshape(b, t, *outq.shape[1:])

        return outq, outp, out_det

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
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
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
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


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


class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
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


def get_dmac(model_name="dmac", pretrain=True):

    if model_name == "dmac":
        from other_model import dmac

        model = dmac.DMAC_VGG(NoLabels=2, gpu_idx=0, dim=(320, 320))
        model_path = "./weights/DMAC-adv.pth"
    elif model_name == "dmvn":
        from other_model import dmvn

        model = dmvn.DMVN_VGG(NoLabels=2, gpu_idx=0, dim=(320, 320))
        model_path = "./weights/DMVN-BN.pth"
    else:
        raise AssertionError
    if pretrain:
        state_dict = torch.load(model_path, map_location="cuda:0")
        model.load_state_dict(state_dict)
    return model


class CorrDetector(nn.Module):
    def __init__(self, pool_stride=8):
        super().__init__()
        "The pooling of images needs to be researched."
        # self.img_pool = nn.AvgPool2d(pool_stride, stride=pool_stride)

        self.input_dim = 3

        "Feature extraction blocks."
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        "Detection branch."
        self.classifier_det = nn.Sequential(
            nn.Linear(128 * 10 * 10, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1),
        )

        self.apply(weights_init_normal)

    def forward(self, x, m):

        _, _, h, w = x.shape
        m = F.interpolate(m, size=(h // 2, w // 2), align_corners=True, mode="bilinear")
        x = F.interpolate(x, size=(h // 2, w // 2), align_corners=True, mode="bilinear")

        m1 = m[:, [0]]
        m2 = m[:, [1]]

        x1 = torch.mul(x, m1)
        x2 = torch.mul(x, m2)

        x1 = self.conv(x1)
        x2 = self.conv(x2)

        x1 = F.adaptive_avg_pool2d(x1, (10, 10))
        x2 = F.adaptive_avg_pool2d(x2, (10, 10))

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x12_abs = torch.abs(x1 - x2)

        x_det = self.classifier_det(x12_abs)

        return x_det


class Base_DetSegModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.in1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in2 = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.in3 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))

        self.in_reduce = nn.Conv2d(3 * 64, 64, kernel_size=(1, 1), stride=(1, 1))

        self.conv_det = nn.Sequential(
            nn.Conv2d(3 * 64, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  #
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  #
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  #
        )

        self.lin_det = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b = x.shape[0]
        x1 = self.in1(x)
        x2 = self.in2(x)
        x3 = self.in3(x)
        x_cat = torch.cat((x1, x2, x3), dim=-3)
        out_seg = self.conv_det(x_cat)

        x_seg = self.pool(out_seg)
        out_det = x_seg.view(b, -1)
        out_det = self.lin_det(out_det)

        return out_det, out_seg


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


class DetSegModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = Base_DetSegModel()
        self.aspp = models.segmentation.deeplabv3.ASPP(
            in_channels=256, atrous_rates=[12, 24, 36]
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
        )
        self.apply(weights_init_normal)

    def forward(self, x):
        b, c, h, w = x.shape
        out_det, base_seg = self.base(x)
        seg = self.head(base_seg)
        seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=True)
        return out_det, seg

    def freeze_bn(self):
        self.apply(set_bn_eval)

